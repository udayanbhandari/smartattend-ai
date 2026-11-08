import base64
import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from src.face_detection import FaceDetector
from src.face_recognition import FaceRecognitionSystem, RECOGNITION_THRESHOLD
from src.liveness_detection import FaceLivenessDetector
from src.student_management import StudentManager

# Initialize paths
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
REGISTERED_FACES_DIR = BASE_DIR / "registered_faces"
MODELS_DIR = BASE_DIR / "models"
TEMPLATES_DIR = BASE_DIR / "templates"

# Create directories if needed
DATASET_DIR.mkdir(parents=True, exist_ok=True)
REGISTERED_FACES_DIR.mkdir(parents=True, exist_ok=True)

# App instance
app = FastAPI(
    title="SmartAttend AI - Face Recognition & Student Management",
    description="Web service for face recognition attendance and student enrollment.",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount registered faces as static files for thumbnail previews
app.mount("/registered_faces", StaticFiles(directory=str(REGISTERED_FACES_DIR)), name="registered_faces")

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Core systems
student_manager = StudentManager()
detector = FaceDetector()
recognition_system = FaceRecognitionSystem()

# Perform initial training
print("Performing initial face recognizer training...")
recognition_system.train()


# Pydantic schemas
class RegisterStudentRequest(BaseModel):
    student_id: str
    name: str
    images: List[str]  # List of base64 data URLs


class RecognizeRequest(BaseModel):
    image: str  # Single base64 data URL


def decode_base64_image(data_url: str) -> np.ndarray:
    """Decode a base64 image data URL into an OpenCV BGR numpy array."""
    if not data_url:
        raise ValueError("Image data is empty.")

    if "," in data_url:
        _, encoded = data_url.split(",", 1)
    else:
        encoded = data_url

    image_bytes = base64.b64decode(encoded)
    np_arr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Failed to decode image from base64 data.")
    return frame


@app.get("/", response_class=HTMLResponse)
async def index_page(request: Request):
    """Serve the main student management and face recognition dashboard."""
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/api/health")
async def health_check():
    """Health check endpoint for Render/hosting monitoring."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "registered_students": len(student_manager.list_students()),
    }


@app.get("/api/students")
async def get_students():
    """Get all enrolled students with metadata and thumbnail links."""
    students = student_manager.list_students()
    enriched = []

    for s in students:
        s_id = s.get("student_id")
        student_folder = REGISTERED_FACES_DIR / s_id
        thumbnail = None

        if student_folder.exists():
            photos = sorted(student_folder.glob("*.jpg")) + sorted(student_folder.glob("*.png"))
            if photos:
                # Relative URL to the static route
                thumbnail = f"/registered_faces/{s_id}/{photos[0].name}"

        enriched.append({
            "student_id": s_id,
            "name": s.get("name", "Unknown"),
            "registered_at": s.get("registered_at", "N/A"),
            "image_count": s.get("image_count", len(s.get("images", []))),
            "thumbnail": thumbnail,
        })

    return {"status": "success", "count": len(enriched), "students": enriched}


@app.post("/api/students/register")
async def register_student(payload: RegisterStudentRequest):
    """Enroll a new student using captured webcam images from the web client."""
    try:
        normalized_id = student_manager.normalize_student_id(payload.student_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    clean_name = payload.name.strip()
    if not clean_name:
        raise HTTPException(status_code=400, detail="Student name cannot be empty.")

    if not payload.images or len(payload.images) < 1:
        raise HTTPException(status_code=400, detail="At least 1 face capture is required for registration.")

    # Check duplicate
    existing = student_manager.get_student(normalized_id)
    if existing:
        raise HTTPException(status_code=409, detail=f"Student ID '{normalized_id}' is already registered.")

    student_folder = REGISTERED_FACES_DIR / normalized_id
    student_folder.mkdir(parents=True, exist_ok=True)

    saved_images = []
    skipped_count = 0

    for idx, img_b64 in enumerate(payload.images):
        try:
            bgr_frame = decode_base64_image(img_b64)
            gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
            
            # Detect face
            faces = detector.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )

            if len(faces) == 0:
                # If detector didn't find face with strict params, try less strict
                faces = detector.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20)
                )

            if len(faces) > 0:
                # Crop largest face with slight margin
                x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
                face_crop = bgr_frame[y : y + h, x : x + w]
            else:
                # Fallback to entire frame if already cropped by frontend
                face_crop = bgr_frame

            filename = f"face_{idx + 1:03d}.jpg"
            save_path = student_folder / filename
            cv2.imwrite(str(save_path), face_crop)
            saved_images.append(str(Path("registered_faces") / normalized_id / filename))
        except Exception as e:
            print(f"Warning: failed saving sample {idx}: {e}")
            skipped_count += 1

    if not saved_images:
        shutil.rmtree(student_folder, ignore_errors=True)
        raise HTTPException(status_code=400, detail="No usable faces could be extracted from the submitted photos.")

    # Update metadata
    current_students = student_manager.list_students()
    new_entry = {
        "student_id": normalized_id,
        "name": clean_name,
        "registered_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "images": saved_images,
        "image_count": len(saved_images),
    }
    current_students.append(new_entry)
    student_manager._write_metadata(current_students)

    # Retrain recognizer in memory
    trained = recognition_system.train()

    return {
        "status": "success",
        "message": f"Successfully enrolled student {normalized_id}",
        "student": new_entry,
        "model_retrained": trained,
        "saved_samples": len(saved_images),
    }


@app.delete("/api/students/{student_id}")
async def delete_student(student_id: str):
    """Delete a student and their face images, then retrain the model."""
    try:
        deleted = student_manager.delete_student(student_id)
        # Retrain model
        recognition_system.train()
        return {
            "status": "success",
            "message": f"Deleted student {student_id}",
            "student": deleted,
        }
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/recognize")
async def recognize_face(payload: RecognizeRequest):
    """Run face detection, LBPH recognition, and liveness checks on a submitted frame."""
    try:
        frame = decode_base64_image(payload.image)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {exc}")

    frame_h, frame_w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector.face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    if len(faces) == 0:
        return {
            "detected": False,
            "faces_count": 0,
            "faces": [],
            "status": "No Face Detected",
        }

    face_boxes = [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]
    liveness_results = recognition_system.liveness_detector.update_face_set(frame, face_boxes)

    detected_faces = []

    for (x, y, w, h), liveness_res in zip(faces, liveness_results):
        face_roi = gray[y : y + h, x : x + w]
        if face_roi.size == 0:
            continue

        face_roi = cv2.resize(face_roi, (200, 200), interpolation=cv2.INTER_LINEAR)

        is_recognized = False
        student_id = "unknown"
        student_name = "Unknown Person"
        lbph_distance = 999.0

        if len(recognition_system.label_to_student) >= 2:
            try:
                predicted_label, lbph_dist = recognition_system.model.predict(face_roi)
                lbph_distance = round(float(lbph_dist), 2)
                candidate_id = recognition_system.label_to_student.get(predicted_label, "unknown")
                if candidate_id != "unknown" and lbph_distance <= RECOGNITION_THRESHOLD:
                    student_id = candidate_id
                    student_name = recognition_system.student_data.get(student_id, student_id)
                    is_recognized = True
            except Exception as e:
                print(f"Prediction error: {e}")

        liveness_state = liveness_res.get("state", "UNABLE_TO_VERIFY")
        liveness_message = liveness_res.get("message", "Unable to verify")
        liveness_score = liveness_res.get("score", 0.0)

        detected_faces.append({
            "box": {
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h),
                "norm_x": round(float(x) / float(frame_w), 4),
                "norm_y": round(float(y) / float(frame_h), 4),
                "norm_w": round(float(w) / float(frame_w), 4),
                "norm_h": round(float(h) / float(frame_h), 4),
            },
            "is_recognized": is_recognized,
            "student_id": student_id,
            "name": student_name,
            "distance": lbph_distance,
            "confidence": round(max(0.0, min(100.0, 100.0 - (lbph_distance / RECOGNITION_THRESHOLD * 50.0))), 1) if is_recognized else 0.0,
            "liveness_state": liveness_state,
            "liveness_message": liveness_message,
            "liveness_score": liveness_score,
        })

    # Pick best recognized result
    best_result = None
    if detected_faces:
        recognized_candidates = [f for f in detected_faces if f["is_recognized"]]
        if recognized_candidates:
            best_result = min(recognized_candidates, key=lambda f: f["distance"])
        else:
            best_result = detected_faces[0]

    return {
        "detected": True,
        "faces_count": len(detected_faces),
        "faces": detected_faces,
        "best_result": best_result,
        "status": f"Recognized {best_result['name']}" if best_result and best_result["is_recognized"] else "Unknown Face",
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
