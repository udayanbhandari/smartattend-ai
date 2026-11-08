import json
import sys
from pathlib import Path

import cv2
import numpy as np

from src.liveness_detection import (
    CHECKING_LIVENESS,
    LIVE_PERSON,
    SUSPECTED_SPOOF,
    UNABLE_TO_VERIFY,
    FaceLivenessDetector,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"
METADATA_FILE = DATASET_DIR / "students.json"
LEGACY_METADATA_FILE = DATASET_DIR / "student_registry.json"
REGISTERED_FACES_DIR = PROJECT_ROOT / "registered_faces"
MODEL_FILE = PROJECT_ROOT / "models" / "haarcascade_frontalface_default.xml"
# LBPH distance threshold tuned for a small local dataset with variable webcam lighting and face angle.
# Lower values are stricter; this value is intentionally higher than the original 60.0 to reduce false
# unknown results when a real registered face is present but differs slightly from the training samples.
RECOGNITION_THRESHOLD = 85.0
WINDOW_TITLE = "SmartAttend AI - Face Recognition"


class FaceRecognitionSystem:
    """Train an LBPH face recognizer and display a polished live recognition interface."""

    def __init__(self, metadata_path=METADATA_FILE, registered_faces_dir=REGISTERED_FACES_DIR):
        self.metadata_path = Path(metadata_path)
        self.registered_faces_dir = Path(registered_faces_dir)
        self.student_data = {}
        self.training_images = []
        self.training_labels = []
        self.label_to_student = {}
        self.model = cv2.face.LBPHFaceRecognizer_create()
        self.face_cascade = cv2.CascadeClassifier(str(MODEL_FILE))

        if self.face_cascade.empty():
            raise FileNotFoundError(
                "Unable to load the Haar Cascade model. "
                "Please ensure models/haarcascade_frontalface_default.xml exists."
            )

        self.camera = None
        self.camera_active = False
        self.status_message = "Ready"
        self.recognition_count = 0
        self.liveness_detector = FaceLivenessDetector()

    def load_metadata(self):
        """Load the JSON list of registered students from the latest active file."""
        metadata_files = []
        if self.metadata_path.exists():
            metadata_files.append(self.metadata_path)
        legacy_path = Path(LEGACY_METADATA_FILE)
        if legacy_path.exists() and legacy_path != self.metadata_path:
            metadata_files.append(legacy_path)

        if not metadata_files:
            print(f"Warning: metadata file not found: {self.metadata_path}")
            return []

        combined = []
        seen_ids = set()

        for metadata_file in metadata_files:
            try:
                with metadata_file.open("r", encoding="utf-8") as file:
                    raw_text = file.read()
                if raw_text.startswith("\ufeff"):
                    raw_text = raw_text.lstrip("\ufeff")
                if not raw_text.strip():
                    continue
                data = json.loads(raw_text)
                for student in data.get("students", []):
                    student_id = student.get("student_id")
                    if not student_id or student_id in seen_ids:
                        continue
                    seen_ids.add(student_id)
                    combined.append(student)
            except (json.JSONDecodeError, OSError) as exc:
                print(f"Warning: Could not read metadata file: {metadata_file}: {exc}")

        return combined

    def _extract_face(self, gray_image):
        """Return the largest face region found in the image."""
        faces = self.face_cascade.detectMultiScale(
            gray_image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
        return gray_image[y : y + h, x : x + w]

    def train(self):
        """Train the recognizer from all valid student images in the registration folders."""
        students = self.load_metadata()
        if not students:
            print("Warning: no student metadata found. Recognition cannot run without registrations.")
            return False

        training_images = []
        training_labels = []
        label_to_student = {}
        valid_students = 0

        for index, student in enumerate(students):
            student_id = student.get("student_id")
            student_name = student.get("name", student_id)
            if not student_id:
                continue

            student_dir = self.registered_faces_dir / str(student_id)
            if not student_dir.exists():
                print(f"Warning: no registration folder found for {student_id}. Skipping.")
                continue

            image_files = (
                sorted(student_dir.glob("*.jpg"))
                + sorted(student_dir.glob("*.jpeg"))
                + sorted(student_dir.glob("*.png"))
            )
            if not image_files:
                print(f"Warning: no valid images found for {student_id}. Skipping.")
                continue

            self.student_data[student_id] = student_name
            label_to_student[index] = student_id
            valid_students += 1

            for image_path in image_files:
                image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    print(f"Warning: corrupted or unreadable image skipped: {image_path}")
                    continue

                face_crop = self._extract_face(image)
                if face_crop is None:
                    # If image is already cropped, use directly
                    face_crop = image

                normalized_face = cv2.resize(face_crop, (200, 200), interpolation=cv2.INTER_LINEAR)
                training_images.append(normalized_face)
                training_labels.append(index)
                print(f"Loaded sample for {student_name} from {image_path.name}")

        if len(training_images) == 0:
            print("Warning: no usable student face images were found. Recognition cannot run.")
            return False

        self.training_images = training_images
        self.training_labels = training_labels
        self.label_to_student = label_to_student
        self.model.train(self.training_images, np.asarray(self.training_labels, dtype=np.int32))
        print(f"Training complete. Registered students loaded: {valid_students}")
        return True

    def _draw_header(self, frame, title_text, subtitle_text):
        """Draw a floating header bar for a polished interface."""
        height, width = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (width, 80), (12, 15, 20), -1)
        cv2.putText(frame, title_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, subtitle_text, (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    def _draw_status_bar(self, frame, status_text, color):
        """Draw a persistent status panel at the bottom of the frame."""
        height, width = frame.shape[:2]
        panel_height = 50
        cv2.rectangle(frame, (0, height - panel_height), (width, height), color, -1)
        cv2.putText(
            frame,
            status_text,
            (20, height - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

    def _camera_stopped_frame(self):
        """Create a placeholder frame shown while the camera is stopped."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (18, 18, 22)
        cv2.putText(frame, "SmartAttend AI", (210, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
        cv2.putText(frame, "Camera Stopped", (220, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, "Press S to Start", (220, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 255), 2)
        cv2.putText(frame, "Press Q to Quit", (220, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return frame

    def recognize_frame(self, frame):
        """Detect faces and identify known students with overlays."""
        if frame is None:
            raise ValueError("Cannot recognize from an empty frame.")

        if len(self.label_to_student) < 1:
            self.status_message = "Unknown Person"
            cv2.putText(
                frame,
                "Unknown Person",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )
            cv2.putText(
                frame,
                "Need at least 2 enrolled students for reliable recognition.",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )
            return frame, None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        if len(faces) == 0:
            self.status_message = "No Face Detected"
            return frame, None

        face_results = []
        liveness_results = self.liveness_detector.update_face_set(frame, [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces])

        for index, ((x, y, w, h), liveness_result) in enumerate(zip(faces, liveness_results)):
            face_roi = gray[y : y + h, x : x + w]
            face_roi = cv2.resize(face_roi, (200, 200), interpolation=cv2.INTER_LINEAR)

            predicted_label, lbph_distance = self.model.predict(face_roi)
            student_id = self.label_to_student.get(predicted_label, "unknown")
            is_recognized = student_id != "unknown" and lbph_distance <= RECOGNITION_THRESHOLD

            if is_recognized:
                display_name = f"{student_id} - {self.student_data.get(student_id, student_id)}"
                display_color = (0, 255, 0)
                status_text = f"Recognized: {display_name}"
                display_id = student_id
                display_label = display_name
            else:
                display_name = "Unknown Person"
                display_color = (0, 0, 255)
                status_text = "Unknown Person"
                display_id = "unknown"
                display_label = "Unknown Person"

            match_text = f"LBPH Distance={lbph_distance:.2f}"

            liveness_state = liveness_result["state"]
            liveness_message = liveness_result["message"]
            liveness_score = liveness_result["score"]

            cv2.rectangle(frame, (x, y), (x + w, y + h), display_color, 2)
            cv2.putText(
                frame,
                display_label,
                (x, y - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                display_color,
                2,
            )
            cv2.putText(
                frame,
                match_text,
                (x, y + h + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                display_color,
                1,
            )
            cv2.putText(
                frame,
                f"Liveness: {liveness_message} ({liveness_score})",
                (x, y + h + 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )

            if liveness_state in {CHECKING_LIVENESS, UNABLE_TO_VERIFY}:
                overlay_color = (0, 165, 255)
            elif liveness_state == SUSPECTED_SPOOF:
                overlay_color = (0, 0, 255)
            else:
                overlay_color = (0, 255, 0)

            cv2.putText(
                frame,
                f"Liveness: {liveness_state}",
                (x, y + h + 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                overlay_color,
                1,
            )

            face_result = {
                "student_id": display_id,
                "name": self.student_data.get(student_id, "Unknown Person") if display_id != "unknown" else "Unknown Person",
                "status": status_text,
                "is_recognized": is_recognized,
                "distance": lbph_distance,
                "liveness_state": liveness_state,
                "liveness_message": liveness_message,
                "liveness_score": liveness_score,
            }
            face_results.append(face_result)

        best_result = None
        if face_results:
            recognized_candidates = [result for result in face_results if result["is_recognized"]]
            if recognized_candidates:
                best_result = min(recognized_candidates, key=lambda result: result["distance"])
            else:
                best_result = min(face_results, key=lambda result: result["distance"])

        if best_result is None:
            self.status_message = "No Face Detected"
        elif not best_result["is_recognized"]:
            self.status_message = "Unknown Person"
        else:
            self.status_message = (
                f"Recognized: {best_result['student_id']} - {best_result['name']} | "
                f"Liveness: {best_result['liveness_message']}"
            )

        self.recognition_count = len(faces)
        return frame, best_result

    def start_camera(self):
        """Open the default webcam and return the camera object."""
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            raise RuntimeError("Unable to open webcam. Please check the camera connection and try again.")
        self.camera_active = True
        print("Webcam started successfully.")
        return self.camera

    def stop_camera(self):
        """Stop the active camera and release resources."""
        if self.camera is not None:
            self.camera.release()
        self.camera = None
        self.camera_active = False
        cv2.destroyAllWindows()
        print("Webcam stopped.")

    def run(self):
        """Run the live webcam recognition interface with start/stop controls."""
        print("SmartAttend AI - Face Recognition Interface")
        print("Commands: S = Start camera, C = Stop camera, Q = Quit")

        if not self.train():
            print("Recognition system did not load any valid training images.")
            print("Register students before running face recognition.")
            return 1

        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_TITLE, 960, 720)

        try:
            while True:
                key = cv2.waitKey(50) & 0xFF

                if key == ord('q'):
                    print("Exiting recognition interface.")
                    break

                if key == ord('s'):
                    if self.camera is None:
                        try:
                            self.start_camera()
                        except RuntimeError as exc:
                            print(f"Error: {exc}")
                            self.status_message = "Camera Error"
                            display = self._camera_stopped_frame()
                            self._draw_header(display, "SmartAttend AI", "Camera unavailable")
                            self._draw_status_bar(display, "Camera Error", (0, 0, 255))
                            cv2.imshow(WINDOW_TITLE, display)
                    else:
                        self.status_message = "Camera already running"

                if key == ord('c'):
                    self.stop_camera()

                if self.camera is not None and self.camera_active:
                    ret, frame = self.camera.read()
                    if not ret or frame is None:
                        print("Error: Failed to capture a frame from the webcam.")
                        self.status_message = "Frame Capture Error"
                        display = self._camera_stopped_frame()
                        self._draw_header(display, "SmartAttend AI", "Frame capture failed")
                        self._draw_status_bar(display, "Frame Capture Error", (0, 0, 255))
                        cv2.imshow(WINDOW_TITLE, display)
                        continue

                    processed_frame, _ = self.recognize_frame(frame.copy())
                    self._draw_header(processed_frame, "SmartAttend AI", "Live Face Recognition")
                    self._draw_status_bar(
                        processed_frame,
                        f"Status: {self.status_message} | Faces: {self.recognition_count}",
                        (0, 128, 255) if self.status_message == "No Face Detected" else (0, 160, 70),
                    )
                    cv2.imshow(WINDOW_TITLE, processed_frame)
                else:
                    display = self._camera_stopped_frame()
                    self._draw_header(display, "SmartAttend AI", "Recognition System Ready")
                    self._draw_status_bar(display, "Status: Camera Stopped | Press S to start", (60, 60, 60))
                    cv2.imshow(WINDOW_TITLE, display)

        except Exception as exc:
            print(f"Unexpected recognition interface error: {exc}")
        finally:
            if self.camera is not None:
                self.camera.release()
            cv2.destroyAllWindows()
            print("Recognition interface closed successfully.")

        return 0


def main():
    """Entry point for the polished face recognition interface."""
    try:
        system = FaceRecognitionSystem()
        return system.run()
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return 1
    except Exception as exc:
        print(f"Unexpected startup error: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
