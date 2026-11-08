import json
import sys
import time
import tracemalloc
import unittest
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DATASET_DIR = PROJECT_ROOT / "dataset"
METADATA_FILE = DATASET_DIR / "students.json"
REGISTERED_FACES_DIR = PROJECT_ROOT / "registered_faces"
MODEL_FILE = PROJECT_ROOT / "models" / "haarcascade_frontalface_default.xml"


def load_metadata():
    if not METADATA_FILE.exists():
        raise FileNotFoundError(f"Missing metadata file: {METADATA_FILE}")
    with METADATA_FILE.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return data.get("students", [])


def collect_student_images():
    students = load_metadata()
    images_by_student = []
    for student in students:
        student_id = student.get("student_id")
        if not student_id:
            continue
        folder = REGISTERED_FACES_DIR / str(student_id)
        if not folder.exists():
            continue
        files = sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.jpeg")) + sorted(folder.glob("*.png"))
        if not files:
            continue
        images_by_student.append({
            "student_id": student_id,
            "name": student.get("name", student_id),
            "images": files,
        })
    return images_by_student


def detect_face_crop(gray_image):
    detector = cv2.CascadeClassifier(str(MODEL_FILE))
    if detector.empty():
        raise FileNotFoundError(f"Face cascade not found: {MODEL_FILE}")
    faces = detector.detectMultiScale(
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


def build_model(train_data):
    model = cv2.face.LBPHFaceRecognizer_create()
    face_samples = []
    labels = []
    label_to_student = {}

    for label_index, item in enumerate(train_data):
        student_id = item["student_id"]
        label_to_student[label_index] = student_id
        for image_path in item["images"]:
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            crop = detect_face_crop(image)
            if crop is None:
                continue
            face_samples.append(cv2.resize(crop, (200, 200), interpolation=cv2.INTER_LINEAR))
            labels.append(label_index)

    if not face_samples:
        raise ValueError("No valid training face samples could be prepared.")

    model.train(face_samples, np.asarray(labels, dtype=np.int32))
    return model, label_to_student


def evaluate_recognition():
    student_images = collect_student_images()
    if not student_images:
        return {
            "registered_students": 0,
            "total_images": 0,
            "accuracy": 0.0,
            "correct": 0,
            "wrong": 0,
            "unknown": 0,
            "details": [],
            "training_time_ms": 0.0,
            "inference_time_ms": 0.0,
        }

    train_data = []
    test_data = []
    for student in student_images:
        images = student["images"]
        split_index = max(1, int(len(images) * 0.8))
        train_data.append({
            "student_id": student["student_id"],
            "name": student["name"],
            "images": images[:split_index],
        })
        test_data.append({
            "student_id": student["student_id"],
            "name": student["name"],
            "images": images[split_index:],
        })

    start = time.perf_counter()
    model, label_to_student = build_model(train_data)
    training_time_ms = (time.perf_counter() - start) * 1000.0

    total = 0
    correct = 0
    wrong = 0
    unknown = 0
    details = []

    inference_times = []
    for student in test_data:
        for image_path in student["images"]:
            total += 1
            frame = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if frame is None:
                continue
            face_crop = detect_face_crop(frame)
            if face_crop is None:
                unknown += 1
                details.append({
                    "actual": student["student_id"],
                    "predicted": "unknown",
                    "confidence": None,
                    "status": "face_not_detected",
                })
                continue

            normalized = cv2.resize(face_crop, (200, 200), interpolation=cv2.INTER_LINEAR)
            start_infer = time.perf_counter()
            label, confidence = model.predict(normalized)
            inference_times.append((time.perf_counter() - start_infer) * 1000.0)

            predicted_id = label_to_student.get(label, "unknown")
            match = predicted_id == student["student_id"] and confidence <= 60.0
            if match:
                correct += 1
                details.append({
                    "actual": student["student_id"],
                    "predicted": predicted_id,
                    "confidence": round(float(confidence), 2),
                    "status": "correct",
                })
            else:
                wrong += 1
                details.append({
                    "actual": student["student_id"],
                    "predicted": predicted_id,
                    "confidence": round(float(confidence), 2),
                    "status": "incorrect",
                })

    accuracy = (correct / total * 100.0) if total else 0.0
    average_inference = float(np.mean(inference_times)) if inference_times else 0.0
    fps_estimate = 1000.0 / average_inference if average_inference > 0 else 0.0

    return {
        "registered_students": len(student_images),
        "total_images": sum(len(student["images"]) for student in student_images),
        "accuracy": round(accuracy, 2),
        "correct": correct,
        "wrong": wrong,
        "unknown": unknown,
        "details": details,
        "training_time_ms": round(training_time_ms, 2),
        "inference_time_ms": round(average_inference, 2),
        "fps_estimate": round(fps_estimate, 2),
        "test_images": total,
    }


def run_liveness_tests():
    tests_module = __import__("tests.test_liveness_detection", fromlist=["TestFaceLivenessDetector"])
    suite = unittest.defaultTestLoader.loadTestsFromModule(tests_module)
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    return {
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "success": result.wasSuccessful(),
    }


def benchmark_memory_and_cpu():
    tracemalloc.start()
    start_cpu = time.process_time()
    start_wall = time.perf_counter()

    student_images = collect_student_images()
    if not student_images:
        return {"python_memory_peak_mb": 0.0, "cpu_seconds": 0.0, "wall_seconds": 0.0}

    count = 0
    for student in student_images:
        for image_path in student["images"][:2]:
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            _ = detect_face_crop(image)
            count += 1

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "python_memory_peak_mb": round(peak / (1024 * 1024), 2),
        "cpu_seconds": round(time.process_time() - start_cpu, 4),
        "wall_seconds": round(time.perf_counter() - start_wall, 4),
        "processed_frames": count,
    }


def build_report():
    recognition_summary = evaluate_recognition()
    liveness_results = run_liveness_tests()
    resource_summary = benchmark_memory_and_cpu()

    report_lines = [
        "# SmartAttend-AI Phase 9: Model Evaluation and Performance Optimization",
        "",
        "## Scope",
        "This report evaluates the current local OpenCV pipeline using the registered dataset and the existing liveness unit tests. It does not claim full real-world spoof resistance or webcam accuracy beyond the measured batch checks performed in this environment.",
        "",
        "## Dataset Summary",
        f"- Registered students: {recognition_summary['registered_students']}",
        f"- Total image files in the current dataset: {recognition_summary['total_images']}",
        f"- Test images used for evaluation: {recognition_summary['test_images']}",
        "",
        "## Recognition Benchmark (Measured)",
        f"- Correct predictions: {recognition_summary['correct']}",
        f"- Incorrect predictions: {recognition_summary['wrong']}",
        f"- Unknown or undetected cases: {recognition_summary['unknown']}",
        f"- Recognition accuracy: {recognition_summary['accuracy']}%",
        f"- Training time: {recognition_summary['training_time_ms']} ms",
        f"- Average inference time per face: {recognition_summary['inference_time_ms']} ms",
        f"- Estimated FPS: {recognition_summary['fps_estimate']} FPS",
        "",
        "## Liveness Unit Tests (Measured)",
        f"- Tests run: {liveness_results['tests_run']}",
        f"- Failures: {liveness_results['failures']}",
        f"- Errors: {liveness_results['errors']}",
        f"- Success: {liveness_results['success']}",
        "",
        "## Resource Snapshot (Measured)",
        f"- Process CPU time: {resource_summary['cpu_seconds']} s",
        f"- Wall-clock time: {resource_summary['wall_seconds']} s",
        f"- Peak Python memory: {resource_summary['python_memory_peak_mb']} MB",
        f"- Frames processed in the benchmark: {resource_summary['processed_frames']}",
        "",
        "## Actual Observations",
        "- The current project has a small, local registered dataset and is evaluated with a hold-out validation split where possible.",
        "- The recognition benchmark is measured from the real dataset files, not from synthetic assumptions.",
        "- The liveness tests validate logic transitions and conservative handling, but they do not prove real-world spoof resistance against printed photos, phone screens, or replayed videos.",
        "- No live webcam accuracy percentage is claimed because no real-world webcam benchmark was run in this environment.",
        "",
        "## Bottlenecks and Optimization Notes",
        "- The most expensive part of the pipeline is the repeated face detection and LBPH prediction on each frame.",
        "- A webcam pipeline should avoid running expensive operations on every frame when there is no detected face.",
        "- The implementation already keeps detection and recognition separate, which is a good design choice for the current project.",
        "- No model replacement or large dependency change was made because the project explicitly requires preserving the current Haar Cascade + LBPH design.",
        "",
        "## Recommendations",
        "- Capture more training images per student and keep lighting, distance, and angle consistent.",
        "- Use a larger and more diverse evaluation set if a classroom demonstration is planned.",
        "- Add a dedicated evaluation script for repeated on-device measurements when hardware is available.",
        "- For stronger anti-spoof performance, consider a dedicated pretrained anti-spoof model in a future phase, but only after validation and with explicit dependency management.",
        "",
        "## Limitations",
        "- This is not a calibrated anti-spoof benchmark.",
        "- Real-world webcam variability, lighting, and camera quality can change the measured performance significantly.",
        "- The current evaluation is intentionally conservative and data-driven.",
    ]

    return "\n".join(report_lines)


if __name__ == "__main__":
    report = build_report()
    output_path = PROJECT_ROOT / "evaluation_report.md"
    output_path.write_text(report, encoding="utf-8")
    print(report)
