import json
import sys
from datetime import datetime
from pathlib import Path

import cv2

from src.face_detection import FaceDetector


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"
REGISTERED_FACES_DIR = PROJECT_ROOT / "registered_faces"
LEGACY_METADATA_FILE = DATASET_DIR / "student_registry.json"
METADATA_FILE = DATASET_DIR / "students.json"
CAPTURE_COUNT = 10


class FaceRegistrationManager:
    """Capture a large set of face images for one student using the webcam."""

    def __init__(self):
        self.dataset_dir = DATASET_DIR
        self.registered_faces_dir = REGISTERED_FACES_DIR
        self.metadata_file = METADATA_FILE
        self.legacy_metadata_file = LEGACY_METADATA_FILE
        self.detector = FaceDetector()
        self._prepare_storage()

    def _prepare_storage(self):
        """Create the required storage directories and metadata files."""
        self.dataset_dir.mkdir(exist_ok=True)
        self.registered_faces_dir.mkdir(exist_ok=True)

        if not self.metadata_file.exists():
            self._write_metadata([])
        if not self.legacy_metadata_file.exists():
            self.legacy_metadata_file.write_text('{"students": []}', encoding='utf-8')

    def _write_metadata(self, students):
        """Write the enrollment metadata to both the recommended and legacy JSON file names."""
        payload = {"students": students}
        self.metadata_file.write_text(json.dumps(payload, indent=2), encoding='utf-8')
        self.legacy_metadata_file.write_text(json.dumps(payload, indent=2), encoding='utf-8')

    def load_metadata(self):
        """Load all registered students from the supported metadata files."""
        metadata_files = []
        if self.metadata_file.exists():
            metadata_files.append(self.metadata_file)
        if self.legacy_metadata_file.exists():
            metadata_files.append(self.legacy_metadata_file)

        combined = []
        seen_ids = set()

        for metadata_path in metadata_files:
            try:
                raw_text = metadata_path.read_text(encoding='utf-8')
                if raw_text.startswith('\ufeff'):
                    raw_text = raw_text.lstrip('\ufeff')
                if not raw_text.strip():
                    continue
                data = json.loads(raw_text)
                students = data.get("students", [])
                for student in students:
                    student_id = student.get("student_id")
                    if not student_id or student_id in seen_ids:
                        continue
                    seen_ids.add(student_id)
                    combined.append(student)
            except json.JSONDecodeError:
                print(f"Warning: metadata file is invalid JSON: {metadata_path}")

        return combined

    def normalize_student_id(self, student_id):
        """Validate and normalize the student ID format."""
        student_id = str(student_id).strip()
        if not student_id:
            raise ValueError("Student ID cannot be empty.")

        student_id = student_id.strip().lower()
        if student_id.startswith("student_"):
            suffix = student_id.split("student_", 1)[1]
            if not suffix:
                raise ValueError("Student ID must include a valid value after 'student_'.")
            return f"student_{suffix}"

        if student_id.isdigit():
            return f"student_{int(student_id):03d}"

        if any(char.isspace() for char in student_id):
            raise ValueError("Student ID cannot contain spaces. Please use a format like student_001.")

        if not student_id.replace("_", "").isalnum():
            raise ValueError("Student ID can only contain letters, numbers, and underscores.")

        return f"student_{student_id}"

    def is_student_registered(self, student_id):
        """Check whether a student already exists in a registered folder or metadata."""
        metadata = self.load_metadata()
        if any(item.get("student_id") == student_id for item in metadata):
            return True

        student_dir = self.registered_faces_dir / student_id
        if not student_dir.exists():
            return False
        return any(student_dir.iterdir())

    def register_student(self, student_id, student_name):
        """Capture up to 50 face images for a student and save them locally."""
        normalized_id = self.normalize_student_id(student_id)
        clean_name = student_name.strip()

        if not clean_name:
            raise ValueError("Student name cannot be empty.")

        if self.is_student_registered(normalized_id):
            raise FileExistsError(
                f"Student '{normalized_id}' is already registered. "
                "Choose a different Student ID or delete the existing folder before registering again."
            )

        student_dir = self.registered_faces_dir / normalized_id
        student_dir.mkdir(parents=True, exist_ok=True)

        existing_metadata = self.load_metadata()
        config = {
            "student_id": normalized_id,
            "name": clean_name,
            "registered_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "images": [],
        }

        print(f"\nStudent registration started for: {clean_name} ({normalized_id})")
        print("Instructions:")
        print("  1. Stand in front of the webcam.")
        print("  2. Keep one face visible in the camera frame.")
        print("  3. Press SPACE to capture one image.")
        print(f"  4. Capture up to {CAPTURE_COUNT} useful images.")
        print("  5. Press Q to cancel registration.\n")

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Unable to open webcam for student registration.")

        captured_count = 0

        try:
            while captured_count < CAPTURE_COUNT:
                ret, frame = cap.read()
                if not ret or frame is None:
                    raise RuntimeError("Failed to capture a frame from the webcam.")

                frame_with_boxes = self.detector.detect_faces(frame.copy())
                face_count = len(self.detector.face_cascade.detectMultiScale(
                    cv2.cvtColor(frame_with_boxes, cv2.COLOR_BGR2GRAY),
                    1.1,
                    5,
                    0,
                    (30, 30),
                ))

                cv2.putText(
                    frame_with_boxes,
                    f"Captured: {captured_count}/{CAPTURE_COUNT}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                if face_count == 0:
                    cv2.putText(
                        frame_with_boxes,
                        "No face detected. Please face the camera.",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                    )
                elif face_count > 1:
                    cv2.putText(
                        frame_with_boxes,
                        "Multiple faces detected. Keep only one person.",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                    )

                cv2.imshow(f"SmartAttend AI - Register: {normalized_id}", frame_with_boxes)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Registration cancelled by user.")
                    return None

                if key == ord(' ') and face_count == 1:
                    image_number = captured_count + 1
                    image_name = f"face_{image_number:03d}.jpg"
                    image_path = student_dir / image_name
                    saved = cv2.imwrite(str(image_path), frame)

                    if not saved:
                        print(f"Warning: Could not save image {image_name}. Retrying...")
                        continue

                    config["images"].append(str(image_path.relative_to(PROJECT_ROOT)))
                    captured_count += 1
                    print(f"Saved {image_name} ({captured_count}/{CAPTURE_COUNT})")

            config["image_count"] = len(config["images"])
            existing_metadata.append(config)
            self._write_metadata(existing_metadata)
            print(f"Registration complete for {clean_name} ({normalized_id}).")
            print(f"Saved {len(config['images'])} images to: {student_dir}")
            return config

        finally:
            cap.release()
            cv2.destroyAllWindows()


def main():
    """Run the student registration flow from the terminal."""
    manager = FaceRegistrationManager()

    try:
        student_id = input("Enter Student ID (example: student_001): ").strip()
        student_name = input("Enter Student Name: ").strip()

        manager.register_student(student_id, student_name)
    except (ValueError, FileExistsError, RuntimeError) as exc:
        print(f"Registration error: {exc}")
        return 1
    except KeyboardInterrupt:
        print("\nRegistration stopped by user.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
