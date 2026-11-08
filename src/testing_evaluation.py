import json
import time
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"
METADATA_FILE = DATASET_DIR / "student_registry.json"
REGISTERED_FACES_DIR = PROJECT_ROOT / "registered_faces"
REPORT_FILE = PROJECT_ROOT / "testing_report.md"


def get_metadata_status():
    """Return metadata and registration status for the project dataset."""
    status = {
        "metadata_exists": METADATA_FILE.exists(),
        "student_count": 0,
        "image_count": 0,
        "student_folders": [],
        "valid_students": [],
        "warnings": [],
    }

    if not METADATA_FILE.exists():
        status["warnings"].append("Metadata file is missing.")
        return status

    try:
        with METADATA_FILE.open("r", encoding="utf-8") as file:
            raw_text = file.read()
        if raw_text.startswith("\ufeff"):
            raw_text = raw_text.lstrip("\ufeff")
        if not raw_text.strip():
            status["warnings"].append("Metadata file is empty.")
            return status

        data = json.loads(raw_text)
        students = data.get("students", [])
        status["student_count"] = len(students)
        status["student_folders"] = [student.get("student_id") for student in students if student.get("student_id")]
        status["valid_students"] = list(status["student_folders"])

        if not students:
            status["warnings"].append("No students are currently registered in the dataset.")

        for student in students:
            student_id = student.get("student_id")
            if not student_id:
                continue
            folder = REGISTERED_FACES_DIR / str(student_id)
            if not folder.exists():
                status["warnings"].append(f"Missing folder for registered student: {student_id}")
                continue
            images = list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg")) + list(folder.glob("*.png"))
            status["image_count"] += len(images)
            if len(images) == 0:
                status["warnings"].append(f"No valid face images found in folder: {student_id}")
    except json.JSONDecodeError:
        status["warnings"].append("Metadata file is malformed JSON.")

    return status


def generate_report():
    """Create a testing report using actual observed project state and clear manual test steps."""
    start_time = time.time()
    dataset_status = get_metadata_status()
    elapsed_seconds = time.time() - start_time

    observed_lines = []
    observed_lines.append("# SmartAttend AI Testing and Evaluation Report")
    observed_lines.append("")
    observed_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    observed_lines.append("")
    observed_lines.append("## Actual observed results")
    observed_lines.append("")
    observed_lines.append("- Dataset metadata file exists: " + ("Yes" if dataset_status["metadata_exists"] else "No"))
    observed_lines.append(f"- Registered students found: {dataset_status['student_count']}")
    observed_lines.append(f"- Valid face images found: {dataset_status['image_count']}")
    observed_lines.append(f"- Validation duration: {elapsed_seconds:.3f} seconds")

    if dataset_status["warnings"]:
        observed_lines.append("")
        observed_lines.append("Warnings:")
        for warning in dataset_status["warnings"]:
            observed_lines.append(f"- {warning}")
    else:
        observed_lines.append("")
        observed_lines.append("Warnings: None")

    observed_lines.append("")
    observed_lines.append("## Manual evaluation checklist")
    observed_lines.append("")
    observed_lines.append("1. Register four students with clear frontal views and consistent lighting.")
    observed_lines.append("2. Test each registered student individually at normal distance (about 0.5 to 1.0 meter).")
    observed_lines.append("3. Test each registered student with different head angles: straight, slight left, slight right, slight up, slight down.")
    observed_lines.append("4. Test under normal indoor lighting, dim lighting, and strong side lighting.")
    observed_lines.append("5. Test unregistered person detection: place an unknown face in front of the camera.")
    observed_lines.append("6. Test multiple people in the same frame to confirm only one or multiple faces are handled correctly.")
    observed_lines.append("7. Record false acceptance (unknown person accepted as known) and false rejection (known person rejected).")
    observed_lines.append("8. Measure response time by estimating time between frame capture and display update using the webcam loop and frame counter.")

    observed_lines.append("")
    observed_lines.append("## Recommended scoring method")
    observed_lines.append("")
    observed_lines.append("- True positive: registered student correctly recognized")
    observed_lines.append("- False positive: unknown person identified as a student")
    observed_lines.append("- False negative: registered student not recognized")
    observed_lines.append("- Precision = true positive / (true positive + false positive)")
    observed_lines.append("- Recall = true positive / (true positive + false negative)")
    observed_lines.append("- Average recognition latency = total evaluation time / number of recognized frames")

    observed_lines.append("")
    observed_lines.append("## Improvement recommendations")
    observed_lines.append("")
    observed_lines.append("- Capture more varied training images per student.")
    observed_lines.append("- Include different lighting conditions during registration.")
    observed_lines.append("- Improve face alignment by centering the face before capture.")
    observed_lines.append("- Use a stronger model such as OpenCV FaceRecognizer with better feature sets or an external deep-learning model later.")
    observed_lines.append("- Add confidence thresholds and a rejection queue for uncertain matches.")
    observed_lines.append("- Add a preview panel to help users align their face before capture.")

    report_text = "\n".join(observed_lines)
    REPORT_FILE.write_text(report_text, encoding="utf-8")
    return report_text


def main():
    """Print and save the testing report."""
    report = generate_report()
    print(report)
    print(f"\nTesting report saved to: {REPORT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
