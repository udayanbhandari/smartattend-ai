import json
import shutil
import sys
from pathlib import Path

from src.face_registration import FaceRegistrationManager

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"
METADATA_FILE = DATASET_DIR / "students.json"
LEGACY_METADATA_FILE = DATASET_DIR / "student_registry.json"
REGISTERED_FACES_DIR = PROJECT_ROOT / "registered_faces"


class StudentManager:
    """Manage the local student registry without altering the recognition logic."""

    def __init__(self, metadata_path=METADATA_FILE, legacy_metadata_path=LEGACY_METADATA_FILE, registered_faces_dir=REGISTERED_FACES_DIR):
        self.metadata_path = Path(metadata_path)
        self.legacy_metadata_path = Path(legacy_metadata_path)
        self.registered_faces_dir = Path(registered_faces_dir)
        self.dataset_dir = self.metadata_path.parent
        self.dataset_dir.mkdir(exist_ok=True, parents=True)
        self.registered_faces_dir.mkdir(exist_ok=True, parents=True)

        if not self.metadata_path.exists():
            self._write_metadata([])
        if not self.legacy_metadata_path.exists():
            self.legacy_metadata_path.write_text('{"students": []}', encoding='utf-8')

    def _write_metadata(self, students):
        payload = {"students": students}
        self.metadata_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
        self.legacy_metadata_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')

    def _read_metadata(self):
        metadata_files = []
        if self.metadata_path.exists():
            metadata_files.append(self.metadata_path)
        if self.legacy_metadata_path.exists() and self.legacy_metadata_path != self.metadata_path:
            metadata_files.append(self.legacy_metadata_path)

        combined = []
        seen = set()

        for file_path in metadata_files:
            try:
                text = file_path.read_text(encoding='utf-8')
                if text.startswith('\ufeff'):
                    text = text.lstrip('\ufeff')
                if not text.strip():
                    continue
                data = json.loads(text)
                students = data.get("students", [])
                for item in students:
                    student_id = item.get("student_id")
                    if not student_id or student_id in seen:
                        continue
                    seen.add(student_id)
                    combined.append(item)
            except (json.JSONDecodeError, OSError):
                continue
        return combined

    def normalize_student_id(self, student_id):
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
            raise ValueError("Student ID cannot contain spaces. Use a format like student_001.")

        if not student_id.replace("_", "").isalnum():
            raise ValueError("Student ID can only contain letters, numbers, and underscores.")

        return f"student_{student_id}"

    def list_students(self):
        return sorted(self._read_metadata(), key=lambda item: item.get("student_id", ""))

    def search_students(self, query):
        query = str(query).strip().lower()
        if not query:
            return self.list_students()
        matches = []
        for student in self.list_students():
            student_id = str(student.get("student_id", "")).lower()
            student_name = str(student.get("name", "")).lower()
            if query in student_id or query in student_name:
                matches.append(student)
        return matches

    def get_student(self, student_id):
        normalized = self.normalize_student_id(student_id)
        for student in self.list_students():
            if student.get("student_id") == normalized:
                return student
        return None

    def student_folder(self, student_id):
        return self.registered_faces_dir / self.normalize_student_id(student_id)

    def update_student(self, student_id, new_name=None, new_student_id=None):
        current = self.get_student(student_id)
        if current is None:
            raise ValueError(f"Student '{student_id}' was not found.")

        old_id = current["student_id"]
        updated = dict(current)

        if new_id_value := (new_student_id or "").strip():
            candidate_id = self.normalize_student_id(new_id_value)
            if candidate_id != old_id and self.get_student(candidate_id) is not None:
                raise ValueError(f"Student ID '{candidate_id}' already exists.")
            updated["student_id"] = candidate_id

        if new_name is not None:
            name_value = str(new_name).strip()
            if not name_value:
                raise ValueError("Student name cannot be empty.")
            updated["name"] = name_value

        students = self.list_students()
        for index, student in enumerate(students):
            if student.get("student_id") == old_id:
                students[index] = updated
                break

        self._write_metadata(students)

        if updated.get("student_id") != old_id:
            old_dir = self.registered_faces_dir / old_id
            new_dir = self.registered_faces_dir / updated["student_id"]
            if old_dir.exists() and not new_dir.exists():
                old_dir.rename(new_dir)

        return updated

    def delete_student(self, student_id):
        student = self.get_student(student_id)
        if student is None:
            raise ValueError(f"Student '{student_id}' was not found.")

        student_id = student["student_id"]
        students = [entry for entry in self.list_students() if entry.get("student_id") != student_id]
        self._write_metadata(students)

        folder = self.registered_faces_dir / student_id
        if folder.exists():
            shutil.rmtree(folder)

        return student

    def display_students(self):
        students = self.list_students()
        if not students:
            print("No students are currently registered.")
            return []

        print("\nRegistered students:")
        for student in students:
            student_id = student.get("student_id", "unknown")
            name = student.get("name", "Unknown")
            images = student.get("images", [])
            print(f"- {student_id} | {name} | {len(images)} image(s)")
        return students

    def display_student_detail(self, student_id):
        student = self.get_student(student_id)
        if student is None:
            raise ValueError(f"Student '{student_id}' was not found.")

        folder = self.registered_faces_dir / student["student_id"]
        image_count = len([p for p in folder.glob("*") if p.is_file()]) if folder.exists() else 0
        print("\nStudent details:")
        print(f"Student ID: {student.get('student_id')}")
        print(f"Name: {student.get('name')}")
        print(f"Registration time: {student.get('registered_at', 'Not available')}")
        print(f"Stored image count: {image_count}")
        print(f"Folder: {folder}")
        return student

    def register_student_from_cli(self):
        manager = FaceRegistrationManager()
        student_id = input("Enter Student ID (example: student_001): ").strip()
        student_name = input("Enter Student Name: ").strip()
        try:
            config = manager.register_student(student_id, student_name)
            if config is None:
                print("Registration cancelled.")
            else:
                print(f"Registration successful for {config['student_id']}.")
        except (ValueError, FileExistsError, RuntimeError) as exc:
            print(f"Registration error: {exc}")

    def _prompt_choice(self, options):
        print("\nStudent management menu:")
        for index, option in enumerate(options, start=1):
            print(f"{index}. {option}")
        print("0. Back to main menu")

        while True:
            choice = input("Select an option: ").strip()
            if choice.isdigit() and 0 <= int(choice) <= len(options):
                return int(choice)
            print("Invalid selection. Please choose a valid option.")

    def menu(self):
        options = [
            "Register new student",
            "List all students",
            "Search student",
            "View student details",
            "Update student",
            "Delete student",
            "Run recognition",
        ]

        while True:
            choice = self._prompt_choice(options)
            if choice == 0:
                return 0
            if choice == 1:
                self.register_student_from_cli()
            elif choice == 2:
                self.display_students()
            elif choice == 3:
                query = input("Enter student ID or name to search: ").strip()
                matches = self.search_students(query)
                if not matches:
                    print("No matching students found.")
                else:
                    for student in matches:
                        print(f"- {student.get('student_id')} | {student.get('name')}")
            elif choice == 4:
                student_id = input("Enter Student ID to view: ").strip()
                try:
                    self.display_student_detail(student_id)
                except ValueError as exc:
                    print(exc)
            elif choice == 5:
                student_id = input("Enter Student ID to update: ").strip()
                try:
                    current = self.get_student(student_id)
                    if current is None:
                        raise ValueError(f"Student '{student_id}' was not found.")
                    new_name = input(f"Enter new name [{current.get('name')}]: ").strip() or current.get("name")
                    new_id = input(f"Enter new Student ID [{current.get('student_id')}]: ").strip() or current.get("student_id")
                    updated = self.update_student(current["student_id"], new_name=new_name, new_student_id=new_id)
                    print(f"Updated student: {updated['student_id']} | {updated['name']}")
                except ValueError as exc:
                    print(exc)
            elif choice == 6:
                student_id = input("Enter Student ID to delete: ").strip()
                try:
                    current = self.get_student(student_id)
                    if current is None:
                        raise ValueError(f"Student '{student_id}' was not found.")
                    confirm = input(f"Delete {current['student_id']} ({current['name']}) permanently? [y/N]: ").strip().lower()
                    if confirm not in {"y", "yes"}:
                        print("Deletion cancelled.")
                        continue
                    deleted = self.delete_student(student_id)
                    print(f"Deleted student: {deleted['student_id']} | {deleted['name']}")
                except ValueError as exc:
                    print(exc)
            elif choice == 7:
                from src.face_recognition import main as recognition_main
                raise SystemExit(recognition_main())


def main():
    manager = StudentManager()
    try:
        return manager.menu()
    except KeyboardInterrupt:
        print("\nStudent management closed by user.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
