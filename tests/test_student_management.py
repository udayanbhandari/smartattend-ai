import json
import tempfile
import unittest
from pathlib import Path

from src.student_management import StudentManager


class TestStudentManager(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.dataset_dir = self.root / "dataset"
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.faces_dir = self.root / "registered_faces"
        self.faces_dir.mkdir(parents=True, exist_ok=True)
        self.metadata = self.dataset_dir / "students.json"
        self.legacy = self.dataset_dir / "student_registry.json"

        self.metadata.write_text(json.dumps({"students": []}), encoding='utf-8')
        self.legacy.write_text(json.dumps({"students": []}), encoding='utf-8')
        self.manager = StudentManager(self.metadata, self.legacy, self.faces_dir)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_list_and_search_students(self):
        student_one = {"student_id": "student_001", "name": "Alice", "images": ["a.jpg"], "registered_at": "now"}
        student_two = {"student_id": "student_002", "name": "Bob", "images": ["b.jpg"], "registered_at": "now"}
        self._write_students([student_one, student_two])

        students = self.manager.list_students()
        self.assertEqual(len(students), 2)
        self.assertEqual(self.manager.search_students("bob")[0]["student_id"], "student_002")

    def test_update_student_renames_folder_and_metadata(self):
        student = {"student_id": "student_001", "name": "Alice", "images": ["a.jpg"], "registered_at": "now"}
        self._write_students([student])
        folder = self.faces_dir / "student_001"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "face_001.jpg").write_text("x", encoding='utf-8')

        updated = self.manager.update_student("student_001", new_name="Alice Updated", new_student_id="student_099")
        self.assertEqual(updated["student_id"], "student_099")
        self.assertEqual(updated["name"], "Alice Updated")
        self.assertTrue((self.faces_dir / "student_099").exists())

    def test_delete_student_removes_record_and_folder(self):
        student = {"student_id": "student_001", "name": "Alice", "images": ["a.jpg"], "registered_at": "now"}
        self._write_students([student])
        folder = self.faces_dir / "student_001"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "face_001.jpg").write_text("x", encoding='utf-8')

        deleted = self.manager.delete_student("student_001")
        self.assertEqual(deleted["student_id"], "student_001")
        self.assertEqual(self.manager.list_students(), [])
        self.assertFalse(folder.exists())

    def _write_students(self, students):
        self.metadata.write_text(json.dumps({"students": students}, indent=2), encoding='utf-8')
        self.legacy.write_text(json.dumps({"students": students}, indent=2), encoding='utf-8')


if __name__ == "__main__":
    unittest.main()
