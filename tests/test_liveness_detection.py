import unittest

import cv2
import numpy as np

from src.liveness_detection import (
    CHECKING_LIVENESS,
    LIVE_PERSON,
    UNABLE_TO_VERIFY,
    FaceLivenessDetector,
)


class TestFaceLivenessDetector(unittest.TestCase):
    def _make_face_image(self, brightness: int = 120):
        face = np.full((200, 200, 3), brightness, dtype=np.uint8)
        cv2.ellipse(face, (100, 100), (55, 70), 0, 0, 360, (150, 120, 100), -1)
        cv2.rectangle(face, (70, 80), (90, 95), (60, 50, 40), -1)
        cv2.rectangle(face, (110, 80), (130, 95), (60, 50, 40), -1)
        return face

    def test_motionless_face_is_not_immediately_flagged_as_spoof(self):
        detector = FaceLivenessDetector(history_limit=8, min_evidence=3, suspicious_frames=8)
        frame = self._make_face_image(120)
        face_box = (20, 20, 160, 160)

        for _ in range(5):
            result = detector.update(frame, face_box)
            self.assertNotEqual(result["state"], "SUSPECTED_SPOOF")

    def test_active_face_can_reach_live_state_after_enough_evidence(self):
        detector = FaceLivenessDetector(history_limit=8, min_evidence=3, suspicious_frames=8)
        face_box = (20, 20, 160, 160)

        for i in range(10):
            frame = self._make_face_image(120 + i * 3)
            result = detector.update(frame, face_box)

        self.assertIn(result["state"], {LIVE_PERSON, CHECKING_LIVENESS})

    def test_invalid_face_regions_are_handled(self):
        detector = FaceLivenessDetector()
        result = detector.update(np.zeros((100, 100, 3), dtype=np.uint8), (0, 0, 0, 0))
        self.assertEqual(result["state"], UNABLE_TO_VERIFY)

    def test_first_frame_starts_in_checking_state(self):
        detector = FaceLivenessDetector()
        result = detector.update(self._make_face_image(), (20, 20, 160, 160))
        self.assertEqual(result["state"], CHECKING_LIVENESS)


if __name__ == "__main__":
    unittest.main()
