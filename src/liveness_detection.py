from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

CHECKING_LIVENESS = "CHECKING_LIVENESS"
LIVE_PERSON = "LIVE_PERSON"
SUSPECTED_SPOOF = "SUSPECTED_SPOOF"
UNABLE_TO_VERIFY = "UNABLE_TO_VERIFY"


class FaceLivenessDetector:
    """Heuristic liveness detector designed for a small OpenCV-based project.

    This detector intentionally does not claim perfect anti-spoof protection. It
    accumulates weak signals over a short time window and prefers conservative
    results for motionless or ambiguous faces.
    """

    def __init__(
        self,
        history_limit: int = 8,
        min_evidence: int = 3,
        suspicious_frames: int = 8,
    ) -> None:
        self.history_limit = max(4, history_limit)
        self.min_evidence = max(2, min_evidence)
        self.suspicious_frames = max(4, suspicious_frames)
        self._face_cache: Dict[Tuple[int, int, int, int], Dict[str, object]] = {}

    def reset(self) -> None:
        self._face_cache.clear()

    def _make_key(self, face_box: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        x, y, w, h = face_box
        return (int(x), int(y), int(w), int(h))

    def _normalize_face(self, face: np.ndarray) -> np.ndarray:
        if face is None or face.size == 0:
            raise ValueError("Face region is empty.")

        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY) if len(face.shape) == 3 else face
        gray = cv2.resize(gray, (160, 160), interpolation=cv2.INTER_AREA)
        gray = cv2.equalizeHist(gray)
        return gray.astype(np.float32)

    def _compute_motion_score(self, current: np.ndarray, previous: np.ndarray) -> float:
        if previous is None:
            return 0.0
        diff = cv2.absdiff(current, previous)
        return float(np.mean(diff) / 255.0)

    def _compute_texture_score(self, gray: np.ndarray) -> float:
        laplacian = cv2.Laplacian(gray, cv2.CV_32F)
        return float(np.mean(np.abs(laplacian)))

    def _compute_eye_variation(self, gray: np.ndarray) -> float:
        eye_region = gray[int(gray.shape[0] * 0.35):int(gray.shape[0] * 0.60), :]
        if eye_region.size == 0:
            return 0.0
        return float(np.std(eye_region))

    def update(self, frame: np.ndarray, face_box: Optional[Tuple[int, int, int, int]]) -> Dict[str, object]:
        if frame is None or frame.size == 0:
            return {
                "state": UNABLE_TO_VERIFY,
                "message": "Unable to Verify",
                "score": 0.0,
            }

        if face_box is None:
            return {
                "state": UNABLE_TO_VERIFY,
                "message": "No face available for verification",
                "score": 0.0,
            }

        x, y, w, h = face_box
        if w <= 0 or h <= 0 or x < 0 or y < 0:
            return {
                "state": UNABLE_TO_VERIFY,
                "message": "Unable to verify: invalid face region",
                "score": 0.0,
            }

        face_roi = frame[y : y + h, x : x + w]
        if face_roi.size == 0:
            return {
                "state": UNABLE_TO_VERIFY,
                "message": "Unable to Verify",
                "score": 0.0,
            }

        face_gray = self._normalize_face(face_roi)
        key = self._make_key(face_box)
        state_info = self._face_cache.setdefault(
            key,
            {
                "previous_gray": None,
                "motion_history": [],
                "texture_history": [],
                "static_frames": 0,
                "evidence_count": 0,
            },
        )

        previous_gray = state_info["previous_gray"]
        motion_score = self._compute_motion_score(face_gray, previous_gray)
        texture_score = self._compute_texture_score(face_gray)
        eye_variation = self._compute_eye_variation(face_gray)

        state_info["motion_history"].append(motion_score)
        state_info["texture_history"].append(texture_score)

        if len(state_info["motion_history"]) > self.history_limit:
            state_info["motion_history"] = state_info["motion_history"][-self.history_limit:]
            state_info["texture_history"] = state_info["texture_history"][-self.history_limit:]

        recent_motion = float(np.mean(state_info["motion_history"][-3:])) if state_info["motion_history"] else 0.0
        recent_texture = float(np.mean(state_info["texture_history"][-3:])) if state_info["texture_history"] else 0.0
        recent_eye_variation = eye_variation

        if previous_gray is not None:
            if motion_score > 0.02 or recent_eye_variation > 12.0:
                state_info["evidence_count"] += 1
            else:
                state_info["static_frames"] += 1
        else:
            state_info["static_frames"] = 0

        if state_info["static_frames"] > self.suspicious_frames and state_info["evidence_count"] <= 1:
            # Strongly static faces over several frames can indicate a spoofed image,
            # but only when there is also very little evidence of natural motion.
            state = SUSPECTED_SPOOF
            message = "Possible Spoof Detected"
            score = 0.85
        elif state_info["evidence_count"] >= self.min_evidence or recent_motion > 0.04:
            state = LIVE_PERSON
            message = "Live Person"
            score = min(1.0, 0.4 + (state_info["evidence_count"] / max(1, self.min_evidence)) * 0.6)
        elif recent_motion > 0.01 or recent_eye_variation > 7.0:
            state = CHECKING_LIVENESS
            message = "Checking Liveness..."
            score = 0.5
        elif state_info["evidence_count"] == 0:
            state = CHECKING_LIVENESS
            message = "Please blink naturally or gently turn your head."
            score = 0.2
        else:
            state = UNABLE_TO_VERIFY
            message = "Unable to Verify"
            score = 0.1

        state_info["previous_gray"] = face_gray.copy()

        # Never label a motionless person as a spoof purely because they are still.
        # If the person is naturally still, we keep the system in a conservative state.
        if state == SUSPECTED_SPOOF and recent_motion < 0.01 and recent_texture < 8.0:
            state = CHECKING_LIVENESS
            message = "Checking Liveness..."
            score = 0.35

        return {
            "state": state,
            "message": message,
            "score": round(float(score), 2),
            "motion_score": round(float(recent_motion), 4),
            "texture_score": round(float(recent_texture), 4),
            "eye_variation": round(float(recent_eye_variation), 4),
        }

    def update_face_set(self, frame: np.ndarray, face_boxes: List[Tuple[int, int, int, int]]) -> List[Dict[str, object]]:
        results: List[Dict[str, object]] = []
        if frame is None or frame.size == 0:
            return results

        for face_box in face_boxes:
            results.append({"box": face_box, **self.update(frame, face_box)})
        return results
