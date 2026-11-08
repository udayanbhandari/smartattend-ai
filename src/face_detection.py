from pathlib import Path

import cv2


class FaceDetector:
    """Detect faces in video frames using an OpenCV Haar Cascade classifier."""

    def __init__(self, model_path=None):
        default_model = Path(__file__).resolve().parents[1] / "models" / "haarcascade_frontalface_default.xml"

        if model_path is None:
            model_path = default_model

        self.model_path = str(model_path)

        # If the project model file is missing, fall back to the packaged OpenCV model.
        if not Path(self.model_path).exists():
            built_in_model = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
            if built_in_model.exists():
                self.model_path = str(built_in_model)

        self.face_cascade = cv2.CascadeClassifier(self.model_path)

        if self.face_cascade.empty():
            raise FileNotFoundError(
                f"Unable to load Haar Cascade model from: {self.model_path}"
            )

    def detect_faces(self, frame):
        """Return a frame with detected faces highlighted and the count displayed."""
        if frame is None:
            raise ValueError("Cannot detect faces: input frame is empty.")

        # Convert to grayscale because Haar cascades are trained on grayscale images.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        # Detect faces.
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        # Draw rectangles around each detected face.
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the number of detected faces.
        cv2.putText(
            frame,
            f"Faces detected: {len(faces)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        return frame
