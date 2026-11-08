# SmartAttend AI Facial Recognition Module

This project is an independent Python-based AI module for the SmartAttend system. It is designed to run separately from the existing PHP, MySQL, and website implementation in the workspace.

## Project purpose

- Build a webcam-based face recognition prototype
- Keep the existing SmartAttend website unchanged
- Register students locally using a laptop webcam
- Detect and recognize faces in real time
- Add lightweight liveness checks to reduce obvious spoof attempts
- Keep the AI subsystem modular and standalone

## Core features

- Webcam access with OpenCV
- Haar Cascade face detection
- Student registration with multiple face images
- Local JSON metadata storage
- LBPH face recognition for registered students
- Unknown-person detection
- Liveness detection state tracking
- Student management menu for list/search/update/delete
- Automated tests for liveness and student management

## Technologies used

- Python 3.10+
- OpenCV (`opencv-contrib-python`)
- NumPy
- Standard library modules such as JSON, pathlib, and shutil

## System requirements

- Windows 10 or 11
- VS Code
- Laptop webcam
- Python installed and available on PATH
- XAMPP is not required for this module

## Project architecture

The project is intentionally separate from the existing PHP website:

- The website remains untouched in the existing htdocs project
- This AI module runs in its own folder: SmartAttend-AI
- Face detection, registration, recognition, liveness, and student management are handled by separate modules in the src folder
- All student metadata is stored in the dataset folder
- Student face images are stored in the registered_faces directory

## Folder structure

```text
SmartAttend-AI/
├── dataset/
│   ├── students.json
│   └── student_registry.json
├── models/
│   └── haarcascade_frontalface_default.xml
├── registered_faces/
│   ├── student_001/
│   ├── student_002/
│   ├── student_003/
│   ├── student_004/
│   └── student_039/
├── src/
│   ├── face_detection.py
│   ├── face_registration.py
│   ├── face_recognition.py
│   ├── liveness_detection.py
│   ├── student_management.py
│   ├── model_evaluation.py
│   └── __init__.py
├── tests/
│   ├── test_liveness_detection.py
│   └── test_student_management.py
├── evaluation_report.md
├── evaluation_console.txt
├── main.py
├── requirements.txt
├── README.md
├── venv/
└── .gitignore (if present)
```

## Installation

Open PowerShell in the project folder and run:

```powershell
Set-Location "D:\VIT\3rd sem\DEAM\DEAM CP\SmartAttend-AI"
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks script execution, run this once before activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## Requirements file

The current project dependency list is:

```text
opencv-contrib-python
numpy
```

## Running the Web Management Dashboard & Live Scanner

To start the modern web interface for student management and live face recognition:

```powershell
Set-Location "d:\VIT\Dev\SmartAttend-AI"
.\venv\Scripts\Activate.ps1
uvicorn app:app --reload --port 8000
```

Open your browser and navigate to:
**`http://localhost:8000`**

From this single web dashboard, you can:
- **Manage Students:** View registered students, preview face photos, search, and delete students.
- **In-Browser Face Enrollment:** Enroll new students directly through the browser webcam (capturing 5-10 face samples) with automatic model retraining.
- **Live Attendance Scanner:** Scan classroom faces in real-time with visual bounding boxes, LBPH distance scores, liveness badges (`LIVE_PERSON`, `CHECKING_LIVENESS`, `SUSPECTED_SPOOF`), and real-time attendance logging with CSV export.

---

## Deploying to Render (Cloud Hosting)

### Why Render instead of Vercel?
- **Vercel** is a serverless platform with an ephemeral filesystem and strict 250 MB size limits. Files stored locally (`dataset/students.json` and `registered_faces/`) disappear whenever a serverless lambda shuts down. Furthermore, loading the LBPH model on every cold start causes severe latency.
- **Render** runs full Docker web containers 24/7 with persistent memory. The model stays trained in RAM for instant inference (~6ms), and Render offers a free tier for web services.

### Steps to Deploy on Render:
1. **Push your code to GitHub:**
   ```powershell
   git init
   git add .
   git commit -m "Add web interface and Render deployment configuration"
   git remote add origin https://github.com/<your-username>/<your-repo-name>.git
   git push -u origin main
   ```
2. **Create Web Service on Render:**
   - Log into [Render.com](https://render.com).
   - Click **New +** -> **Web Service**.
   - Connect your GitHub repository.
   - Choose **Docker** as the Environment.
   - Render will automatically use the included `Dockerfile` and `render.yaml`.
3. **Open your Public App:**
   - Once the build succeeds, Render provides a public URL (e.g. `https://smartattend-ai.onrender.com`).
   - Anyone on PC or mobile can access the student management portal and live face scanner!

---

## Running the Application via Desktop CLI

From the project root:

```powershell
Set-Location "D:\VIT\3rd sem\DEAM\DEAM CP\SmartAttend-AI"
.\venv\Scripts\Activate.ps1
python main.py
```

The application starts with a student management menu:

1. Register new student
2. List all students
3. Search student
4. View student details
5. Update student
6. Delete student
7. Run recognition
0. Exit

To start recognition directly:

```powershell
python .\src\face_recognition.py
```

## Student registration workflow

1. Run the application with `python main.py`
2. Choose the registration option
3. Enter a valid student ID such as `student_001`
4. Enter the student's name
5. Keep only one face visible in the webcam
6. Press `Space` to capture images
7. Capture 8 to 10 clear images for a good demo
8. Press `Q` to cancel at any time
9. The images are saved under `registered_faces/student_id`
10. Metadata is saved in `dataset/students.json`

## Recognition workflow

1. Open the main menu
2. Select the recognition option
3. Press `S` to start the webcam
4. The system detects faces and compares them against the trained dataset
5. The recognized student ID and name are shown when a match is accepted
6. If no reliable match is found, the system shows `Unknown Person`
7. Press `Q` to exit the application

## Liveness workflow

The liveness detector is separate and conservative:

- It collects evidence over several frames
- It avoids labeling a still real person as a spoof immediately
- If evidence is insufficient, it uses `CHECKING_LIVENESS` or `UNABLE_TO_VERIFY`
- It does not claim perfect anti-spoof protection against all printed photos, phone displays, or replayed video

## Testing and validation

Run the test suite:

```powershell
Set-Location "D:\VIT\3rd sem\DEAM\DEAM CP\SmartAttend-AI"
.\venv\Scripts\Activate.ps1
python -m unittest discover -s tests -v
```

Run a syntax check:

```powershell
python -m py_compile main.py src\face_detection.py src\face_registration.py src\face_recognition.py src\liveness_detection.py src\student_management.py src\model_evaluation.py
```

Generate the evaluation report:

```powershell
python .\src\model_evaluation.py
```

This writes the report to:

- [evaluation_report.md](./evaluation_report.md)

## Known limitations

- The system is designed for a local prototype and a small student dataset
- Recognition accuracy depends on image quality, lighting, and camera framing
- Haar Cascade and LBPH are lightweight but not the strongest modern deep-learning pipeline
- Liveness checks are heuristic and conservative, not a complete anti-spoofing guarantee
- Real-world webcam conditions can affect both recognition and detection performance

## Future improvements

- Add a larger validated student dataset
- Improve image capture consistency
- Use stronger recognition methods after validation
- Add a dedicated anti-spoof deep learning model in a future phase
- Add a GUI for student list and management if needed

## Packaging and transfer guidance

This project can be transferred to another Windows computer by copying the project folder and reinstalling dependencies with:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Do not include the virtual environment in a final zip archive unless specifically required.
Keep the original dataset and model files, because they are required for recognition and evaluation.

## Final notes

- The website and PHP project remain untouched.
- This module is separate and self-contained.
- The project is ready for demonstration, testing, and future extension.
