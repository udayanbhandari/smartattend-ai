# SmartAttend AI Testing and Evaluation Report

Generated on: 2026-11-06 20:31:15

## Actual observed results

- Dataset metadata file exists: Yes
- Registered students found: 0
- Valid face images found: 0
- Validation duration: 0.000 seconds

Warnings:
- No students are currently registered in the dataset.

## Manual evaluation checklist

1. Register four students with clear frontal views and consistent lighting.
2. Test each registered student individually at normal distance (about 0.5 to 1.0 meter).
3. Test each registered student with different head angles: straight, slight left, slight right, slight up, slight down.
4. Test under normal indoor lighting, dim lighting, and strong side lighting.
5. Test unregistered person detection: place an unknown face in front of the camera.
6. Test multiple people in the same frame to confirm only one or multiple faces are handled correctly.
7. Record false acceptance (unknown person accepted as known) and false rejection (known person rejected).
8. Measure response time by estimating time between frame capture and display update using the webcam loop and frame counter.

## Recommended scoring method

- True positive: registered student correctly recognized
- False positive: unknown person identified as a student
- False negative: registered student not recognized
- Precision = true positive / (true positive + false positive)
- Recall = true positive / (true positive + false negative)
- Average recognition latency = total evaluation time / number of recognized frames

## Improvement recommendations

- Capture more varied training images per student.
- Include different lighting conditions during registration.
- Improve face alignment by centering the face before capture.
- Use a stronger model such as OpenCV FaceRecognizer with better feature sets or an external deep-learning model later.
- Add confidence thresholds and a rejection queue for uncertain matches.
- Add a preview panel to help users align their face before capture.