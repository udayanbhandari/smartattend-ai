# SmartAttend-AI Phase 9: Model Evaluation and Performance Optimization

## Scope
This report evaluates the current local OpenCV pipeline using the registered dataset and the existing liveness unit tests. It does not claim full real-world spoof resistance or webcam accuracy beyond the measured batch checks performed in this environment.

## Dataset Summary
- Registered students: 4
- Total image files in the current dataset: 40
- Test images used for evaluation: 8

## Recognition Benchmark (Measured)
- Correct predictions: 8
- Incorrect predictions: 0
- Unknown or undetected cases: 0
- Recognition accuracy: 100.0%
- Training time: 3951.3 ms
- Average inference time per face: 6.28 ms
- Estimated FPS: 159.36 FPS

## Liveness Unit Tests (Measured)
- Tests run: 4
- Failures: 0
- Errors: 0
- Success: True

## Resource Snapshot (Measured)
- Process CPU time: 2.5781 s
- Wall-clock time: 0.4653 s
- Peak Python memory: 0.63 MB
- Frames processed in the benchmark: 8

## Actual Observations
- The current project has a small, local registered dataset and is evaluated with a hold-out validation split where possible.
- The recognition benchmark is measured from the real dataset files, not from synthetic assumptions.
- The liveness tests validate logic transitions and conservative handling, but they do not prove real-world spoof resistance against printed photos, phone screens, or replayed videos.
- No live webcam accuracy percentage is claimed because no real-world webcam benchmark was run in this environment.

## Bottlenecks and Optimization Notes
- The most expensive part of the pipeline is the repeated face detection and LBPH prediction on each frame.
- A webcam pipeline should avoid running expensive operations on every frame when there is no detected face.
- The implementation already keeps detection and recognition separate, which is a good design choice for the current project.
- No model replacement or large dependency change was made because the project explicitly requires preserving the current Haar Cascade + LBPH design.

## Recommendations
- Capture more training images per student and keep lighting, distance, and angle consistent.
- Use a larger and more diverse evaluation set if a classroom demonstration is planned.
- Add a dedicated evaluation script for repeated on-device measurements when hardware is available.
- For stronger anti-spoof performance, consider a dedicated pretrained anti-spoof model in a future phase, but only after validation and with explicit dependency management.

## Limitations
- This is not a calibrated anti-spoof benchmark.
- Real-world webcam variability, lighting, and camera quality can change the measured performance significantly.
- The current evaluation is intentionally conservative and data-driven.