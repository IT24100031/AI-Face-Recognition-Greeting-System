# AI Face Recognition Greeting System

An AI-powered real-time face recognition system that identifies registered individuals through a live camera feed and provides a personalized voice greeting.

## Features

- Real-time face recognition
- Face detection and face embedding comparison
- Personalized voice greeting
- Live camera processing
- Multithreaded AI processing
- GUI interface using Tkinter
- macOS camera support
- Registered face database
- InsightFace-based recognition

## Technologies Used

- Python
- OpenCV
- InsightFace
- ONNX Runtime
- NumPy
- Tkinter
- Threading

## How It Works

1. The application loads the registered face database.
2. InsightFace loads the face recognition model.
3. OpenCV accesses the camera.
4. Live video frames are captured continuously.
5. InsightFace detects faces and generates face embeddings.
6. The generated embeddings are compared with registered face embeddings.
7. When a registered person is recognized, the system provides a personalized voice greeting.

## Project Structure

```text
AI-Face-Recognition-Greeting-System/
│
├── app/
│   └── main.py
│
├── camera_test.py
├── face_detection.py
├── live_recognition.py
├── main.py
├── recognition.py
├── register_faces.py
└── README.md
