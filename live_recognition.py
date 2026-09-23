import cv2
import pickle
import numpy as np
import subprocess
import time
import threading

from insightface.app import FaceAnalysis

DATABASE_FILE = "data/face_database.pkl"

# Recognition threshold
SIMILARITY_THRESHOLD = 0.45

# Seconds before a person can be greeted again
PERSON_ABSENT_SECONDS = 3

# AI recognition interval
AI_INTERVAL = 0.25


# Shared data between camera and AI
latest_frame = None
latest_result = []

frame_lock = threading.Lock()
result_lock = threading.Lock()

running = True


def load_database():

    print("Loading face database...")

    with open(DATABASE_FILE, "rb") as file:
        database = pickle.load(file)

    print(
        f"Registered people: {list(database.keys())}"
    )

    return database


def load_face_model():

    print("Loading AI face model...")

    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"]
    )

    app.prepare(
        ctx_id=0,
        det_size=(320, 320)
    )

    print("AI model loaded.")

    return app


def recognize_face(embedding, database):

    embedding = embedding / np.linalg.norm(embedding)

    best_name = "Unknown"
    best_similarity = -1

    for name, stored_embedding in database.items():

        similarity = float(
            np.dot(embedding, stored_embedding)
        )

        if similarity > best_similarity:

            best_similarity = similarity
            best_name = name

    if best_similarity >= SIMILARITY_THRESHOLD:

        return best_name, best_similarity

    return "Unknown", best_similarity


def speak_greeting(name):

    message = f"Hi {name}!"

    print(f"🔊 {message}")

    subprocess.Popen(
        ["say", message]
    )


def ai_worker(database, app):

    global latest_frame
    global latest_result
    global running

    last_seen = {}
    greeted_people = set()

    while running:

        # Get latest camera frame
        with frame_lock:

            if latest_frame is None:
                frame = None

            else:
                frame = latest_frame.copy()

        if frame is None:

            time.sleep(0.05)
            continue

        # Resize image before AI processing
        small_frame = cv2.resize(
            frame,
            (480, 360)
        )

        # Run InsightFace
        faces = app.get(small_frame)

        results = []

        current_time = time.time()

        currently_seen = set()

        # Scale coordinates back to 640x480
        scale_x = 640 / 480
        scale_y = 480 / 360

        for face in faces:

            x1, y1, x2, y2 = face.bbox.astype(int)

            # Convert coordinates back to camera resolution
            x1 = int(x1 * scale_x)
            x2 = int(x2 * scale_x)

            y1 = int(y1 * scale_y)
            y2 = int(y2 * scale_y)

            name, similarity = recognize_face(
                face.embedding,
                database
            )

            if name != "Unknown":

                currently_seen.add(name)

                last_seen[name] = current_time

                if name not in greeted_people:

                    speak_greeting(name)

                    greeted_people.add(name)

                label = f"{name} ({similarity:.2f})"

            else:

                label = f"Unknown ({similarity:.2f})"

            results.append(
                (
                    x1,
                    y1,
                    x2,
                    y2,
                    label
                )
            )

        # Remove people who have left
        for name in list(greeted_people):

            if name not in currently_seen:

                if (
                    name in last_seen
                    and
                    current_time - last_seen[name]
                    > PERSON_ABSENT_SECONDS
                ):

                    greeted_people.remove(name)

                    print(
                        f"{name} left the camera view."
                    )

        # Update recognition result
        with result_lock:

            latest_result = results

        # Don't run AI continuously
        time.sleep(AI_INTERVAL)


def main():

    global latest_frame
    global running

    # Load database
    database = load_database()

    # Load AI model
    app = load_face_model()

    # Open camera
    camera = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

    if not camera.isOpened():

        print("ERROR: Could not open camera.")

        return

    # Camera resolution
    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        640
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        480
    )

    # Reduce camera buffering
    camera.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1
    )

    # Create fullscreen window
    window_name = "AI Face Recognition"

    # Create camera window
    window_name = "AI Face Recognition"

    cv2.namedWindow(
    window_name,
    cv2.WINDOW_NORMAL
    )

    cv2.resizeWindow(
     window_name,
     960,
     720
    )

    print()
    print("================================")
    print("AI FACE RECOGNITION SYSTEM")
    print("================================")
    print("FULL SCREEN MODE")
    print("Press Q to quit.")

    # Start AI thread
    ai_thread = threading.Thread(
        target=ai_worker,
        args=(database, app),
        daemon=True
    )

    ai_thread.start()

    while True:

        # Read camera
        success, frame = camera.read()

        if not success:

            print(
                "ERROR: Could not read camera."
            )

            break

        # Send latest frame to AI
        with frame_lock:

            latest_frame = frame.copy()

        # Get latest AI result
        with result_lock:

            results = latest_result.copy()

        # Draw recognition results
        for (
            x1,
            y1,
            x2,
            y2,
            label
        ) in results:

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3
            )

            cv2.putText(
                frame,
                label,
                (x1, y1 - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                3
            )

        # Show camera
        cv2.imshow(
            window_name,
            frame
        )

        # Quit
        if cv2.waitKey(1) & 0xFF == ord("q"):

            break

    # Stop everything
    running = False

    camera.release()

    cv2.destroyAllWindows()

    print("\nSystem stopped.")


if __name__ == "__main__":

    main()
