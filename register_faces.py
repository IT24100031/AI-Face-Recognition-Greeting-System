import os
import pickle

import cv2
import numpy as np
from insightface.app import FaceAnalysis


REGISTERED_DIR = "registered"
OUTPUT_FILE = "data/face_database.pkl"


def load_face_model():
    print("Loading AI face model...")

    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"]
    )

    app.prepare(
        ctx_id=0,
        det_size=(640, 640)
    )

    print("AI model loaded.")
    return app


def create_database(app):
    database = {}

    if not os.path.exists(REGISTERED_DIR):
        print("ERROR: registered folder does not exist.")
        return database

    for person_name in sorted(os.listdir(REGISTERED_DIR)):

        person_folder = os.path.join(
            REGISTERED_DIR,
            person_name
        )

        if not os.path.isdir(person_folder):
            continue

        print(f"\nProcessing: {person_name}")

        embeddings = []

        for filename in sorted(os.listdir(person_folder)):

            image_path = os.path.join(
                person_folder,
                filename
            )

            image = cv2.imread(image_path)

            if image is None:
                print(f"  Could not read: {filename}")
                continue

            faces = app.get(image)

            if len(faces) == 0:
                print(f"  No face found: {filename}")
                continue

            if len(faces) > 1:
                print(
                    f"  WARNING: Multiple faces found in {filename}"
                )
                continue

            embedding = faces[0].embedding

            # Normalize the face embedding
            embedding = embedding / np.linalg.norm(embedding)

            embeddings.append(embedding)

            print(f"  Added: {filename}")

        if embeddings:

            # Average multiple photos
            mean_embedding = np.mean(
                embeddings,
                axis=0
            )

            # Normalize the averaged embedding
            mean_embedding = (
                mean_embedding /
                np.linalg.norm(mean_embedding)
            )

            database[person_name] = mean_embedding

            print(
                f"  ✓ Registered {person_name} "
                f"using {len(embeddings)} photo(s)"
            )

        else:
            print(
                f"  ✗ No usable photos for {person_name}"
            )

    return database


def save_database(database):
    os.makedirs("data", exist_ok=True)

    with open(OUTPUT_FILE, "wb") as file:
        pickle.dump(database, file)

    print(f"\nDatabase saved to: {OUTPUT_FILE}")


def main():

    app = load_face_model()

    database = create_database(app)

    if not database:
        print("\nNo people were registered.")
        return

    save_database(database)

    print("\n================================")
    print("FACE DATABASE CREATED")
    print("================================")

    for name in database:
        print(f"✓ {name}")


if __name__ == "__main__":
    main()