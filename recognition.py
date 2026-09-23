import cv2
from insightface.app import FaceAnalysis


def main():
    print("Loading AI face model...")

    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"]
    )

    app.prepare(
        ctx_id=0,
        det_size=(640, 640)
    )

    print("AI model loaded successfully.")

    image_path = input("Enter image path: ")

    image = cv2.imread(image_path)

    if image is None:
        print("ERROR: Could not load image.")
        return

    faces = app.get(image)

    print(f"Faces detected: {len(faces)}")

    for index, face in enumerate(faces, start=1):
        print(f"Face {index}")
        print(f"Bounding box: {face.bbox}")
        print(f"Embedding size: {len(face.embedding)}")


if __name__ == "__main__":
    main()