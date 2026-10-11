import os
import pickle
import cv2
import numpy as np
from django.conf import settings
from tensorflow.keras.models import load_model


# Project model paths
BASE_DIR = settings.BASE_DIR

MODEL_PATH = os.path.join(
    BASE_DIR, "trained_model_cnn_sift.h5"
)

CLASS_NAMES_PATH = os.path.join(
    BASE_DIR, "class_names.pkl"
)

CASCADE_PATH = os.path.join(
    BASE_DIR,
    "models",
    "haarcascade_frontalface_default.xml"
)

MAX_KEYPOINTS = 100


def load_class_names():
    """Load the face-shape class names."""
    with open(CLASS_NAMES_PATH, "rb") as file:
        return pickle.load(file)


def detect_face_shape(image_path):
    """
    Predict a face shape from a saved image.

    The image must be processed using the same
    preprocessing method used during model training.
    """

    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Trained CNN model has not been added yet."
        )

    if not os.path.exists(CLASS_NAMES_PATH):
        raise FileNotFoundError(
            "class_names.pkl has not been added yet."
        )

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Unable to read the image.")

    # Match the preprocessing used during training
    height = image.shape[0]
    image = image[:height // 2, :]

    image = cv2.resize(image, (200, 150))

    gray = cv2.cvtColor(
        image, cv2.COLOR_BGR2GRAY
    )

    gray = gray.astype("float32") / 255.0

    # Detect the face
    face_cascade = cv2.CascadeClassifier(
        CASCADE_PATH
    )

    if face_cascade.empty():
        raise FileNotFoundError(
            "Unable to load the Haar Cascade file."
        )

    faces = face_cascade.detectMultiScale(
        (gray * 255).astype("uint8"),
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    if len(faces) == 0:
        raise ValueError(
            "No face detected. Please try another image."
        )

    x, y, w, h = faces[0]

    face = gray[y:y + h, x:x + w]

    # Extract SIFT descriptors
    sift = cv2.SIFT_create()

    keypoints, descriptors = sift.detectAndCompute(
        (face * 255).astype("uint8"), None
    )

    if descriptors is None:
        descriptors = np.empty((0, 128))

    # Pad or truncate to 100 descriptors
    descriptors = descriptors[:MAX_KEYPOINTS]

    if len(descriptors) < MAX_KEYPOINTS:
        padding = np.zeros(
            (MAX_KEYPOINTS - len(descriptors), 128),
            dtype=np.float32
        )

        descriptors = np.vstack(
            (descriptors, padding)
        )

    # Prepare input for the trained CNN
    features = descriptors.flatten()
    features = features.reshape(1, -1, 1)

    model = load_model(MODEL_PATH)

    predictions = model.predict(
        features, verbose=0
    )

    predicted_index = int(
        np.argmax(predictions[0])
    )

    class_names = load_class_names()

    if isinstance(class_names, dict):
        face_shape = class_names[predicted_index]
    else:
        face_shape = class_names[predicted_index]

    confidence = float(
        predictions[0][predicted_index]
    )

    return {
        "face_shape": str(face_shape).lower(),
        "confidence": confidence
    }


def get_hairstyle_recommendations(face_shape):
    """Return hairstyle suggestions for a face shape."""

    recommendations = {
        "heart": [
            "Side-swept bangs",
            "Chin-length bob",
            "Shoulder-length layers"
        ],
        "oblong": [
            "Curtain bangs",
            "Wavy shoulder-length hair",
            "Layered bob"
        ],
        "oval": [
            "Long layers",
            "Bob hairstyle",
            "Pixie cut"
        ],
        "round": [
            "Long layered hairstyle",
            "Side-parted hairstyle",
            "Long bob"
        ],
        "square": [
            "Soft waves",
            "Layered hairstyle",
            "Side-swept bangs"
        ]
    }

    return recommendations.get(
        face_shape.lower(),
        []
  )
