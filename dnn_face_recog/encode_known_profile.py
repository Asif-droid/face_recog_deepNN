import insightface
import numpy as np
import pickle
import os
import cv2

# Initialize the InsightFace model
PROFILES_DIR = "profiles"  # Directory containing known profiles
OUTPUT_FILE = "profile_encodings.pkl"  # File to save the encodings


def create_img_encoding(image_path, model):
    # Load the image
    image = cv2.imread(image_path)

    # Detect faces and get encodings
    faces = model.get(image)
    if len(faces) == 0:
        print(f"No face detected in {image_path}")
        return None

    # Use the first detected face
    encoding = faces[0].normed_embedding
    return encoding


def save_encodings():
    model = insightface.app.FaceAnalysis()
    model.prepare(ctx_id=-1)  # CPU
    encodings_dict = {}
    profiles_dir=PROFILES_DIR
      # Dictionary to hold name: [encodings]

    for person_name in os.listdir(profiles_dir):
        person_folder = os.path.join(profiles_dir, person_name)
        if not os.path.isdir(person_folder):
            continue

        encodings = []
        for filename in os.listdir(person_folder):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(person_folder, filename)
                encoding = create_img_encoding(image_path, model)
                if encoding is not None:
                    encodings.append(encoding)

        if encodings:
            encodings_dict[person_name] = encodings

    print(encodings_dict)
    # Save the dictionary to a pickle file
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(encodings_dict, f)

    print(f"Encodings saved to '{OUTPUT_FILE}'")
    return True


save_encodings()