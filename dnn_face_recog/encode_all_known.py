import insightface
import numpy as np
import pickle
import os
import cv2

# Initialize the InsightFace model


   

def create_encoding(image_path, name,model):
    # Load the image
    image = cv2.imread(image_path)
    
    # Detect faces and get encodings
    faces = model.get(image)
    if len(faces) == 0:
        print(f"No face detected in {image_path}")
        return None  # No face detected

    # Use the first detected face (assuming one face per image for whitelist/blacklist)
    encoding = faces[0].normed_embedding
    return encoding, name

# face_recg2\face_recog_deepNN\dnn_face_recog\images\white_listed\8.jpg
# result = create_encoding("images/white_listed/8.jpg", "Asif")


def save_encodings():
    
    model = insightface.app.FaceAnalysis()
    model.prepare(ctx_id=-1)  # Set to -1 for CPU

    known_images = "images/known"


    known_encodings = []
    known_names = []
    # Encode whitelist faces
    for filename in os.listdir(known_images):
        if filename.endswith((".jpg", ".png")):
            path = os.path.join(known_images, filename)
            name = os.path.splitext(filename)[0]  # Assuming filename as the person's name
            name = name.split('.')[0]
            result = create_encoding(path, name,model)
            if result:
                encoding, name = result
                known_encodings.append(encoding)
                known_names.append(name)

    # # Encode blacklist faces
    # for filename in os.listdir(blacklist_images):
    #     if filename.endswith((".jpg", ".png")):
    #         path = os.path.join(blacklist_images, filename)
    #         name = os.path.splitext(filename)[0]  # Assuming filename as the person's name
    #         result = create_encoding(path, name)
    #         if result:
    #             encoding, name = result
    #             blacklist_encodings.append(encoding)
    #             blacklist_names.append(name)


    with open('known_list_EncodeFile.p', 'wb') as f:
        pickle.dump((known_encodings, known_names), f)


    # with open('White_list_EncodeFile.p', 'wb') as f:
    #     pickle.dump((whitelist_encodings, whitelist_names), f)

    # with open('Black_list_EncodeFile.p', 'wb') as f:
    #     pickle.dump((blacklist_encodings, blacklist_names), f)

    # print(result)
    print("Encoding done")
    
    return True


# save_encodings()