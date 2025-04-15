import cv2
import insightface
from insightface.app import FaceAnalysis
import numpy as np
import pickle
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt

# Initialize the InsightFace app with desired models
app = FaceAnalysis(allowed_modules=['detection', 'recognition'])
app.prepare(ctx_id=-1)  # Use -1 for CPU, or specify a GPU ID if available

video_path = 'videos/ron.mp4'
# Video source (0 for webcam or provide video file path)face_recg2\face_recog_deepNN\dnn_face_recog\videos\spectrum_cctv.mp4 face_recg2\face_recog_deepNN\dnn_face_recog\videos\received_1179123699699768.mp4
# face_recg2\face_recog_deepNN\dnn_face_recog\videos\face-demographics-walking.mp4 face_recg2\face_recog_deepNN\dnn_face_recog\videos\classroom.mp4
video_capture = cv2.VideoCapture(0) 
rtsp_url="rtsp://admin:Sscl1234@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0"
# video_capture = cv2.VideoCapture(video_path)
# video_capture = cv2.VideoCapture(rtsp_url)

# Load encodings for whitelist and blacklist
with open('White_list_EncodeFile.p', 'rb') as f:
    whitelist_encodings, whitelist_names = pickle.load(f)
with open('Black_list_EncodeFile.p', 'rb') as f:
    blacklist_encodings, blacklist_names = pickle.load(f)

known_encodings = np.array(whitelist_encodings + blacklist_encodings)
known_names = whitelist_names + blacklist_names

# for tracking unknown persons
unknown_encodings = []
unknown_names = []

def calculate_similarity_vectorized(embedding, known_encodings):
    embedding = embedding / np.linalg.norm(embedding)
    known_encodings = known_encodings / np.linalg.norm(known_encodings, axis=1, keepdims=True)
    return 1 - np.dot(known_encodings, embedding)

# Define frame processing interval
frame_skip = 1  # Process every 2nd frame
frame_count = 0
unknown_count=0
# Set up threading for parallel face detection
with ThreadPoolExecutor(max_workers=25) as executor:
    while True:
        
        ret, frame = video_capture.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue  # Skip frames to improve performance

        # Resize frame for faster processing
        # frame_resized = cv2.resize(frame, (640, 480))
        frame_resized=frame
        # Detect faces in the frame
        future = executor.submit(app.get, frame_resized)
        faces = future.result()
        
        for face in faces:
            box = face.bbox.astype(int)
            
            embedding = face.normed_embedding
            print(f"Face detected with probability: {face.det_score:.2f}")
            # Compute similarities and find best match
            if(face.det_score>.6):
                similarities = calculate_similarity_vectorized(embedding, known_encodings)
                min_distance = min(similarities)
                min_index = np.argmin(similarities)
                name = known_names[min_index]
                color_box = (0, 255, 0) if name in whitelist_names else (0, 0, 255)

                if min_distance < 0.7:
                    label = "Whitelist" if name in whitelist_names else "Blacklist"
                else:
                    name, label, color_box = "Unknown", "Unknown", (0, 255, 255)
                    if(len(unknown_encodings)==0):
                        unknown_encodings.append(embedding)
                        id=f'U_{unknown_count}'
                        unknown_names.append(id)
                        unknown_count+=1
                    else:
                        similarities_unknown = calculate_similarity_vectorized(embedding, unknown_encodings)
                        min_distance_u = min(similarities_unknown)
                        min_index_u = np.argmin(similarities_unknown)
                        if min_distance_u < 0.7:
                            name = unknown_names[min_index_u]
                        else:
                            name, label, color_box = "Unknown", "Unknown", (0, 255, 255)
                            unknown_encodings.append(embedding)
                            id=f'U_{unknown_count}'
                            unknown_names.append(id)
                            unknown_count+=1
                            
                
                print(f" unknown length: {len(unknown_encodings)}")
                cv2.rectangle(frame_resized, (box[0], box[1]), (box[2], box[3]), color_box, 2)
                cv2.putText(frame_resized, name, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_box, 2)

        # Display the frame
        cv2.imshow('InsightFace Video Processing', frame_resized)
        # plt.imshow(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))

        # Exit loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the video capture and close windows
video_capture.release()
cv2.destroyAllWindows()
