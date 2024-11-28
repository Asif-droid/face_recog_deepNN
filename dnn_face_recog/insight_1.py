import cv2
import insightface
from insightface.app import FaceAnalysis
import matplotlib.pyplot as plt
import pickle
from scipy.spatial.distance import cosine
import numpy as np

# Initialize the InsightFace app with desired models
app = FaceAnalysis(allowed_modules=['detection', 'recognition'])
# app.prepare(ctx_id=-1)  # Use -1 for CPU, or specify a GPU ID if available
app.prepare(ctx_id=0)
# model = insightface.app.FaceAnalysis()videos\asif_spoofed.mp4
# model.prepare(ctx_id=-1) face_recg2\face_recog_deepNN\dnn_face_recog\videos\harry_potter_premier.mp4
# face_recg2\face_recog_deepNN\dnn_face_recog\videos\received_1179123699699768.mp4 face_recg2\face_recog_deepNN\dnn_face_recog\videos\face-demographics-walking.mp4
video_path = 'videos/asif_spoofed.mp4'  # Replace with your video file pathface_recg2\face_recog_deepNN\dnn_face_recog\videos\classroom.mp4
# video_capture = cv2.VideoCapture(0)
rtsp_url="rtsp://admin:Secl1234@192.168.1.108:554/cam/realmonitor?channel=1&subtype=1"
video_capture = cv2.VideoCapture(video_path)
# video_capture = cv2.VideoCapture(rtsp_url)

output_video_path = 'videos/out1.mp4'

whitelist_encodings = []
whitelist_names = []
blacklist_encodings = []
blacklist_names = []

attendance = []

with open('White_list_EncodeFile.p', 'rb') as f:
    whitelist_encodings, whitelist_names = pickle.load(f)

with open('Black_list_EncodeFile.p', 'rb') as f:
    blacklist_encodings, blacklist_names = pickle.load(f)


known_encodings = whitelist_encodings + blacklist_encodings
known_names = whitelist_names + blacklist_names
# print(whitelist_names)
# print(blacklist_names)



def calculate_similarity_vectorized(embedding, known_encodings):
    # Normalize the embedding and known encodings for cosine similarity
    embedding = embedding / np.linalg.norm(embedding)
    known_encodings = known_encodings / np.linalg.norm(known_encodings, axis=1, keepdims=True)

    # Compute cosine similarities (1 - cosine distance)
    distance = 1 - np.dot(known_encodings, embedding)
    return distance


frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video_capture.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Process each frame in the video
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    if not ret:
        break  # End of video

    # Detect faces in the frame
    faces = app.get(frame)

    # Draw bounding boxes and labels for each detected face
    for face in faces:
        box = face.bbox.astype(int)  # Bounding box
        

        # Get recognition info if needed (e.g., embedding)
        embedding = face.normed_embedding
        # similarities = [cosine(embedding, known_enc) for known_enc in known_encodings]
        similarities = calculate_similarity_vectorized(embedding, np.array(known_encodings))
        min_distance = min(similarities)
        label = ""
        name = ""
        color_box=(0, 255, 0)
        # min_index = similarities.index(min_distance)
        min_index = np.argmin(similarities)
        name = known_names[min_index]
        # print(f"name:{name} and dis:{min_distance}")
        print(attendance)
        if min_distance < 0.75:
            # min_index = similarities.index(min_distance)
            min_index = np.argmin(similarities)
            name = known_names[min_index]
            if name not in attendance or len(attendance)==0:
                attendance.append(name)
            if name in whitelist_names:
                label = "Whitelist"  
                color_box=(0, 255, 0)
            else: 
                label = "Blacklist"  
                color_box=(0, 0, 255)
        else:
            name = "Unknown" 
            color_box=(0, 255, 255)
        
        # Add additional information on the frame (e.g., gender, age)
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color_box, 2)
        cv2.putText(frame, name, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_box, 2)
    for i in range(0, len(attendance)):
        cv2.putText(frame, f"{i+1}. {attendance[i]}", (0, (i*30)+100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)   
    # Display the frame with annotations
    cv2.imshow('InsightFace Video Processing', frame)
    video_writer.write(frame)
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    
    # Press 'q' to quit the video display
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close display windows
video_capture.release()
video_writer.release()
cv2.destroyAllWindows()
print("Video processing completed!")
