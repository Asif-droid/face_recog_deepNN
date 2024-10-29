import cv2
import insightface
from insightface.app import FaceAnalysis
import matplotlib.pyplot as plt
import pickle
from scipy.spatial.distance import cosine

# Initialize the InsightFace app with desired models
app = FaceAnalysis(allowed_modules=['detection', 'recognition'])
app.prepare(ctx_id=-1)  # Use -1 for CPU, or specify a GPU ID if available

# model = insightface.app.FaceAnalysis()
# model.prepare(ctx_id=-1)
# face_recg2\face_recog_deepNN\dnn_face_recog\videos\received_1179123699699768.mp4 face_recg2\face_recog_deepNN\dnn_face_recog\videos\face-demographics-walking.mp4
video_path = 'videos/face-demographics-walking.mp4'  # Replace with your video file pathface_recg2\face_recog_deepNN\dnn_face_recog\videos\classroom.mp4
video_capture = cv2.VideoCapture(0)
# video_capture = cv2.VideoCapture(video_path)

whitelist_encodings = []
whitelist_names = []
blacklist_encodings = []
blacklist_names = []


with open('White_list_EncodeFile.p', 'rb') as f:
    whitelist_encodings, whitelist_names = pickle.load(f)

with open('Black_list_EncodeFile.p', 'rb') as f:
    blacklist_encodings, blacklist_names = pickle.load(f)


known_encodings = whitelist_encodings + blacklist_encodings
known_names = whitelist_names + blacklist_names
# print(whitelist_names)
# print(blacklist_names)







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
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

        # Get recognition info if needed (e.g., embedding)
        embedding = face.normed_embedding
        similarities = [cosine(embedding, known_enc) for known_enc in known_encodings]
        min_distance = min(similarities)
        label = ""
        name = ""
        color_box=(0, 255, 0)
        min_index = similarities.index(min_distance)
        name = known_names[min_index]
        print(f"name:{name} and dis:{min_distance}")
        if min_distance < 0.76:
            min_index = similarities.index(min_distance)
            name = known_names[min_index]
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
        
        cv2.putText(frame, name, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_box, 2)

    # Display the frame with annotations
    cv2.imshow('InsightFace Video Processing', frame)
    # plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


    # Press 'q' to quit the video display
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close display windows
video_capture.release()
cv2.destroyAllWindows()
