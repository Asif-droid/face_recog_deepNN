import cv2
from deepface import DeepFace

# face_recg2\face_recog_deepNN\dnn_face_recog\videos\received_1179123699699768.mp4 
video_path = 'videos/classroom.mp4'  # Replace with your video file pathface_recg2\face_recog_deepNN\dnn_face_recog\videos\classroom.mp4
# video_capture = cv2.VideoCapture(video_path)
video_capture = cv2.VideoCapture(video_path)

# Loop through each frame in the video
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    if not ret:
        break  # Break the loop if there are no frames left to read
    
    frame = cv2.resize(frame, None, fx=1.5, fy=1.5)
    
    try:
        # Analyze the frame for faces
        results = DeepFace.analyze(frame, actions=['gender'], enforce_detection=False)
        
        # Draw bounding boxes and labels on the frame
        for face_data in results:
            # Get the bounding box coordinates
            x, y, w, h = face_data['region']['x'], face_data['region']['y'], face_data['region']['w'], face_data['region']['h']
            
            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display gender, age, and emotion labels
            label = f"{face_data['gender']}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    except Exception as e:
        print("Error processing frame:", e)

    # Display the frame with the bounding boxes and labels
    cv2.imshow('Video with DeepFace', frame)

    # Press 'q' to exit the video display window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close display window
video_capture.release()
cv2.destroyAllWindows()
