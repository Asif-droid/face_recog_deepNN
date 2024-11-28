import cv2

output_video_path = 'videos/sample7.mp4'


rtsp_url="rtsp://admin:Sscl1234@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0"
# video_capture = cv2.VideoCapture(video_path)
video_capture = cv2.VideoCapture(rtsp_url)

frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video_capture.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))


while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    if not ret:
        break  # End of video

   
    # Draw bounding boxes and labels for each detected face
    
    # Display the frame with annotations
    cv2.imshow('InsightFace Video Processing', frame)
    video_writer.write(frame)
    # plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    
    # Press 'q' to quit the video display
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close display windows
video_capture.release()
video_writer.release()
cv2.destroyAllWindows()
print("Video processing completed!")


