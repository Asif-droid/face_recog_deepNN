# face_recog_deepNN
deep face:
pip install deepface
pip install opencv-python
pip install tf-keras

insightface:
pip install insightface
pip install onnxruntime 
pip uninstall opencv-python-headless
pip install opencv
---can be a prblem with opencv uninstall make sure it uninstalls all open cv module and re install

to run 
```
python .\insight_face.py
```
increase skip frame rate in case the video rendering is slow.
change the cv2.VideoCapture() function to capture video from different src.
``` 
 video_capture = cv2.VideoCapture(video_path)  # recorded video
 video_capture = cv2.VideoCapture(rtsp_url)     # camera feed
 video_capture = cv2.VideoCapture(0)     # web cam

```

## Run web attendance
first activate the python venv in 2 separate terminal
```
faceID\faceID\face_recg2\face_recog_deepNN\dnn_face_recog\Scripts\activate
```
Run insight_face.py in 1 terminal and app.py in another

```
python insight_face.py
python app.py
```
 