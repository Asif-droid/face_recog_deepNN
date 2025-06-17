import cv2
import insightface
from insightface.app import FaceAnalysis
import matplotlib.pyplot as plt
import pickle
from scipy.spatial.distance import cosine
import numpy as np
import time
import threading
import os
from queue import Queue

# Initialize the InsightFace app with desired models
model_path="models/model"
app = FaceAnalysis(allowed_modules=['detection', 'recognition'])
app.prepare(ctx_id=-1)  # Use -1 for CPU, or specify a GPU ID if available

# model = insightface.app.FaceAnalysis()
# model.prepare(ctx_id=-1) face_recg2\face_recog_deepNN\dnn_face_recog\videos\harry_potter_premier.mp4
# face_recg2\face_recog_deepNN\dnn_face_recog\videos\received_1179123699699768.mp4 face_recg2\face_recog_deepNN\dnn_face_recog\videos\face-demographics-walking.mp4
video_path = 'videos/all_video4.mp4'  # Replace with your video file pathface_recg2\face_recog_deepNN\dnn_face_recog\videos\classroom.mp4
# video_capture = cv2.VideoCapture(0)
rtsp_url="rtsp://admin:Sscl1234@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0"
# video_capture = cv2.VideoCapture(video_path)
video_capture = cv2.VideoCapture(rtsp_url)

output_video_path = 'videos/all_4_faces.mp4'

whitelist_encodings = []
whitelist_names = []
blacklist_encodings = []
blacklist_names = []
known_encodings=[]
known_names=[]

unknown_encodings = []
unknown_names = []

attendance_list={}

face_buffer = []

width = 1920
height = 1080

# with open('White_list_EncodeFile.p', 'rb') as f:
#     whitelist_encodings, whitelist_names = pickle.load(f)

# with open('Black_list_EncodeFile.p', 'rb') as f:
#     blacklist_encodings, blacklist_names = pickle.load(f)


def load_known_encodings(filename):
    """Load the whitelist encodings from the file."""
    with open(filename, 'rb') as f:
        return pickle.load(f)

# with open('known_list_EncodeFile.p', 'rb') as f:
#     known_encodings, known_names = pickle.load(f)



# known_encodings = whitelist_encodings + blacklist_encodings
# known_names = whitelist_names + blacklist_names
# print(whitelist_names)
# print(blacklist_names)

# for name in whitelist_names:
#     attendance_list[name] = 'Absent'

# print(attendance_list)

def calculate_similarity_vectorized(embedding, known_encodings):
    # Normalize the embedding and known encodings for cosine similarity
    embedding = embedding / np.linalg.norm(embedding)
    known_encodings = known_encodings / np.linalg.norm(known_encodings, axis=1, keepdims=True)

    # Compute cosine similarities (1 - cosine distance)
    distance = 1 - np.dot(known_encodings, embedding)
    return distance


# frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = int(video_capture.get(cv2.CAP_PROP_FPS))
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

def is_blurry(frame, threshold=100.0,resize_factor=0.5):
    # Resize the frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=resize_factor, fy=resize_factor)
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    return variance < threshold, variance

def add_face_to_buffer(frame,box):
    x1, y1, x2, y2 = box
    x1 = max(0, x1 - 80)
    y1 = max(0, y1 - 80)
    x2 = min(width, x2 + 80)
    y2 = min(height, y2 + 80)
    
    face_image = frame[y1:y2, x1:x2]
    face_buffer.append(face_image)

def load_profile(known_encodings, known_names):
    encoding_dict = load_known_encodings('profile_encodings.pkl')
    for name, encodings in encoding_dict.items():
        known_encodings.extend(encodings)
        known_names.extend([name] * len(encodings))

def get_attendance():
    return attendance_list


    

def process_video():
    # Process each frame in the video
    frame_skip =  18 # Process every 2nd frame works best with 18
    frame_count = 0
    unknown_count=0
    
    encode_filename = 'known_list_EncodeFile.p'

    known_encodings, known_names = load_known_encodings(encode_filename)
    # known_encodings=[]
    # known_names=[]
    load_profile(known_encodings, known_names)
    last_modified_time = os.path.getmtime(encode_filename)
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        if not ret:
            break  # End of video
        
        
        frame_count += 1
        if frame_count % frame_skip != 0:
            continue 
        
        blurry, variance = is_blurry(frame, threshold=100.0)
        if blurry:
            print(f"Frame discarded due to blur (variance: {variance})")
            continue
        
        frame = cv2.resize(frame, (width, height))
        # for demo video 640,480
        
        # frame = cv2.resize(frame, (640,480))
        
        
        # Detect faces in the frame
        faces = app.get(frame)
        
        current_modified_time = os.path.getmtime(encode_filename)
        if (current_modified_time != last_modified_time):
            known_encodings, known_names = load_known_encodings(encode_filename)
            last_modified_time = current_modified_time
            print("Known encodings updated.")

        # Draw bounding boxes and labels for each detected face
        for face in faces:
            box = face.bbox.astype(int)  # Bounding box
            
            # print("new face detected")
            # Get recognition info if needed (e.g., embedding)
            embedding = face.normed_embedding
            # print(len(embedding))
            # print(f"Face detected with probability: {face.det_score:.2f}")
            if(face.det_score>.5):
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
                if min_distance < 0.68:
                    # min_index = similarities.index(min_distance)
                    min_index = np.argmin(similarities)
                    name = known_names[min_index]
                    if name in whitelist_names:
                        label = "Whitelist"  
                        color_box=(0, 255, 0)
                    else: 
                        label = "Blacklist"  
                        color_box=(0, 255, 0)
                    attendance_list[name] = 'Present'
                    # print(f"Recognized: {name}")
                else:
                    name = "Unknown" 
                    color_box=(0, 255, 255)
                    if(len(unknown_encodings)==0):
                        if (face.det_score >.5):
                            unknown_encodings.append(embedding)
                            id=f'U_{unknown_count}'
                            unknown_names.append(id)
                            add_face_to_buffer(frame,box)
                            print(f"image added to buffer{id}")
                            unknown_count+=1
                    else:
                        similarities_unknown = calculate_similarity_vectorized(embedding, unknown_encodings)
                        min_distance_u = min(similarities_unknown)
                        min_index_u = np.argmin(similarities_unknown)
                        name = unknown_names[min_index_u]
                        
                        if min_distance_u < 0.7:
                            name = unknown_names[min_index_u]
                        else:
                            if (face.det_score >.845):  #0.84561
                                print(f"name:{name} and dis:{min_distance}")
                                name, label, color_box = "Unknown", "Unknown", (0, 255, 255)
                                unknown_encodings.append(embedding)
                                id=f'U_{unknown_count}'
                                unknown_names.append(id)
                                add_face_to_buffer(frame,box)
                                print(f"image added to buffer{id}")
                                unknown_count+=1
                
                # # # Add additional information on the frame (e.g., gender, age)
                # cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color_box, 2)
                # cv2.putText(frame, name, (box[0], box[1]-3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_box, 2)

        # Display the frame with annotations
        # cv2.imshow('InsightFace Video Processing', frame)
        # video_writer.write(frame)
        # plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        
        # Press 'q' to quit the video display
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    # close()
    
    
# def close():
#     # Release the video capture object and close display windows
#     video_capture.release()
#     video_writer.release()
#     cv2.destroyAllWindows()
#     print("Video processing completed!")

def clean_unknown(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)

def write_attendance_to_file():
    un_im_dir='./images/unknown'
    clean_unknown(un_im_dir)
    unknown_written=0
    attendance_file='./attendance/attendance_list.txt'
    atndnc_file_last_modified_time = None
    
    while True:
        time.sleep(1)  # 
        with open(attendance_file, 'w') as file:
            for name, status in attendance_list.items():
                file.write(f"{name}: {status}\n")
        # atndnc_file_current_modified_time=os.path.getmtime(attendance_file)
        # print(f"last modified time:{atndnc_file_last_modified_time} and current modified time:{atndnc_file_current_modified_time}")
        # if(atndnc_file_current_modified_time!=atndnc_file_last_modified_time):
        #     # print(attendance_list)
        #     print("Attendance list updated.")
        #     new_attendance_list = {}
        #     with open(attendance_file, 'r') as file:
        #         for line in file:
        #             line = line.strip()
        #             if not line:
        #                 continue
        #             name, status = line.split(': ')
        #             new_attendance_list[name.strip()] = status.strip()
                    
        #     attendance_list = new_attendance_list
            
        # else:
            
        #     with open(attendance_file, 'w') as file:
        #         for name, status in attendance_list.items():
        #             file.write(f"{name}: {status}\n")
        
        # atndnc_file_last_modified_time=atndnc_file_current_modified_time
        
        for i, face_image in enumerate(face_buffer):
            if face_image is not None and face_image.size > 0:
                blurry, variance = is_blurry(face_image, threshold=300.0)
                print(variance)
                if blurry:
                    print(f"Frame discarded due to blur (variance: {variance})")
                    continue
                output_path = os.path.join(un_im_dir, f'{unknown_names[unknown_written]}.jpg')
                unknown_written+=1
                cv2.imwrite(output_path, face_image)
            
        # with open('./attendance/unknown_list.txt', 'w') as file:
        #     for name in unknown_names:
        #         file.write(f"{name}\n")
        face_buffer.clear()
        # print("Attendance list updated in file.")
        
        
        

video_thread = threading.Thread(target=process_video)
video_thread.daemon = True
video_thread.start()

file_thread = threading.Thread(target=write_attendance_to_file)
file_thread.daemon = True
file_thread.start()

while True:
    time.sleep(1)