from flask import Flask, request, jsonify, render_template, url_for, send_from_directory
from shutil import copyfile
import os
import encode_all_known as encoder
import time
app = Flask(__name__)

# def run_video_processing():
#     face_app.precess_video()


# video_thread = threading.Thread(target=run_video_processing)
# video_thread.daemon = True  # Daemonize the thread so it closes when the main program exits
# video_thread.start()


# @app.route('/process_frame', methods=['POST'])
# def process_frame():
#     # Decode the received image
    

#     attendance = face_app.attendance_list
    

#     return jsonify(attendance)

def get_attendance():
    # Read the attendance file  face_recg2\face_recog_deepNN\dnn_face_recog\attendance\attendance_list.txt
    people = []
    with open('./attendance/attendance_list.txt', 'r') as file:
        for line in file:
        # Split the line into name and status
            name, status = line.strip().split(': ')
            # Create a dictionary for the person
            person = {'name': name, 'status': status}
            # Append the dictionary to the people array
            people.append(person)
    
    return people
    # print(people)

IMAGE_DIR_U = './images/unknown'
IMAGE_DIR_K = './images/known'
IMAGE_DIR_D = '/images/default'



@app.route('/')
def index():
    return render_template('index4.html')

@app.route('/get_status')
def get_status():
    # Simulate dynamic updates
    # for person in people:
    #     person['status'] = random.choice(['present', 'absent'])
    # return jsonify(people)
    attendance = get_attendance()
    
    # print(attendance)
    return jsonify(attendance)


@app.route('/save_image', methods=['POST'])
def save_image():
    data = request.get_json()
    image_url = data.get('imageUrl')
    box_name = data.get('boxName')

    # Extract the image filename from the URL
    filename = os.path.basename(image_url)

    save_directory = './images/known/'
    
    
    # Define the save path (ensure the directory exists)
    save_path = os.path.join(save_directory, f'{box_name}.jpg')
    os.chmod(save_directory, 0o777)
    

    # Copy the image to the new folder
    src_path = os.path.join(IMAGE_DIR_U, filename)
    # print("this is srcpath")
    # print(src_path)
    # dst_path = os.path.join(save_path, filename)
    if os.path.exists(src_path):
        
        copyfile(src_path, save_path)
        os.remove(src_path)
        return jsonify({'message': 'Image saved successfully!'}), 200
    else:
        return jsonify({'message': 'Image not found'}), 404


@app.route('/delete_image', methods=['POST'])
def delete_image():
    data = request.get_json()
    image_url = data.get('imageUrl')
    

    # Extract the image filename from the URL
    filename = os.path.basename(image_url)

    
    
    
    # Define the save path (ensure the directory exists)
    

    # Copy the image to the new folder
    src_path = os.path.join(IMAGE_DIR_U, filename)
    # print("this is srcpath")
    # print(src_path)
    # dst_path = os.path.join(save_path, filename)
    if os.path.exists(src_path):
        os.remove(src_path)
        return jsonify({'message': 'Image saved successfully!'}), 200
    else:
        return jsonify({'message': 'Image not found'}), 404



@app.route('/get_images', methods=['GET'])
def get_images():
    images = []
    # print("Getting images called")
    
    # Check if the directory exists
    if os.path.exists(IMAGE_DIR_U):
        # Iterate over files in the directory
        for filename in os.listdir(IMAGE_DIR_U):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                image_url = url_for('serve_image', filename=filename)
                images.append({
                    'url': image_url,
                    'name': filename
                })

    return jsonify(images)

@app.route('/images/<filename>')
def serve_image(filename):
    """Serve images from the IMAGE_DIR."""
    return send_from_directory(IMAGE_DIR_U, filename)  

@app.route('/images/known/<filename>')
def serve_image_known(filename):
    """Serve images from the IMAGE_DIR."""
    return send_from_directory(IMAGE_DIR_K, filename) 

@app.route('/images/default/<filename>')
def serve_image_default(filename):
    """Serve images from the IMAGE_DIR."""
    return send_from_directory(IMAGE_DIR_D, filename)  


@app.route('/start_encoding', methods=['POST'])
def start_process():
    # Your backend processing logic here
    print("Encoding started started...")
    encoder.save_encodings()
    # time.sleep(5)  # Simulate a long-running process
    return jsonify({'message': 'Process completed successfully!'}), 200

 
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)