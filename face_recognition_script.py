import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime
import pickle

# Define constants
PATH = "Person_Images"

def load_images(path):
    """Load images from the specified path"""
    images = []
    class_names = []
    my_list = os.listdir(path)
    for cl in my_list:
        if cl.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):  # Check if the file is an image
            cur_img = cv2.imread(f'{path}/{cl}')
            if cur_img is not None:  # Check if the image was loaded successfully
                images.append(cur_img)
                class_names.append(os.path.splitext(cl)[0])
    return images, class_names

def find_encodings(images):
    """Encode all the train images and store them in a variable encoded_face_train"""
    encode_list = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encoded_face = face_recognition.face_encodings(img)[0]
        encode_list.append(encoded_face)
    return encode_list

def mark_attendance(name):
    """Mark attendance in the Attendance.csv file"""
    with open('Attendance.csv', 'r+') as f:
        my_data_list = f.readlines()
        name_list = []
        for line in my_data_list:
            entry = line.split(',')
            name_list.append(entry[0])
        if name not in name_list:
            now = datetime.now()
            time = now.strftime('%I:%M:%S:%p')
            date = now.strftime('%d-%B-%Y')
            f.writelines(f'\n{name}, {time}, {date}')

def main():
    images, class_names = load_images(PATH)
    encoded_face_train = find_encodings(images)

    # Take pictures from webcam
    cap = cv2.VideoCapture(0)
    while True:
        success, img = cap.read()
        img_s = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        img_s = cv2.cvtColor(img_s, cv2.COLOR_BGR2RGB)
        faces_in_frame = face_recognition.face_locations(img_s)
        encoded_faces = face_recognition.face_encodings(img_s, faces_in_frame)
        for encode_face, face_loc in zip(encoded_faces, faces_in_frame):
            matches = face_recognition.compare_faces(encoded_face_train, encode_face)
            face_dist = face_recognition.face_distance(encoded_face_train, encode_face)
            match_index = np.argmin(face_dist)
            print(match_index)
            if matches[match_index]:
                name = class_names[match_index].upper().lower()
                y1, x2, y2, x1 = face_loc
                # Since we scaled down by 4 times
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                mark_attendance(name)
        cv2.imshow('webcam', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()