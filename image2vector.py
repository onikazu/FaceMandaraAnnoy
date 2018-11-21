import dlib

import face_recognition
import json
import sys
import glob
import pickle

# trimmed_image_path = "./database/img_align_celeb"
trimmed_image_path = "./big_database"
trimmed_images = glob.glob(trimmed_image_path + "/*.jpg")

vector_images = {}
data_num = 0
# 辞書型のデータを作る
for image_file in trimmed_images:
    image = face_recognition.load_image_file(image_file)
    # 顔認識
    detector = dlib.get_frontal_face_detector()
    rects = detector(image, 1)
    print(data_num)

    # 顔認識してい無いとき
    if not rects:
        continue
    face_encoding = face_recognition.face_encodings(image)[0]
    print(len(face_encoding.tolist()))
    vector_images[image_file.split("/")[-1]] = face_encoding.tolist()
    data_num += 1

# vector_images is like {"face0.jpg":[0.11,.....], }
# with open('data.pickle', mode='wb') as f:
with open('big_data.pickle', mode='wb') as f:
    pickle.dump(vector_images, f)

print("finished")
