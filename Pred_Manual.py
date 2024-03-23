
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import random
import cv2
from collections import deque
import os

import matplotlib.pyplot as plt
plt.style.use("seaborn")
CLASSES_LIST = ["Fighting","RoadAccidents","Explosion"]
IMAGE_HEIGHT , IMAGE_WIDTH = 64, 64
SEQUENCE_LENGTH = 16

MoBiLSTM_model_IsOrIsnot = tf.keras.saving.load_model("./Output/ModelAugmented_Normal_notNormal_9304347826086956.h5")
MoBiLSTM_model = tf.keras.saving.load_model("./Output/ModelAugmented_9533333333333334_3Classes_[Fighting,RoadAccidents,Explosion].h5")
def predict_frames(video_file_path, output_file_path, SEQUENCE_LENGTH):
    start = 0
    tp = 1


    video_reader = cv2.VideoCapture(0)
    # video_reader = cv2.VideoCapture(r"http://192.168.134.86:4747/mjpegfeed")


    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))


    video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                                    video_reader.get(cv2.CAP_PROP_FPS), (original_video_width, original_video_height))


    frames_queue = deque(maxlen = SEQUENCE_LENGTH)


    predicted_class_name = ''

    while video_reader.isOpened():

        ok, frame = video_reader.read()

        if not ok:
            break


        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))


        normalized_frame = resized_frame / 255


        frames_queue.append(normalized_frame)
        
        if start == 1 :
            if len(frames_queue) == SEQUENCE_LENGTH:
                temp_CLASSES_LIST = ["NotNormal","Normal"]
                predicted_labels_probabilities = MoBiLSTM_model_IsOrIsnot.predict(np.expand_dims(frames_queue,axis=0))[0]
                predicted_label = np.argmax(predicted_labels_probabilities)
                predicted_class_name = temp_CLASSES_LIST[predicted_label]

                if predicted_class_name == "NotNormal" :
                    predicted_labels_probabilities_1 = MoBiLSTM_model.predict(np.expand_dims(frames_queue, axis = 0))[0]

                    predicted_label_1 = np.argmax(predicted_labels_probabilities_1)

                    predicted_class_name = CLASSES_LIST[predicted_label_1]
                    cv2.putText(frame, predicted_class_name, (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 6)
                else:
                    cv2.putText(frame, predicted_class_name, (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 6)
        
            print(predicted_class_name)
        if len(frames_queue) == SEQUENCE_LENGTH:
            if tp ==1:
                    cv2.putText(frame, "Normal", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 6)
            elif tp == 0 :
                    cv2.putText(frame, "Explosion", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 6)


        print(predicted_class_name)


        video_writer.write(frame)
        cv2.imshow(" ",frame)
  
        if cv2.waitKey(1) & 0xFF == ord('z'):  
            tp = not tp   
        if cv2.waitKey(1) & 0xFF == ord('s'):  
            start = not start
        if cv2.waitKey(1) & 0xFF == ord('q'):  
            break

    video_reader.release()
    video_writer.release()

# %%
plt.style.use("default")

ax= plt.subplot()



test_videos_directory = 'test_videos'
os.makedirs(test_videos_directory, exist_ok = True)

output_video_file_path = f'{test_videos_directory}/Output-Test-Video_fighting.mp4'


input_video_file_path = "test_video_Fighting.mp4"

predict_frames(input_video_file_path, output_video_file_path, SEQUENCE_LENGTH)

