
# %%
import os
import numpy as np
import cv2
import glob


def virticle(a):
    folder_path = "./Dataset/"+a
    videos = glob.glob(os.path.join(folder_path, "*.mp4"))
    print(videos)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 30
    for video in videos:
        cap = cv2.VideoCapture(video)

        base_filename = os.path.splitext(os.path.basename(video))[0]

        output_path = os.path.join("./Augmented_Data/"+a, f"{base_filename}_outputver.mp4")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            flipped_frame = cv2.flip(frame, 0)

            out.write(flipped_frame)

            frame_count += 1

        cap.release()
        out.release()


# %%
"""
Horizontal
"""
def Horizontal(a):
    folder_path =  "./Dataset/"+a


    videos = glob.glob(os.path.join(folder_path, "*.mp4"))


    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 30


    for video in videos:

        cap = cv2.VideoCapture(video)


        base_filename = os.path.splitext(os.path.basename(video))[0]


        output_path = os.path.join("./Augmented_Data/"+a, f"{base_filename}_outputhorizontal.mp4")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            flipped_frame = cv2.flip(frame, 1)

            out.write(flipped_frame)


            frame_count += 1


        cap.release()
        out.release()


# %%
"""
Sheared
"""

def Sheared(a):
    folder_path =  "./Dataset/"+a



    shear_angle = 20 
    shear_matrix = np.array([[1, np.tan(np.deg2rad(shear_angle)), 0],
                            [0, 1, 0]])


    videos = glob.glob(os.path.join(folder_path, "*.mp4"))


    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 30


    for video in videos:
        cap = cv2.VideoCapture(video)
        base_filename = os.path.splitext(os.path.basename(video))[0]


        output_path = os.path.join("./Augmented_Data/"+a, f"{base_filename}_outputsheared.mp4")


        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        while True:

            ret, frame = cap.read()

            if not ret:

                break


            sheared_frame = cv2.warpAffine(frame, shear_matrix, (width, height))




            out.write(sheared_frame)


            frame_count += 1


        cap.release()
        out.release()


# %%

def brightness(a):
    folder_path = "./Dataset/"+a

    output_path = "./Augmented_Data/"+a
    brightness_factor = 130  # adjust brightness by adding this value to the pixel intensities

    videos = glob.glob(os.path.join(folder_path, "*.mp4"))

    for video in videos:
        cap = cv2.VideoCapture(video)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS)
        base_filename = os.path.splitext(os.path.basename(video))[0]
        output_file = os.path.join(output_path, f"{base_filename}_output.mp4")
        out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

        frame_count = 0
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            brightness_adjusted_frame = cv2.add(frame, brightness_factor)


            out.write(brightness_adjusted_frame)

            frame_count += 1

        cap.release()
        out.release()

l = ["Explosion","Fighting","RoadAccidents","Normal"]
# l = ["Normal"]
for s in l:
    virticle(s)
    Horizontal(s)
    Sheared(s)
    brightness(s)
    print("Done",s)

# videos = glob.glob(os.path.join("./Dataset/"+l[0], "*.mp4"))
# print(videos)

