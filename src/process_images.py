"""
A script for processing images:
creating a dictionary of hog-predicted face bounding boxes
and a dictionary of 128D face encodings and pickles them.
"""

# import modules needed
from utils import *
from tqdm import tqdm
import pickle
import cv2
import os

# define constants and variables
LEFT_EYE_LOCATION = (0.35, 0.35)
FACE_WIDTH = 252
FACE_HEIGHT = 252
images_directory = '../examples'

# create data holders
detected_boxes = {'image_id': [], 'x_1': [], 'y_1': [],
                   'width': [], 'height': []}
face_encodings = {'image_id': [], 'encoding': []}

# iterate over photos
files = os.listdir(images_directory)
for filename in tqdm(files):
    if filename.endswith(".jpg"):
        filepath = os.path.join(images_directory, filename)
        image, locations, eyes = detect_faces(filepath)
        if len(locations) == 1:
            top, right, bottom, left = locations[0]
            detected_boxes = add_dict_entry(detected_boxes, filename,
                                             left, top, right-left, bottom-top)
            matrix = get_rotation_matrix(eyes[0], LEFT_EYE_LOCATION,
                                         FACE_WIDTH, FACE_HEIGHT)
            image_aligned = cv2.warpAffine(image, matrix,
                                           (FACE_WIDTH, FACE_HEIGHT),
                                           flags=cv2.INTER_CUBIC)
            encoding = get_face_encoding(image_aligned)
            face_encodings['image_id'].append(filename)
            face_encodings['encoding'].append(encoding)
        elif len(locations) == 0:
            detected_boxes = add_dict_entry(detected_boxes, filename,
                                             0, 0, 0, 0)
        else:
            print(f"Problem z {filename}, liczba twarzy: {len(locations)}")

# write bounding boxes dictionary to a file
with open('../data/hog_bounding_boxes_dict.pkl', 'wb') as f:
    pickle.dump(detected_boxes, f)
# write encodings dictionary to a file
with open('../data/face_encodings.pkl', 'wb') as f:
    pickle.dump(face_encodings, f)
