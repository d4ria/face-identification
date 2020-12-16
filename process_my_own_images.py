"""
A script for processing images:
creating a dictionary of hog-predicted face bounding boxes
and a dictionary of 128D face encodings and pickles them.
"""

# import modules needed
from src.utils import *
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import pandas as pd
import pickle
import cv2
import os

# define constants and variables
LEFT_EYE_LOCATION = (0.35, 0.35)
FACE_WIDTH = 252
FACE_HEIGHT = 252
images_directory = './my_images'
num_processes = 10


def run_imap_multiprocessing(func, argument_list, num_processes):

    pool = Pool(processes=num_processes)

    result_list_tqdm = []
    for result in tqdm(pool.imap(func=func, iterable=argument_list), total=len(argument_list)):
        result_list_tqdm.append(result)

    return result_list_tqdm


def analyze_face(filename):
    filepath = os.path.join(images_directory, filename)
    image, locations, eyes = detect_faces(filepath)
    text = ""
    if len(locations) == 0:  # if no face was found
        box = (0, 0, 0, 0)
        encoding = None
    elif len(locations) > 1:  # if more faces found
        areas = [(loc[1] - loc[3]) * (loc[2] - loc[0]) for loc in locations]
        text = f"Na zdjęciu {filename} znaleziono {len(locations)} twarze o rozmiarach {areas}."
        biggest = np.asarray(areas).argmax()
        locations, eyes = [locations[biggest]], [eyes[biggest]]  # change locs and eyes length to 1

    if len(locations) == 1:  # if only one face found
        top, right, bottom, left = locations[0]
        box = (left, top, right - left, bottom - top)
        matrix = get_rotation_matrix(eyes[0], LEFT_EYE_LOCATION, FACE_WIDTH, FACE_HEIGHT)
        image_aligned = cv2.warpAffine(image, matrix, (FACE_WIDTH, FACE_HEIGHT), flags=cv2.INTER_CUBIC)
        encoding = get_face_encoding(image_aligned)
    return (filename, box, encoding, text)


if __name__ == "__main__":
    print('starting own images analysis')
    # create data holders
    detected_boxes = {'image_id': [], 'x_1': [], 'y_1': [],
                      'width': [], 'height': []}
    face_encodings = {'image_id': [], 'encoding': []}

    filenames = [filename for filename in os.listdir(images_directory) if filename.endswith('.jpg')]

    # iterate over photos
    result_list = run_imap_multiprocessing(func=analyze_face, argument_list=filenames, num_processes=num_processes)

    for result in tqdm(result_list):
        filename, box, encoding, text = result
        if encoding is not None:
            face_encodings['image_id'].append(filename)
            face_encodings['encoding'].append(encoding)
        add_dict_entry(detected_boxes, filename, box[0], box[1], box[2], box[3])

    # write bounding boxes dictionary to a file
    with open('data/my_photos_hog_bounding_boxes_dict.pkl', 'wb') as f:
        pickle.dump(detected_boxes, f)
    # write encodings dictionary to a file
    with open('data/my_photos_face_encodings.pkl', 'wb') as f:
        pickle.dump(face_encodings, f)
