import face_recognition
import numpy as np
import cv2


def detect_faces(filepath):
    """
    Detects faces on a given photograph, along with their landmarks.
    :param filepath: path to the image
    :return: (image, locations, eyes) where locations is a list of face bounding boxes
             in a form of a tuple (top, right, bottom, left), and eyes is a list
             of dictionaries with keys 'left_eye' and 'right_eye'
    """
    image = face_recognition.load_image_file(filepath)
    locations = face_recognition.face_locations(image, model="hog")
    if len(locations) == 0:
        return image, [], {}
    landmarks = face_recognition.face_landmarks(image)
    eyes = [{'left_eye': marks['left_eye'], 'right_eye': marks['right_eye']}
            for marks in landmarks]
    return image, locations, eyes


def get_rotation_matrix(eyes, desired_left_eye_location=(0.35, 0.35),
                        desired_face_width=252, desired_face_height=252):
    """
    Calculates a rotation matrix based on given parameters.
    :param eyes: dictionary with keys 'left_eye', 'right_eye'
    :param desired_left_eye_location: tuple, new left eye location in a warped image
    :param desired_face_width: int, new img width
    :param desired_face_height: int, new img height
    :return: rotation matrix
    """
    left_eye = eyes['left_eye']
    right_eye = eyes['right_eye']
    left_eye_center = np.array(left_eye).mean(axis=0).astype("int")
    right_eye_center = np.array(right_eye).mean(axis=0).astype("int")
    left_eye_center = (left_eye_center[0], left_eye_center[1])
    right_eye_center = (right_eye_center[0], right_eye_center[1])
    dY = right_eye_center[1] - left_eye_center[1]
    dX = right_eye_center[0] - left_eye_center[0]
    angle = np.degrees(np.arctan2(dY, dX))
    dist = np.sqrt((dX ** 2) + (dY ** 2))
    new_right_eye_x = 1.0 - desired_left_eye_location[0]
    scale = (new_right_eye_x - desired_left_eye_location[0]) * desired_face_width / dist
    center = ((left_eye_center[0] + right_eye_center[0]) // 2, (left_eye_center[1] + right_eye_center[1]) // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)

    rotation_matrix[0, 2] += (desired_face_width * 0.5 - center[0])
    rotation_matrix[1, 2] += (desired_face_height * desired_left_eye_location[1] - center[1])
    return rotation_matrix


def get_face_encoding(face_image):
    """
    Creates a 128-dimensional encoding for a given face image.
    :param face_image: aligned face image
    :return: face encoding
    """
    encoding = face_recognition.face_encodings(face_image)[0]
    return encoding


def add_dict_entry(dictionary, image_id, x1, y1, width, height):
    """
    Add an entry to the bounding boxes dictionary.
    :param dictionary: an already existing bboxes dict
    :param image_id: name of an image
    :param x1: left upper corner x-coordinate
    :param y1: left upper corner y-coordinate
    :param width: bbox width
    :param height: bbox height
    :return: updated dictionary
    """
    dictionary['image_id'].append(image_id)
    dictionary['x_1'].append(x1)
    dictionary['y_1'].append(y1)
    dictionary['width'].append(width)
    dictionary['height'].append(height)
    return dictionary
