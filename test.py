import face_recognition
import cv2
import numpy as np
from PIL import Image

LEFT_EYE_LOCATION = (0.35, 0.35)
FACE_WIDTH = 252
FACE_HEIGHT = 252

# load an image as an arrary
image = face_recognition.load_image_file("DSC_4121.jpeg")

# detect faces from input image.
locations = face_recognition.face_locations(image, model="hog")
landmarks = face_recognition.face_landmarks(image)

for (top, right, bottom, left), marks in zip(locations, landmarks):
    image = cv2.rectangle(image, (left, bottom), (right, top), (255, 0, 0), 5)
    left_eye = marks['left_eye']
    right_eye = marks['right_eye']

    left_eye_center = np.array(left_eye).mean(axis=0).astype("int")
    right_eye_center = np.array(right_eye).mean(axis=0).astype("int")

    left_eye_center = (left_eye_center[0], left_eye_center[1])
    right_eye_center = (right_eye_center[0], right_eye_center[1])

    dY = right_eye_center[1] - left_eye_center[1]
    dX = right_eye_center[0] - left_eye_center[0]
    angle = np.degrees(np.arctan2(dY, dX))
    dist = np.sqrt((dX ** 2) + (dY ** 2))
    new_right_eye_x = 1.0 - LEFT_EYE_LOCATION[0]
    scale = (new_right_eye_x - LEFT_EYE_LOCATION[0]) * FACE_WIDTH / dist

    center = ((left_eye_center[0] + right_eye_center[0]) // 2, (left_eye_center[1] + right_eye_center[1]) // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)

    rotation_matrix[0, 2] += (FACE_WIDTH * 0.5 - center[0])
    rotation_matrix[1, 2] += (FACE_HEIGHT * LEFT_EYE_LOCATION[1] - center[1])

    # apply the affine transformation
    output = cv2.warpAffine(image, rotation_matrix, (FACE_WIDTH, FACE_HEIGHT), flags=cv2.INTER_CUBIC)
    encoding = face_recognition.face_encodings(output)[0]
    print(encoding)

image = cv2.resize(image, (860, 540))
pil_image = Image.fromarray(image)
pil_image.show()
pil_image_face = Image.fromarray(output)
pil_image_face.show()