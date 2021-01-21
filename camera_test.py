import face_recognition
import cv2
import numpy as np
import pickle
import pandas as pd

SCALE = 0.6
ACCEPTANCE_THRESHOLD = 0.6
NAME = 'daria'
vc = cv2.VideoCapture(0)
width = vc.get(cv2.CAP_PROP_FRAME_WIDTH)
height = vc.get(cv2.CAP_PROP_FRAME_HEIGHT)
frame_dims = (round(width * SCALE), round(height * SCALE))
process = False

# load encodings
with open('./data/my_photos_face_encodings.pkl', 'rb') as f:
    my_encodings = pickle.load(f)['encoding']
with open('./data/encodings_dataframe.pkl', 'rb') as f:
    encodings_df = pickle.load(f)
my_encodings_df = pd.DataFrame({'encoding': my_encodings,
                                'person': len(my_encodings) * [NAME]})
df = pd.concat([encodings_df, my_encodings_df], ignore_index=True)
known_encodings = df.encoding.tolist()
identities = df.person.tolist()

ok = True
while ok:
    _, frame = vc.read()
    # Convert the image from BGR to RGB color
    resized = cv2.resize(frame, frame_dims)
    rgb_frame = resized[:, :, ::-1]

    if process:
        faces_found = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, faces_found)

        for (top, right, bottom, left), face_encoding in zip(faces_found, face_encodings):
            cv2.rectangle(resized, (left, top), (right, bottom), (0, 255, 255), 2)
            distances = face_recognition.face_distance(known_encodings, face_encoding)
            best = np.argmin(distances)
            name = 'nieznany'
            if distances[best] < ACCEPTANCE_THRESHOLD:
                name = identities[best]
            cv2.putText(resized, name, (left + 10, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 1)

    process = not process
    cv2.imshow('face recognition', resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to quit
        break

vc.release()
cv2.destroyAllWindows()
