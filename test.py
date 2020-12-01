import pandas as pd

data = pd.read_table('annotations/identity_CelebA.txt',
                     delim_whitespace=True)
print(data.identity.value_counts())

#image, locations, eyes = detect_faces(filepath)
#top, right, bottom, left = locations[0]
#image = cv2.rectangle(image, (left, bottom), (right, top), (255, 0, 0), 5)
#matrix = get_rotation_matrix(eyes[0], LEFT_EYE_LOCATION, FACE_WIDTH, FACE_HEIGHT)
#image_aligned = cv2.warpAffine(image, matrix, (FACE_WIDTH, FACE_HEIGHT), flags=cv2.INTER_CUBIC)
#encoding = get_face_encoding(image_aligned)
#pil_image = Image.fromarray(image)
#pil_image.show()