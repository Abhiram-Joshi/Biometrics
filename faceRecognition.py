import cv2
from simple_facerec import SimpleFacerec
import os

sfr = SimpleFacerec()

file_read = open("users.txt", "r")
file_read.seek(0)
content = file_read.readlines()

for i in content:
    name = i.strip().split("-")[0]
    print(name)
    sfr.load_encoding_images(os.path.join("users", name))

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    face_locations, face_name = sfr.detect_known_faces(frame)
    for face_loc, name in zip(face_locations, face_name):
        y1, x1, y2, x2 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("frame", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()