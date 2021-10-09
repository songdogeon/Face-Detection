import cv2, dlib, sys
import numpy as np

scaler = 0.7

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

ALL = list(range(0, 68))

index = ALL

while True:
    ret, img = cap.read()
    if not ret:
        break

    img = cv2.resize(img, (int(img.shape[1] * scaler), int(img.shape[0] * scaler)))

    img = cv2.flip(img, 1)
    original = img.copy()

    faces = detector(img)

    dets = detector(img, 1)

    for face in dets:
        shape = predictor(img, face)  # 얼굴에서의 랜드마크 찾기 68개점

        list_points = []  # 검출된 랜드마크를 리스트에 저장
        for p in shape.parts():
            list_points.append([p.x, p.y])

        list_points = np.array(list_points)  # 리스트를 넘파이 배열로 변환

        for i, pt in enumerate(list_points[index]):  # 지정된 부분에 원으로 표시해줌
            pt_pos = (pt[0], pt[1])
            cv2.circle(img, pt_pos, 2, (0, 255, 0), -1)

        img = cv2.rectangle(img, (face.left(), face.top()), (face.right(), face.bottom()), (0, 0, 255), 3)


    cv2.imshow('CAM', img)
    cv2.waitKey(1)