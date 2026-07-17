import cv2
import numpy as np

# ============ НАСТРОЙКИ ============
ARUCO_SIZE = 15  # в см
DIST_COEFFS = np.array([[0.76721061, -7.07075198, -0.028711, -0.07152724, 25.9788756]],
                       dtype=np.float32)  # коэффициенты дисторсии камеры
CAMERA_MATRIX = np.array([[819.95272854, 0., 320.0],
                          [0., 796.39696541, 240.0],
                          [0., 0., 1.]], dtype=np.float32)  # матрица камеры

# запускаем детектор ArUCo
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

cam = cv2.VideoCapture("http://10.136.121.14:5000/video_feed")  # подключаемся к камере
if not cam.isOpened():
    print("Не удалось открыть камеру")
    exit()

object_points = np.array([[ARUCO_SIZE / 2, ARUCO_SIZE / 2, 0],
                          [ARUCO_SIZE / 2, -ARUCO_SIZE / 2, 0],
                          [-ARUCO_SIZE / 2, -ARUCO_SIZE / 2, 0],
                          [-ARUCO_SIZE / 2, ARUCO_SIZE / 2, 0]], dtype=np.float32)  # 3D координаты углов ArUCo в см

while True:
    ret, frame = cam.read()
    if not ret:
        print("Не удалось получить кадр")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is not None:
        for i in range(len(ids)):
            success, rvec, tvec = cv2.solvePnP(object_points, corners[i][0], CAMERA_MATRIX, DIST_COEFFS)

            if success:
                # tvec содержит координаты [X, Y, Z] в сантиметрах
                # Z - расстояние прямо
                distance_z = tvec[2][0]
                x_offset = tvec[0][0]
                y_offset = tvec[1][0]
    else:
        cv2.putText(frame, "No markers detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Robot Camera View", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()