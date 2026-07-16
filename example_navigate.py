import cv2
import numpy as np
import argparse
from filter import CameraPreprocessor
from trajectory import detect_red_points, matrix
from vector import VectorNavigator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', '-c', default="http://192.168.2.2:5000/video_feed",
                        help='Camera index or URL (default: robot video feed)')
    parser.add_argument('--warp', action='store_true', help='Apply trajectory perspective warp before detection')
    args = parser.parse_args()

    cam = CameraPreprocessor(args.camera)

    vn = None
    robot_x = None
    robot_y = None
    robot_theta = 0.0

    try:
        while True:
            original_frame, processed_frame = cam.get_processed_hsv()
            if original_frame is None:
                break

            perspective = matrix if args.warp else None
            pts, vis, red_mask = detect_red_points(processed_frame, perspective_matrix=perspective)

            if vn is None and len(pts) >= 4:
                # Создаём навигатор по найденным точкам
                vn = VectorNavigator(pts, lookahead_distance=30.0)
                # Инициализируем простую стартовую позицию робота
                robot_x = float(pts[0][0])
                robot_y = float(pts[0][1]) + 50.0

            if vn is not None:
                vl, vr = vn.compute_action(robot_x, robot_y, robot_theta)
                # Печатаем команды
                print(f"vl={vl:.3f}, vr={vr:.3f}")

                # Простая аппроксимация движения (для демонстрации): двигаемся вперёд
                speed = (vl + vr) / 2.0
                robot_x += speed * 5.0 * np.cos(robot_theta)
                robot_y += speed * 5.0 * np.sin(robot_theta)

                # Нарисуем позицию робота
                if vis is not None:
                    cv2.circle(vis, (int(robot_x), int(robot_y)), 6, (255, 255, 0), -1)

            # Показываем окна
            cv2.imshow("Original", original_frame)
            cv2.imshow("Processed", processed_frame)
            cv2.imshow("Red Mask", red_mask)
            if vis is not None:
                cv2.imshow("Output with Points", vis)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cam.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
