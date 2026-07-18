from flask import Flask, jsonify, Response
import cv2
import serial
import atexit

app = Flask(__name__)

SERIAL_PORT = '/dev/ttyUSB0'
BAUD_RATE = 115200

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f"[SERIAL] Подключено к порту {SERIAL_PORT} на скорости {BAUD_RATE}")
except serial.SerialException as e:
    print(f"[SERIAL ОШИБКА] Не удалось открыть порт {SERIAL_PORT}: {e}")
    ser = None


# Закрываем порт при остановке программы
def close_serial():
    if ser and ser.is_open:
        ser.write(b"0,0\n")  # Остановка моторов перед закрытием
        ser.close()
        print("[SERIAL] Порт закрыт.")

atexit.register(close_serial)

# --- ИНИЦИАЛИЗАЦИЯ КАМЕРЫ ---
camera = cv2.VideoCapture(1)  # Поменяйте индекс, если меняли ранее
camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if not ret:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


# --- УПРАВЛЕНИЕ ЖЕЛЕЗОМ (МОТОРАМИ) ---
def set_motors(left_speed, right_speed):
    """Отправляет скорости на микроконтроллер через Serial"""
    print(f"[УПРАВЛЕНИЕ] L: {left_speed} | R: {right_speed}")

    if ser and ser.is_open:
        command = f"{int(left_speed)},{int(right_speed)}\n"
        try:
            ser.write(command.encode('utf-8'))
        except serial.SerialTimeoutException:
            print("[SERIAL ОШИБКА] Таймаут отправки данных!")


# --- ЭНДПОИНТЫ ДАННЫХ И УПРАВЛЕНИЯ ---
@app.route('/video_feed')
def video_feed():
    """Эндпоинт для отдачи видеопотока"""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# Параметры передаются прямо в адресе, например: /move/50/-50
@app.route('/move/<int:left>/<int:right>')
def move(left, right):
    """Эндпоинт для движения: параметры читаются прямо из URL пути"""
    set_motors(left, right)
    return jsonify({"status": "moving", "left": left, "right": right})


@app.route('/stop')
def stop():
    """Эндпоинт для экстренной остановки робота"""
    set_motors(0, 0)
    print("[УПРАВЛЕНИЕ] Робот остановлен")
    return jsonify({"status": "stopped"})


if __name__ == '__main__':
    try:
        print("Запуск сервера управления роботом...")
        app.run(host='0.0.0.0', port=5010, debug=False)
    except Exception as e:
        RuntimeError(e)
