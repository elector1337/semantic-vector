from flask import Flask, jsonify, Response, render_template_string
import cv2
import serial
import atexit
import subprocess
import sys
from pathlib import Path
from collections import deque
from datetime import datetime
from threading import Thread
from camera import Camera
from filter import FrameProcessor

app = Flask(__name__)

PROJECT_DIR = Path(__file__).resolve().parent
TEST_SCRIPT = PROJECT_DIR / "test_script.py"
logs = deque(maxlen=100)


def add_log(message):
    """Сохраняет строку для отображения на главной странице."""
    logs.append(f"[{datetime.now():%H:%M:%S}] {message}")


def collect_script_output(process):
    """Читает вывод запущенного скрипта, пока тот не завершится."""
    for line in process.stdout:
        add_log(line.rstrip())
    add_log(f"Скрипт завершён с кодом {process.wait()}")

# --- ИНИЦИАЛИЗАЦИЯ КАМЕРЫ ---
camera = Camera()
if not camera.capture(1):
    add_log("Не удалось открыть камеру 1")
atexit.register(camera.release)
robot_thread = None


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

@app.route('/')
def index():
    """Главная страница с видеопотоком и кнопкой запуска тестового скрипта."""
    return render_template_string('''
<!doctype html>
<html lang="ru">
<head>
    <meta charset="utf-8">
    <title>робот</title>
</head>
<body>
    <h1>Камера робота</h1>
    <img src="{{ url_for('video_feed') }}" alt="Видеопоток с камеры">

    <form id="run-script-form" action="{{ url_for('run_script') }}" method="post">
        <button type="submit">Запуск</button>
    </form>

    <h2>Логи</h2>
    <pre id="logs">Загрузка логов...</pre>

    <script>
        async function updateLogs() {
            const response = await fetch('{{ url_for('get_logs') }}');
            const data = await response.json();
            document.getElementById('logs').textContent = data.logs.join('\n');
        }

        updateLogs();
        setInterval(updateLogs, 1000);

        document.getElementById('run-script-form').addEventListener('submit', async (event) => {
            event.preventDefault();
            await fetch(event.currentTarget.action, { method: 'POST' });
            updateLogs();
        });
    </script>
</body>
</html>
''')


@app.route('/run-script', methods=['POST'])
def run_script():
    """Запускает управление роботом в потоке с общей камерой сервера."""
    global robot_thread
    if robot_thread and robot_thread.is_alive():
        return jsonify({"status": "already_running"}), 409

    from run_robot import main as run_robot

    robot_thread = Thread(target=run_robot, args=(camera,), daemon=True)
    robot_thread.start()
    add_log("Запущено управление роботом с общей камерой")
    return jsonify({"status": "started", "script": "run_robot.py"})


@app.route('/logs')
def get_logs():
    """Возвращает последние сообщения для блока логов."""
    return jsonify({"logs": list(logs)})


@app.route('/video_feed')
def video_feed():
    """Эндпоинт для отдачи видеопотока"""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    try:
        print("Запуск сервера управления роботом...")
        app.run(host='0.0.0.0', port=5010, debug=False)
    except Exception as e:
        RuntimeError(e)
