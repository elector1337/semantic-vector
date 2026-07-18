from aruco_detector import ArucoDetector
from cleaner import TrajectoryCleaner
from history import HistoryManager
from generator import DatasetGenerator, PointsStorage
from logging import log
import cv2

IP = "10.248.142.14:5010"

history = HistoryManager(history_length=30)
storage = PointsStorage()
generator = DatasetGenerator(input_window=20, output_window=10)
detector = ArucoDetector(IP) 

def save_dataset():
    storage.save()
    cleaner = TrajectoryCleaner(storage.df).full_clean().save()
    

    X, Y = generator.generate(cleaner.df)

    generator.save("dataset.npz", X, Y)
    

def main():
    log.info("start servise")
    while True:
        detections = detector.get_detections()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            detector.stop()
            log.info("servese stopped")
            save_dataset()
            break
        if detections:
            history.update(detections)
            for detection in detections:
                storage.append(detection)
        else:
            continue

if __name__ == "__main__":
    main()