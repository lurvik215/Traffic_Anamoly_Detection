import os
import cv2
from ultralytics import YOLO


# -------------------------------------
# Project Paths
# -------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "plate_detector.pt")
VIDEO_PATH = os.path.join(BASE_DIR, "data","videos", "traffic_video.mp4")
OUTPUT_PATH = os.path.join(BASE_DIR, "results", "plate_output.mp4")


# -------------------------------------
# Plate Detector Class
# -------------------------------------

class PlateDetector:

    def __init__(self, model_path):

        print("Loading plate detection model...")
        self.model = YOLO(model_path)
        self.class_names = self.model.names

    def detect(self, frame):

        results = self.model(frame, conf=0.25)

        detections = []

        for r in results:

            boxes = r.boxes.xyxy.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()

            for box, cls, score in zip(boxes, classes, scores):

                x1, y1, x2, y2 = map(int, box)

                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "class_id": int(cls),
                    "class_name": self.class_names[int(cls)],
                    "confidence": float(score)
                })

        return detections


# -------------------------------------
# Draw detections
# -------------------------------------

def draw_detections(frame, detections):

    for d in detections:

        x1, y1, x2, y2 = d["bbox"]
        label = f"{d['class_name']} {d['confidence']:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )

    return frame


# -------------------------------------
# Run detection on video
# -------------------------------------

def test_video():

    detector = PlateDetector(MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("Error opening video")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.join(BASE_DIR, "results"), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(
        OUTPUT_PATH,
        fourcc,
        fps,
        (width, height)
    )

    print("Processing video...")

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        detections = detector.detect(frame)

        frame = draw_detections(frame, detections)

        out.write(frame)

        cv2.imshow("Plate Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("Output saved to:", OUTPUT_PATH)


# -------------------------------------
# Run script
# -------------------------------------

if __name__ == "__main__":
    test_video()