import os
import cv2
from ultralytics import YOLO


# ---------------------------------------------------
# Project Paths
# ---------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "helmet_detector.pt")
VIDEO_PATH = os.path.join(BASE_DIR, "data","videos", "traffic_video.mp4")
OUTPUT_PATH = os.path.join(BASE_DIR, "results", "helmet_output.mp4")


# ---------------------------------------------------
# Helmet Detector Class
# ---------------------------------------------------

class HelmetDetector:

    def __init__(self, model_path):

        print("Loading helmet detection model...")
        self.model = YOLO(model_path)

        # class names
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


# ---------------------------------------------------
# Draw detections
# ---------------------------------------------------

def draw_detections(frame, detections):

    for d in detections:

        x1, y1, x2, y2 = d["bbox"]
        label = f"{d['class_name']} {d['confidence']:.2f}"

        color = (0,255,0)

        if d["class_name"] == "no-helmet":
            color = (0,0,255)

        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)

        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    return frame


# ---------------------------------------------------
# Run detection on video
# ---------------------------------------------------

def test_video():

    detector = HelmetDetector(MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("Error opening video")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    os.makedirs(os.path.join(BASE_DIR, "results"), exist_ok=True)

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

        cv2.imshow("Helmet Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("Output saved to:", OUTPUT_PATH)


# ---------------------------------------------------
# Run script
# ---------------------------------------------------

if __name__ == "__main__":
    test_video()