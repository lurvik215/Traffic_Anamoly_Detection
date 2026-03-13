import os
import cv2
from ultralytics import YOLO


# ---------------------------------------------------
# Get project root directory
# ---------------------------------------------------


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "vehicle_detector.pt")

VIDEO_PATH = os.path.join(BASE_DIR, "data","videos","traffic_video.mp4")

OUTPUT_PATH = os.path.join(BASE_DIR,"videos","output_video.mp4")
print("Loading model from:", MODEL_PATH)



# ---------------------------------------------------
# Vehicle Detector Class
# ---------------------------------------------------

class VehicleDetector:

    def __init__(self, MODEL_PATH):

        print("Loading vehicle detection model...")
        self.model = YOLO(MODEL_PATH)
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
# Draw bounding boxes
# ---------------------------------------------------

def draw_detections(frame, detections):

    for d in detections:

        x1, y1, x2, y2 = d["bbox"]
        conf = d["confidence"]
        label_name = d["class_name"]

        label = f"{label_name} {conf:.2f}"

        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0,255,0),
            2
        )

    return frame


# ---------------------------------------------------
# Crop vehicle regions (for future pipeline)
# ---------------------------------------------------

def crop_vehicles(frame, detections):

    crops = []

    for d in detections:

        x1, y1, x2, y2 = d["bbox"]

        crop = frame[y1:y2, x1:x2]

        crops.append(crop)

    return crops


# ---------------------------------------------------
# Run detection on video
# ---------------------------------------------------

def run_video_detection():

    detector = VehicleDetector(MODEL_PATH)

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

        # crop vehicles for future modules
        vehicle_crops = crop_vehicles(frame, detections)

        frame = draw_detections(frame, detections)

        out.write(frame)

        cv2.imshow("Vehicle Detection", frame)

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
    run_video_detection()