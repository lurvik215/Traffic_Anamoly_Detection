import os
import sys
import cv2
import numpy as np
import supervision as sv

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from scripts.vehicle_detection import VehicleDetector
from scripts.helmet_detection import HelmetDetector


# ---------------- PATHS ----------------

VEHICLE_MODEL = os.path.join(BASE_DIR, "models/vehicle_detector.pt")
HELMET_MODEL  = os.path.join(BASE_DIR, "models/helmet_detector.pt")

VIDEO_PATH  = os.path.join(BASE_DIR, "data/videos/traffic_video.mp4")
OUTPUT_PATH = os.path.join(BASE_DIR, "results/output_tracked.mp4")


# ---------------- COLORS ----------------

BLUE  = (255,0,0)
GREEN = (0,255,0)
RED   = (0,0,255)


# ---------------- IOU ----------------

def compute_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2-x1) * max(0, y2-y1)

    a1 = (box1[2]-box1[0])*(box1[3]-box1[1])
    a2 = (box2[2]-box2[0])*(box2[3]-box2[1])

    union = a1 + a2 - inter

    if union == 0:
        return 0

    return inter/union


# ---------------- PIPELINE ----------------

def run_pipeline():

    print("Loading models...")

    vehicle_detector = VehicleDetector(VEHICLE_MODEL)
    helmet_detector  = HelmetDetector(HELMET_MODEL)

    tracker = sv.ByteTrack()

    cap = cv2.VideoCapture(VIDEO_PATH)

    fps    = int(cap.get(cv2.CAP_PROP_FPS))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.join(BASE_DIR,"results"), exist_ok=True)

    out = cv2.VideoWriter(
        OUTPUT_PATH,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width,height)
    )

    violation_ids = set()

    while True:

        ret, frame = cap.read()

        if not ret:
            break


        # ---------------- VEHICLE DETECTION ----------------

        detections = vehicle_detector.detect(frame)

        two_wheelers = []
        persons = []

        for det in detections:

            cls = det["class_name"].lower()
            x1,y1,x2,y2 = det["bbox"]
            conf = det["confidence"]

            box = (x1,y1,x2,y2)

            if "two" in cls:
                two_wheelers.append(box)

            elif "person" in cls:
                persons.append(box)

            else:

                label = f"{cls} {conf:.2f}"

                cv2.rectangle(frame,(x1,y1),(x2,y2),BLUE,2)

                cv2.putText(frame,label,(x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            BLUE,
                            2)


        # ---------------- CREATE RIDERS ----------------

        rider_boxes = []
        used_person = set()

        for tw in two_wheelers:

            for i,p in enumerate(persons):

                if i in used_person:
                    continue

                if compute_iou(tw,p) > 0.1:

                    px1,py1,px2,py2 = p
                    tx1,ty1,tx2,ty2 = tw

                    rider = (
                        min(px1,tx1),
                        min(py1,ty1),
                        max(px2,tx2),
                        max(py2,ty2)
                    )

                    rider_boxes.append(rider)

                    used_person.add(i)

                    break


        # ---------------- TRACKING ----------------

        if len(rider_boxes) > 0:

            boxes = np.array(rider_boxes)

            detections_sv = sv.Detections(
                xyxy = boxes,
                confidence = np.ones(len(boxes)),
                class_id = np.zeros(len(boxes))
            )

            tracks = tracker.update_with_detections(detections_sv)

        else:

            tracks = sv.Detections.empty()


        if tracks.tracker_id is None:
            continue


        # ---------------- PROCESS TRACKED RIDERS ----------------

        for box, track_id in zip(tracks.xyxy, tracks.tracker_id):

            x1,y1,x2,y2 = map(int, box)
            track_id = int(track_id)

            cv2.rectangle(frame,(x1,y1),(x2,y2),BLUE,2)

            cv2.putText(frame,
                        f"Rider {track_id}",
                        (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        BLUE,
                        2)

            rider_crop = frame[y1:y2, x1:x2]

            if rider_crop.size == 0:
                continue


            # ---------------- HELMET DETECTION ----------------

            helmet_dets = helmet_detector.detect(rider_crop)

            for det in helmet_dets:

                cls = det["class_name"].lower()

                hx1,hy1,hx2,hy2 = det["bbox"]
                conf = det["confidence"]

                hx1 += x1
                hy1 += y1
                hx2 += x1
                hy2 += y1


                if "goodhelmet" in cls:

                    color = GREEN
                    label = "goodhelmet"

                elif "badhelmet" in cls or "nohelmet" in cls:

                    color = RED
                    label = cls

                    violation_ids.add(track_id)

                else:

                    color = BLUE
                    label = cls


                cv2.rectangle(frame,(hx1,hy1),(hx2,hy2),color,2)

                cv2.putText(frame,
                            label,
                            (hx1,hy1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color,
                            2)


        # ---------------- VIOLATION COUNTER ----------------

        cv2.putText(frame,
                    f"Helmet Violations: {len(violation_ids)}",
                    (30,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    RED,
                    3)


        out.write(frame)

        cv2.imshow("Traffic Monitoring", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break


    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("Output saved to:", OUTPUT_PATH)


# ---------------- RUN ----------------

if __name__ == "__main__":

    run_pipeline()