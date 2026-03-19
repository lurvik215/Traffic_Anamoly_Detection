import os
import sys
import cv2
import numpy as np

from deep_sort_realtime.deepsort_tracker import DeepSort

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from scripts.vehicle_detection import VehicleDetector
from scripts.helmet_detection import HelmetDetector


# -------- PATHS --------
VEHICLE_MODEL = os.path.join(BASE_DIR, "models/vehicle_detector.pt")
HELMET_MODEL  = os.path.join(BASE_DIR, "models/helmet_detector.pt")

VIDEO_PATH  = os.path.join(BASE_DIR, "data/videos/traffic2.mp4")
OUTPUT_PATH = os.path.join(BASE_DIR, "results/output_deepsort.mp4")


# -------- COLORS --------
BLUE  = (255,0,0)
GREEN = (0,255,0)
RED   = (0,0,255)


# -------- IOU --------
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


# -------- MAIN --------
def run_pipeline():

    print("Loading models...")

    vehicle_detector = VehicleDetector(VEHICLE_MODEL)
    helmet_detector  = HelmetDetector(HELMET_MODEL)

    tracker = DeepSort(
        max_age=30,
        n_init=3,
        max_cosine_distance=0.4
    )

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
    track_history = {}

    while True:

        ret, frame = cap.read()

        if not ret:
            break


        # -------- VEHICLE DETECTION --------

        detections = vehicle_detector.detect(frame)

        two_wheelers = []
        persons = []

        for det in detections:

            if det["confidence"] < 0.5:
                continue

            cls = det["class_name"].lower()
            x1,y1,x2,y2 = det["bbox"]

            if "two" in cls:
                two_wheelers.append((x1,y1,x2,y2))

            elif "person" in cls:
                persons.append((x1,y1,x2,y2))

            else:
                cv2.rectangle(frame,(x1,y1),(x2,y2),BLUE,2)
                cv2.putText(frame,cls,(x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,BLUE,2)


        # -------- CREATE RIDERS --------

        rider_boxes = []
        used_person = set()

        for tw in two_wheelers:

            for i,p in enumerate(persons):

                if i in used_person:
                    continue

                if compute_iou(tw,p) > 0.2:

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


        # -------- DEEPSORT INPUT --------

        ds_detections = []

        for box in rider_boxes:

            x1,y1,x2,y2 = box
            w = x2 - x1
            h = y2 - y1

            ds_detections.append(([x1,y1,w,h], 0.9, "rider"))


        tracks = tracker.update_tracks(ds_detections, frame=frame)


        # -------- PROCESS TRACKS --------

        for track in tracks:

            if not track.is_confirmed():
                continue

            track_id = track.track_id

            l,t,r,b = map(int, track.to_ltrb())


            # -------- SMOOTHING --------

            if track_id in track_history:

                px1,py1,px2,py2 = track_history[track_id]

                l = int(0.7*px1 + 0.3*l)
                t = int(0.7*py1 + 0.3*t)
                r = int(0.7*px2 + 0.3*r)
                b = int(0.7*py2 + 0.3*b)

            track_history[track_id] = (l,t,r,b)


            cv2.rectangle(frame,(l,t),(r,b),BLUE,2)
            cv2.putText(frame,f"Rider {track_id}",(l,t-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,BLUE,2)


            # -------- HELMET DETECTION --------

            crop = frame[t:b, l:r]

            if crop.size == 0:
                continue

            helmet_dets = helmet_detector.detect(crop)

            for det in helmet_dets:

                cls = det["class_name"].lower()
                hx1,hy1,hx2,hy2 = det["bbox"]

                hx1 += l
                hy1 += t
                hx2 += l
                hy2 += t


                if "good" in cls:

                    color = GREEN

                elif "bad" in cls or "no" in cls:

                    color = RED
                    violation_ids.add(track_id)

                else:

                    color = BLUE


                cv2.rectangle(frame,(hx1,hy1),(hx2,hy2),color,2)
                cv2.putText(frame,cls,(hx1,hy1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)


        # -------- COUNTER --------

        cv2.putText(frame,
                    f"Violations: {len(violation_ids)}",
                    (30,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    RED,
                    3)


        out.write(frame)
        cv2.imshow("DeepSORT Pipeline", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break


    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    run_pipeline()