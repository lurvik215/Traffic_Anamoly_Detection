import os
import sys
import cv2
import torch
import torch.nn as nn  
from torchvision import models, transforms
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from PIL import Image
import numpy as np

from deep_sort_realtime.deepsort_tracker import DeepSort

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

# from scripts.vehicle_detection import VehicleDetector
# from scripts.helmet_detection import HelmetDetector

# class HSRPClassifier(nn.Module):
#     def __init__(self):
#         super(HSRPClassifier, self).__init__()
#         # Matches hsrp_nonhsrp.ipynb: MobileNetV3 Small
#         self.model = models.mobilenet_v3_small(weights=None)
#         self.model.classifier[3] = nn.Linear(self.model.classifier[3].in_features, 2)
    
#     def forward(self, x):
#         return self.model(x)

class HSRPClassifier(nn.Module):
    def __init__(self):
        super(HSRPClassifier, self).__init__()
        # 1. Load the base MobileNetV3 Small directly into the class
        # This ensures the keys are 'features.0.0.weight' instead of 'model.features.0.0.weight'
        base_model = models.mobilenet_v3_small(weights=None)
        
        # 2. Copy the parts over so they are top-level attributes
        self.features = base_model.features
        self.avgpool = base_model.avgpool
        self.classifier = base_model.classifier
        
        # 3. Modify the final layer to match your 2-class training
        self.classifier[3] = nn.Linear(self.classifier[3].in_features, 2)
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# -------- PATHS --------
# VEHICLE_MODEL = os.path.join(BASE_DIR, "models/vehicle_detector.pt")
# HELMET_MODEL  = os.path.join(BASE_DIR, "models/helmet_detector.pt")

VEHICLE_MODEL_PATH = os.path.join(BASE_DIR,'notebooks/runs/results/vehicle_detector/weights/best.pt') # from vehicle_training.ipynb
HELMET_MODEL_PATH  = os.path.join(BASE_DIR,'notebooks/runs/runs/helmet_detector3/weights/best.pt')   # from helmet_training.ipynb
PLATE_MODEL_PATH   = os.path.join(BASE_DIR,'notebooks/runs/plate/plate_detector/weights/best.pt')    # from plate_training.ipynb
HSRP_WEIGHTS_PATH  = os.path.join(BASE_DIR,'models/hsrp_classifier.pth')                # from hsrp_nonhsrp.ipynb

VIDEO_PATH  = os.path.join(BASE_DIR, "data/videos/traffic3.mp4")
OUTPUT_PATH = os.path.join(BASE_DIR, "results/output_deepsort4.mp4")

# Live View Settings
SHOW_VIDEO = True  # Set to False if running on a server without a monitor
DISPLAY_WIDTH = 1280

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- MODEL INITIALIZATION ---
vehicle_model = YOLO(VEHICLE_MODEL_PATH)
helmet_model  = YOLO(HELMET_MODEL_PATH)
plate_model   = YOLO(PLATE_MODEL_PATH)

# Initialize DeepSORT
# max_age: How many frames to keep a 'lost' object alive
# n_init: Number of frames before a track is 'confirmed'
tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0, max_iou_distance=0.7)


hsrp_net = HSRPClassifier().to(device)
hsrp_net.load_state_dict(torch.load(HSRP_WEIGHTS_PATH, map_location=device))
hsrp_net.eval()

# HSRP Preprocessing (Standard ResNet18 transforms)
hsrp_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# -------- COLORS --------
# BLUE  = (255,0,0)
# GREEN = (0,255,0)
# RED   = (0,0,255)


# -------- IOU --------
# def compute_iou(box1, box2):

#     x1 = max(box1[0], box2[0])
#     y1 = max(box1[1], box2[1])
#     x2 = min(box1[2], box2[2])
#     y2 = min(box1[3], box2[3])

#     inter = max(0, x2-x1) * max(0, y2-y1)

#     a1 = (box1[2]-box1[0])*(box1[3]-box1[1])
#     a2 = (box2[2]-box2[0])*(box2[3]-box2[1])

#     union = a1 + a2 - inter

#     if union == 0:
#         return 0

#     return inter/union


# -------- MAIN --------
def pipeline_with_tracking(video_input, video_output):
    cap = cv2.VideoCapture(video_input)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_input}")
        return
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps   = int(cap.get(cv2.CAP_PROP_FPS))
    os.makedirs(os.path.dirname(video_output), exist_ok=True)
    out = cv2.VideoWriter(video_output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # STEP 1: Detect Vehicles
        results = vehicle_model(frame, verbose=False)[0]
        
        # Format detections for DeepSORT: [([x1, y1, w, h], conf, class_id), ...]
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            detections.append(([x1, y1, w, h], conf, cls))

        # STEP 2: Update Tracker
        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:

            if not track.is_confirmed():
                continue
            
            # Get track info
            track_id = track.track_id
            ltrb = track.to_ltrb() # Left, Top, Right, Bottom
            vx1, vy1, vx2, vy2 = map(int, ltrb)
            v_cls = track.get_det_class()
            v_label = vehicle_model.names[v_cls]
            
            # Ensure crop is within frame boundaries
            v_crop = frame[max(0, vy1):min(height, vy2), max(0, vx1):min(width, vx2)]
            if v_crop.size == 0: continue

            anomalies = []

            # STEP 3: Sub-Inference (Only if track is valid)
            # Two-wheeler check (Class 8 from your data.yaml)
            if v_cls == 8:
                h_results = helmet_model(v_crop, verbose=False)[0]
                h_classes = [int(b.cls[0]) for b in h_results.boxes]
                # Labels from helmet_data.yaml: 1: no_helmet, 3: bad_helmet
                if 1 in h_classes or 3 in h_classes:
                    anomalies.append("NO HELMET")

            # Plate & HSRP check
            p_results = plate_model(v_crop, verbose=False)[0]
            for p_box in p_results.boxes:
                px1, py1, px2, py2 = map(int, p_box.xyxy[0])
                p_crop = v_crop[max(0, py1):min(v_crop.shape[0], py2), 
                                max(0, px1):min(v_crop.shape[1], px2)]
                if p_crop.size > 0:
                    # Run your HSRP Classifier here...
                    # if hsrp_pred == 1: anomalies.append("NON-HSRP")
                    pil_plate = Image.fromarray(cv2.cvtColor(p_crop, cv2.COLOR_BGR2RGB))
                    p_tensor = hsrp_transform(pil_plate).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        pred = hsrp_net(p_tensor)
                        status = torch.argmax(pred, dim=1).item()
                        # Label 1: non-hsrp
                        if status == 1:
                            anomalies.append("NON-HSRP")
                    

            # STEP 4: Draw with ID
            color = (0, 0, 255) if anomalies else (0, 255, 0)
            cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), color, 2)
            
            display_text = f"ID:{track_id} {v_label}"
            if anomalies: display_text += " | " + " & ".join(anomalies)
            
            cv2.putText(frame, display_text, (vx1, vy1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        out.write(frame)
        # Live View
        if SHOW_VIDEO:
            # Scale down for display only
            aspect_ratio = height / width
            dim = (DISPLAY_WIDTH, int(DISPLAY_WIDTH * aspect_ratio))
            resized_frame = cv2.resize(frame, dim)
            
            cv2.imshow("Traffic Anomaly Detection", resized_frame)
            
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Processing interrupted by user.")
                break


    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    pipeline_with_tracking(VIDEO_PATH,OUTPUT_PATH)