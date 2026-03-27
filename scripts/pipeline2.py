import os
import cv2
import torch
import torch.nn as nn  
from torchvision import models, transforms
from ultralytics import YOLO
from PIL import Image
import numpy as np

# --- 1. HSRP CLASSIFIER ARCHITECTURE ---
class HSRPClassifier(nn.Module):
    def __init__(self):
        super(HSRPClassifier, self).__init__()
        base_model = models.mobilenet_v3_small(weights=None)
        self.features = base_model.features
        self.avgpool = base_model.avgpool
        self.classifier = base_model.classifier
        self.classifier[3] = nn.Linear(self.classifier[3].in_features, 2)
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# --- 2. CONFIG & PATHS ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HELMET_MODEL_PATH = os.path.join(BASE_DIR, 'notebooks/runs/runs/helmet_detector3/weights/best.pt')
PLATE_MODEL_PATH  = os.path.join(BASE_DIR, 'notebooks/runs/plate/plate_detector/weights/best.pt')
HSRP_WEIGHTS_PATH = os.path.join(BASE_DIR, 'models/hsrp_classifier.pth')

VIDEO_PATH  = os.path.join(BASE_DIR, "data/videos/traffic2.mp4")
OUTPUT_PATH = os.path.join(BASE_DIR, "results/violation_detection.mp4")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Models
helmet_model = YOLO(HELMET_MODEL_PATH)
plate_model  = YOLO(PLATE_MODEL_PATH)

hsrp_net = HSRPClassifier().to(device)
hsrp_net.load_state_dict(torch.load(HSRP_WEIGHTS_PATH, map_location=device))
hsrp_net.eval()

hsrp_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def run_violation_pipeline(video_input, video_output):
    cap = cv2.VideoCapture(video_input)
    width, height = int(cap.get(3)), int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(video_output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # track_history stores: {track_id: {"helmet": bool, "hsrp": bool}}
    track_history = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # STEP 1: Track Riders (Primary Detection)
        # Using ByteTrack interpolation for smooth boxes
        h_results = helmet_model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)[0]

        if h_results.boxes.id is not None:
            boxes = h_results.boxes.xyxy.cpu().numpy().astype(int)
            ids = h_results.boxes.id.cpu().numpy().astype(int)
            classes = h_results.boxes.cls.cpu().numpy().astype(int)

            for box, track_id, cls in zip(boxes, ids, classes):
                x1, y1, x2, y2 = box
                
                # Process only if this ID is new to save computation
                if track_id not in track_history:
                    violations = []
                    
                    # A. Check Helmet (Based on your classes: 1: no_helmet, 3: bad_helmet)
                    if cls == 1 or cls == 3:
                        violations.append("NO HELMET")

                    # B. Check Plate & HSRP
                    rider_crop = frame[max(0,y1):min(height,y2), max(0,x1):min(width,x2)]
                    if rider_crop.size > 0:
                        p_results = plate_model(rider_crop, verbose=False)[0]
                        for p_box in p_results.boxes:
                            px1, py1, px2, py2 = map(int, p_box.xyxy[0])
                            p_crop = rider_crop[py1:py2, px1:px2]
                            
                            if p_crop.size > 0:
                                pil_p = Image.fromarray(cv2.cvtColor(p_crop, cv2.COLOR_BGR2RGB))
                                p_tensor = hsrp_transform(pil_p).unsqueeze(0).to(device)
                                with torch.no_grad():
                                    hsrp_pred = torch.argmax(hsrp_net(p_tensor), dim=1).item()
                                    if hsrp_pred == 1: # 1: Non-HSRP
                                        violations.append("NON-HSRP")
                    
                    track_history[track_id] = violations

                # STEP 2: Visualization
                current_violations = track_history[track_id]
                color = (0, 0, 255) if current_violations else (0, 255, 0)
                
                # Draw main bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Create label text
                label = f"ID:{track_id}"
                if current_violations:
                    label += " | " + " & ".join(current_violations)
                else:
                    label += " | OK"

                # Draw label background for readability
                cv2.putText(frame, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        out.write(frame)
        cv2.imshow("Violation Detection (Helmet & HSRP)", cv2.resize(frame, (1280, 720)))
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_violation_pipeline(VIDEO_PATH, OUTPUT_PATH)