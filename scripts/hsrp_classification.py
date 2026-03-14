import sys
import os

# Allow scripts to import each other
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

from scripts.helmet_detection import HelmetDetector


# -----------------------------
# Paths
# -----------------------------

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

HSRP_MODEL_PATH = os.path.join(BASE_DIR, "models", "hsrp_classifier.pth")
HELMET_MODEL_PATH = os.path.join(BASE_DIR, "models", "helmet_detector.pt")
VIDEO_PATH = os.path.join(BASE_DIR, "data","videos", "traffic_video.mp4")


GREEN = (0,255,0)
RED = (0,0,255)


# -----------------------------
# HSRP Classifier
# -----------------------------

class HSRPClassifier:

    def __init__(self, model_path=HSRP_MODEL_PATH):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = models.mobilenet_v3_small()

        self.model.classifier[3] = nn.Linear(
            self.model.classifier[3].in_features,
            2
        )

        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

        self.model.to(self.device)
        self.model.eval()

        self.classes = ["hsrp", "non_hsrp"]

        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])

        print("HSRP classifier loaded")


    def predict(self, image):

        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():

            outputs = self.model(image)

            probs = torch.softmax(outputs, dim=1)

            conf, pred = torch.max(probs,1)

        return self.classes[pred.item()], conf.item()


# -----------------------------
# Test on video
# -----------------------------

def test_video():

    cap = cv2.VideoCapture(VIDEO_PATH)

    helmet_detector = HelmetDetector(HELMET_MODEL_PATH)

    classifier = HSRPClassifier()

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        detections = helmet_detector.detect(frame)

        for obj in detections:

            cls = obj["class_name"]

            x1,y1,x2,y2 = obj["bbox"]

            if cls == "plate":

                plate_crop = frame[y1:y2, x1:x2]

                if plate_crop.size == 0:
                    continue

                plate_type, conf = classifier.predict(plate_crop)

                color = GREEN if plate_type == "hsrp" else RED

                label = f"{plate_type} {conf:.2f}"

                cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)

                cv2.putText(
                    frame,
                    label,
                    (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

        cv2.imshow("HSRP Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_video()