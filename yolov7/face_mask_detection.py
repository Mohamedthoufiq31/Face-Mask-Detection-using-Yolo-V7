import cv2
import numpy as np
import torch
from pathlib import Path

# Load YOLOv7 model
model_path = 'E:\yolov7\face_mask_detection.py'  # Make sure to provide the correct path
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.hub.load('ultralytics/yolov7', 'custom', path=model_path).to(device)

# Function to run face mask detection
def detect_face_masks(image_path):
    image = cv2.imread(image_path)
    results = model(image)

    # Process results
    for result in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = result
        label = model.names[int(cls)]
        if label == 'mask':
            color = (0, 255, 0)
            text = "Mask"
        else:
            color = (0, 0, 255)
            text = "No Mask"
        
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(image, text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow('Face Mask Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Provide the path to your image here
image_path = 'E:\yolov7\face_mask_detection.py'
detect_face_masks(image_path)
