import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    if frame is not None:
        result = model(frame)
        result_rgb = cv2.cvtColor(np.squeeze(result.render()), cv2.COLOR_BGR2RGB)
        
        # Use Matplotlib instead of OpenCV's imshow
        plt.imshow(result_rgb)
        plt.axis("off")
        plt.draw()
        plt.pause(0.001)  # Short pause to update the figure
        plt.clf()  # Clear the figure for the next frame

    else:
        print("Unable to read frame from video capture")
        break

cap.release()
plt.close()  # Close the Matplotlib window when done
