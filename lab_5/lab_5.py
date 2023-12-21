from ultralytics import YOLO
import dill

model = YOLO("best.pt")  # загрузите предварительно обученную модель YOLOv8n

model.predict(source="datasets/animals/test/*.jpg")