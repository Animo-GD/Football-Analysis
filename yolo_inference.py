from ultralytics import YOLO

model = YOLO("models/best.pt")
results = model.predict("Input_videos/input_video.mp4",save=True,imgsz=640,line_width=2)
print(results[0])

print("===============================")

for box in results[0].boxes:
    print(box)
