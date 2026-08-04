from ultralytics import YOLO

model = YOLO("models/last.pt")

results = model.predict('input-videos/08fd33_4.mp4', save = True)

print(results[0])

print("--------------------------------------------------------------")

for box in results[0].boxes:
    print(box)