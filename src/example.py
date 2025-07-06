from ultralytics import YOLO

# Load a model
model = YOLO("yolo5n.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="../dataset.yaml" epochs=100, imgsz=640)
