# !wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-cls.pt
# !pip install ultralytics

# Load YOLOv11 model
model = YOLO('yolo11n-cls.pt')
yolo_model = model.model

yolo_model.train()

# Freeze all layers except the last one
for param in yolo_model.parameters():
    param.requires_grad = False

yolo_model.model[10].linear = nn.Linear(1280, 10)

yolo_model = yolo_model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(yolo_model.model[10].linear.parameters(), lr=0.002, betas=(0.9, 0.999))