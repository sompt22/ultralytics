from ultralytics import YOLO

# Küçük ve hızlı bir model yüklüyoruz
model = YOLO("yolov8n.pt")

# Demo bir görüntü yolu (örneğin, test.jpg)
results = model("https://ultralytics.com/images/bus.jpg")

# Sonuçları göster
results[0].show()
