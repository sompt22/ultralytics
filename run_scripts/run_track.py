from collections import defaultdict
import cv2
import numpy as np
from pathlib import Path

from ultralytics import YOLO
from boxmot import StrongSort  # 💡 Boxmot'un StrongSort'u


def get_color(idx):
    idx = abs(int(idx))
    hue = (idx * 30) % 180
    color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
    return int(color[0]), int(color[1]), int(color[2])

# 1️⃣ YOLOv10 modelini yükle
model = YOLO("/home/fatih/phd/ultralytics/ultralytics/weights/yolov10x.pt")

# 2️⃣ StrongSORT tracker'ı başlat
tracker = StrongSort(
    reid_weights=Path("/home/fatih/phd/boxmot/tracking/weights/osnet_x1_0_imagenet.pth"),  # 💡 kendi reid model yolunu göster
    device='0',
    half=False,            # opsiyonel: fp16 kullanmak istersen True
    per_class=False,      # opsiyonel: class bazlı takip yapmak istemiyorsan False
    min_conf=0.25,        # opsiyonel: min confidence threshold
    max_cos_dist=0.2,     # opsiyonel: cosine distance threshold
    max_iou_dist=0.7,     # opsiyonel: IoU distance threshold
    max_age=30,           # opsiyonel: track kaybolmadan önceki max frame sayısı
    n_init=3,             # opsiyonel: track başlatılmadan önceki min frame
    nn_budget=100,        # opsiyonel: feature library boyutu
)

# 3️⃣ Video dosyasını aç
video_path = "/home/fatih/phd/mot-simulation-suite/video/sompt22/sompt22_train_4_25_10_circular_constant_0_none_0_occlusion/sompt22_train_4_25_10_circular_constant_0_none_0_occlusion.mp4"
cap = cv2.VideoCapture(video_path)

# 4️⃣ Track history (opsiyonel çizgiler için)
track_history = defaultdict(lambda: [])

# 5️⃣ Video döngüsü
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # YOLOv10 inference
    result = model(frame)[0]  # result: yolov10 frame prediction

    # Boxes, confs ve class_id'leri al
    if result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        clss = result.boxes.cls.cpu().numpy()
        
        # Yalnızca class_id == 0 olan insan sınıfını seç
        mask = clss == 0
        boxes = boxes[mask]
        confs = confs[mask]
        clss = clss[mask]
    else:
        boxes = np.empty((0, 4))
        confs = np.array([])
        clss = np.array([])
    
    if boxes.shape[0] > 0:
        dets = np.concatenate([boxes, confs[:, None], clss[:, None]], axis=1)  # 🚀 (N, 6)
    else:
        dets = np.empty((0, 6))

    # StrongSORT tracker'ı update et
    tracks = tracker.update(dets, frame)

    # Çizimler: yolov10 plot + track id'ler
    #frame = result.plot(labels=False)
    
    # Her track için id ve çizgi ekle
    for track in tracks:
        x1, y1, x2, y2, track_id = track[:5]
        color = get_color(track_id)

        cv2.putText(frame, f"ID: {int(track_id)}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        track_history[track_id].append((cx, cy))
        if len(track_history[track_id]) > 30:
            track_history[track_id].pop(0)

        pts = np.hstack(track_history[track_id]).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=False, color=color, thickness=2)

    # Görüntüyü göster
    cv2.imshow("YOLOv10 + StrongSORT Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
