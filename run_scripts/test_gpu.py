import torch

if torch.cuda.is_available():
    print("✅ CUDA destekli GPU kullanılıyor!")
    print("GPU adı:", torch.cuda.get_device_name(0))
else:
    print("⚠️ CUDA destekli GPU bulunamadı.")
