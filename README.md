# UNET Çukur Tespit Sistemi

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

GPU destekli UNET mimarisi kullanarak gerçek zamanlı çukur tespiti ve segmentasyon sistemi.

<div align="center">
  <img src="https://img.shields.io/badge/Durum-Production%20Ready-brightgreen.svg" alt="Status">
  <img src="https://img.shields.io/badge/İnference-100+%20FPS-orange.svg" alt="FPS">
  <img src="https://img.shields.io/badge/Model%20Boyutu-700KB-blue.svg" alt="Size">
</div>

---

## 📋 İçindekiler

- [Özellikler](#-özellikler)
- [Demo](#-demo)
- [Mimari](#-mimari)
- [Kurulum](#-kurulum)
- [Kullanım](#-kullanım)
- [API Dokümantasyonu](#-api-dokümantasyonu)
- [Eğitim](#-eğitim)
- [Performans](#-performans)
- [Deployment](#-deployment)
- [Proje Yapısı](#-proje-yapısı)
- [Katkıda Bulunma](#-katkıda-bulunma)
- [Lisans](#-lisans)

---

## Özellikler

- **Gerçek Zamanlı Tespit**: GPU'da 100+ FPS
- **Yüksek Doğruluk**: 0.77+ Dice Score
- **Hafif Model**: Sadece 700KB (ONNX optimize)
- **Docker Desteği**: GPU desteği ile tam containerization
- **REST API**: FastAPI tabanlı endpoint'ler
- **Çoklu Format Desteği**: Görsel ve video
- **İnteraktif Arayüz**: Streamlit web arayüzü
- **Karma Hassasiyet Eğitimi**: AMP ile daha hızlı eğitim

---

## Demo

### Görsel Tespiti
```bash
# API kullanarak
curl -X POST "http://localhost:8000/predict" \
  -F "file=@pothole.jpg" \
  --output result.jpg
```

### Video İşleme
```bash
# Gerçek zamanlı video streaming
curl -X POST "http://localhost:8000/predict" \
  -F "file=@video.mp4"
```

### Web Arayüzü
```bash
# Streamlit uygulamasını başlat
streamlit run streamlit_app.py
```

---

## Mimari

### UNET Modeli

```
Input (3, 288, 512)
    │
    ├─ Encoder Yolu (Downsampling)
    │   ├─ Conv Bloğu 1 → 64 kanal
    │   ├─ Conv Bloğu 2 → 128 kanal
    │   ├─ Conv Bloğu 3 → 256 kanal
    │   └─ Conv Bloğu 4 → 512 kanal
    │
    ├─ Bottleneck → 1024 kanal
    │
    └─ Decoder Yolu (Upsampling + Skip Bağlantılar)
        ├─ UpConv Bloğu 1 → 512 kanal
        ├─ UpConv Bloğu 2 → 256 kanal
        ├─ UpConv Bloğu 3 → 128 kanal
        └─ UpConv Bloğu 4 → 64 kanal
            │
Output (1, 288, 512) - Binary Mask
```

**Ana Bileşenler:**
- **TwoConv Blokları**: 2x (Conv2d → BatchNorm → ReLU)
- **Skip Bağlantıları**: Encoder'dan decoder'a özellik korunumu
- **MaxPooling**: 2x2 downsampling
- **Transposed Convolution**: Öğrenilebilir upsampling

---

## Kurulum

### Gereksinimler
- Python 3.10+
- CUDA 12.1+ (GPU desteği için)
- Docker (opsiyonel)

### Seçenek 1: Yerel Kurulum

```bash
# Repository'yi klonla
git clone https://github.com/YaziciKaan/Unet-segmentation.git
cd Unet-segmentation

# Sanal ortam oluştur
python -m venv venv
source venv/bin/activate  # Linux/Mac
# veya
venv\Scripts\activate  # Windows

# Bağımlılıkları yükle
pip install -r requirements.txt

# Eğitilmiş modeli indir (sıfırdan eğitmiyorsan)
# Modelini şuraya yerleştir: models/best_model.onnx
```

### Seçenek 2: Docker Kurulumu

```bash
# Image oluştur
docker compose build

# Container'ı çalıştır
docker compose up
```

---

## Kullanım

### 1. Sistemi Başlat

- Eğer Docker kurulumu yapıldıysa:
    ```bash
    docker compose up
    ```
- Eğer yerel kurulum yapıldıysa:
    ```bash
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
    ```


API şu adreste erişilebilir olacak:
- **Base URL**: http://localhost:8000
- **Swagger Docs**: http://localhost:8000/docs

### 2. Streamlit Arayüzünü Başlat

```bash
streamlit run streamlit_app.py
```

Web arayüzüne şuradan erişin: http://localhost:8501

---

## API Dokümantasyonu

### Endpoint'ler

#### 1. Sağlık Kontrolü
```http
GET /
```
**Yanıt:**
```json
{
  "mesaj": "Pothole Detection API Aktif!"
}
```

#### 2. Tahmin (Otomatik medya türü tespiti)
```http
POST /predict
Content-Type: multipart/form-data
```
**Parametreler:**
- `file`: Görsel veya video dosyası

**Yanıt (Görsel):**
- Tespit edilen çukurlar üzerine kırmızı overlay ile binary görsel

**Yanıt (Video):**
- MJPEG frame streaming

---

## Eğitim

### Veri Setini Hazırla

Beklenen yapı:
```
datasets/
└── Pothole_Segmentation/
    ├── train/
    │   ├── images/
    │   │   ├── img1.jpg
    │   │   └── ...
    │   └── masks/
    │       ├── img1.png
    │       └── ...
    └── valid/
        ├── images/
        └── masks/
```

### Modeli Eğit

```bash
cd training

# Eğitimi başlat
python train.py
```

**Eğitim Özellikleri:**
- Karma hassasiyet eğitimi (AMP)
- Otomatik checkpoint kaydetme (best.pth, latest.pth)
- Her 10 epoch'ta periyodik checkpoint
- Dice score ve piksel doğruluğu metrikleri
- Albumentations veri augmentasyonu

### ONNX'e Dönüştür

```bash
cd training
python transform_to_onnx.py
```

Bu, production inference için optimize edilmiş `best_model.onnx` dosyasını oluşturur.

---

## Performans

### Model Metrikleri

| Metrik | Değer |
|--------|-------|
| Dice Score | 0.7739 |
| Piksel Doğruluğu | %98+ |
| Model Boyutu (PyTorch) | ~300 MB |
| Model Boyutu (ONNX) | 125 MB |
---

## Project Structure

```
Unet-segmentation/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── model.py             # UNET architecture
│   ├── model_loader.py      # Model wrapper (PyTorch/ONNX)
│   └── utils.py             # Training utilities
│
├── training/
│   ├── dataset.py           # PyTorch Dataset class
│   ├── train.py             # Training script
│   ├── test.py              # Inference script
│   ├── transform_to_onnx.py # ONNX conversion
│   └── convert_yolo_to_masks.py  # YOLO format converter
│
├── models/
│   └── best_model.onnx      # Production model
│
├── checkpoints/             # Training checkpoints (gitignored)
│   ├── best.pth
│   ├── latest.pth
│   └── checkpoint_epoch_*.pth
│
├── streamlit_app.py         # Web interface
├── docker-compose.yml       # Docker orchestration
├── Dockerfile               # Container definition
├── requirements.txt         # Python dependencies
├── .gitignore
├── .dockerignore
├── API_README.md
└── README.md
```
---