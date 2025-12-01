# Image Super-Resolution with EDSR (4× Upscaling)

This repository contains an end-to-end **Image Super-Resolution** system built using the **Enhanced Deep Super-Resolution (EDSR)** model.
The project covers everything from **model training** to **optimization** and **deployment**, demonstrating a full deep-learning MLOps workflow.

The model was trained on the **DIV2K** dataset and optimized to generate high-quality upto 4× optimized upscaled images with significantly sharper textures and improved visual fidelity.

---

## 🚀 Features

### 🔹 **Deep Learning Model**

* Implements **EDSR** architecture for Single Image Super-Resolution (SISR)
* Trained for **Upto 4× upscaling**
* Lightweight modifications for faster inference
* PSNR/SSIM-based evaluation pipeline

### 🔹 **Training & Evaluation**

* Trained on **DIV2K** benchmark dataset (high-resolution natural images)
* Includes preprocessing pipeline:

  * Image cropping
  * Bicubic downscaling
  * Normalization
* Evaluation on:

  * **DIV2K validation set**
  * **Set5 benchmark**
* Achieved:

  * **~30 dB PSNR**
  * **~0.85 SSIM**

### 🔹 **ONNX Conversion & Optimization**

* Converts TensorFlow EDSR model → ONNX
* Accelerated inference using **ONNX Runtime**
* ~2× faster inference compared to native TensorFlow

### 🔹 **Production-Ready Deployment**

* Built a REST API using **FastAPI**
* Dockerized application for scalable deployment
* Exposes `/predict` endpoint for image upscaling
* Can be deployed on:

  * AWS EC2
  * Azure Container Instances
  * Docker Compose
  * Kubernetes

---

## 🧱 Architecture Overview

```
+----------------------+       +-----------------+       +------------------+
|  Low-Resolution Img  | --->  |   EDSR Model    | --->  |  High-Res Output |
+----------------------+       +-----------------+       +------------------+

Training: DIV2K → Preprocessing → EDSR → Metrics (PSNR/SSIM)

Deployment: FastAPI → ONNX Runtime → Docker → REST Inference
```

---

## 📁 Project Structure

```
│── data/
│── notebooks/
│── src/
│   ├── model.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   ├── convert_to_onnx.py
│   └── api/
│       └── main.py   (FastAPI app)
│── docker/
│── requirements.txt
│── README.md
```



## ▶️ Running the Training Script

Install requirements:

```bash
pip install -r requirements.txt
```

Train model:

```bash
python src/train.py
```

Evaluate:

```bash
python src/evaluate.py
```

Convert to ONNX:

```bash
python src/convert_to_onnx.py
```

Serve via FastAPI:

```bash
uvicorn src.api.main:app --reload
```

Docker:

```bash
docker build -t edsr-superres .
docker run -p 8000:8000 edsr-superres
```

---

## 📌 Things to do in future (maybe)

* Add **ESRGAN** variant for perceptual quality and comparitive analysis
* Add TFLite model for edge devices such as mobile phones
* Support 8× scaling factors
---

## 🏁 Summary

This project demonstrates a complete **research-to-deployment** pipeline using modern deep-learning tools.
It highlights practical engineering skills including:

* Model implementation
* Data pipeline design
* Training & evaluation
* Model optimization
* API development
* Docker deployment
