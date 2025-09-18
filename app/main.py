import os
import io
import numpy as np
import tensorflow as tf
import uvicorn
import onnxruntime as ort
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Super Resolution Image")

model_path = Path(__file__).parent / "model.onnx"

try:
    start_time = time.perf_counter()
    model = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider"])
    logger.info(f"Model successfully loaded from {model_path} in {time.perf_counter()-start_time:.2f} sec")

except Exception as e:
    logger.error(f"Failed to load model from {model_path}: {e}")

def preprocess_image(image: UploadFile):

    try:
        img = Image.open(image.file).convert("RGB")
        img_arr = np.array(img, dtype=np.float32) / 255.0
        img_arr = np.expand_dims(img_arr, axis=0)

        return img_arr

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def super_resolution_image(image):

    try:
        model_inputs = model.get_inputs()[0].name
        model_outputs = model.get_outputs()[0].name
        sr_image = model.run([model_outputs], {model_inputs: image})[0]
        sr_image = (np.clip(sr_image, 0, 1) * 255).astype(np.uint8)
        return sr_image[0]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed {str(e)}")

@app.post("/predict", summary="Super Resolution Image")
async def predict_image(image: UploadFile = File(...)):

    if not image.content_type.startswith("image/"):
            raise HTTPException(status_code=500, detail="Unsupported file type")

    try:
        logger.info(f"Received Image {image.filename}")
        start_time = time.perf_counter()

        img_array = preprocess_image(image)
        sr_img = super_resolution_image(img_array)

        end_time = time.perf_counter()
        infer_time = end_time-start_time
        logger.info(f"4x Super Resolution completed for {image.filename} in {infer_time:.2f} sec")

        # sr_image = (np.clip(sr_img, 0, 1) * 255).astype(np.uint8)
        sr_pil = Image.fromarray(sr_img)
        sr_out = io.BytesIO()
        sr_pil.save(sr_out, format="PNG")
        sr_out.seek(0)

        return StreamingResponse(sr_out, media_type="image/png")
    
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return JSONResponse(content={"status": "healthy"})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)