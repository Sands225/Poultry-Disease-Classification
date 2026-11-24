from contextlib import asynccontextmanager
import io
import os
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "YOLOv11n_Class Weight.pt")

ml_models = {}  


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"File bobot model tidak ditemukan di: {MODEL_PATH}")
    ml_models["yolo"] = YOLO(MODEL_PATH)
    yield
    ml_models.clear()


app = FastAPI(
    title="Poultry Disease Detection API",
    description="API Deteksi Penyakit Unggas Berbasis Citra Feses Menggunakan YOLOv11",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def preprocess_image(image_bytes: bytes) -> Image.Image:
    """Mengonversi bytes menjadi citra PIL RGB 3-channel."""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return image
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Gagal memproses file gambar: {exc}"
        )


@app.get("/")
def health_check():
    return {
        "status": "online",
        "model": "poultry_disease_detection_YOLOv11.pt",
        "classes": ml_models["yolo"].names,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...), conf: float = 0.4):
    # 1. Terima & Validasi Input
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Input tidak valid. Harap unggah file gambar (JPG/PNG).",
        )

    # 2. Preprocessing Input
    image_bytes = await file.read()
    processed_image = preprocess_image(image_bytes)

    # 3. Inference Model
    model = ml_models["yolo"]
    results = model.predict(
        source=processed_image,
        conf=conf,
        imgsz=640,
        verbose=False,
    )

    # 4. Formatting Output Prediksi
    detections = []
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        detections.append({
            "class_id": class_id,
            "class_name": model.names[class_id],
            "confidence": round(float(box.conf[0]), 4),
            "bbox_xyxy": [round(float(coord), 2) for coord in box.xyxy[0]],
        })

    return {
        "status": "success",
        "filename": file.filename,
        "image_size": {
            "width": processed_image.width,
            "height": processed_image.height,
        },
        "confidence_threshold": conf,
        "total_detections": len(detections),
        "detections": detections,
    }