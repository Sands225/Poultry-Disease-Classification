from contextlib import asynccontextmanager
import os
from typing import BinaryIO
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "YOLOv11n_Class Weight.pt"
MODEL_PATH = os.path.join(BASE_DIR, MODEL_NAME)

VALIDATION_MESSAGES = {
    "file": "Input tidak valid. Field 'file' wajib diisi dengan file gambar (JPG/PNG).",
    "conf": "Parameter 'conf' tidak valid. Harap masukkan nilai angka antara 0.0 dan 1.0 (contoh: 0.4).",
}

ml_models: dict[str, YOLO] = {}


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
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Menyeragamkan respons error validasi request FastAPI ke format Bahasa Indonesia."""
    for err in exc.errors():
        for field, message in VALIDATION_MESSAGES.items():
            if field in err.get("loc", ()):
                return JSONResponse(status_code=422, content={"detail": message})

    return JSONResponse(
        status_code=422,
        content={"detail": "Permintaan tidak valid. Periksa kembali parameter dan body request."},
    )


def preprocess_image(file_obj: BinaryIO) -> Image.Image:
    """Membaca stream gambar dan mengonversinya menjadi citra PIL RGB 3-channel."""
    try:
        with Image.open(file_obj) as image:
            return image.convert("RGB")
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Gagal memproses file gambar: {exc}"
        ) from exc


@app.get("/")
def health_check():
    """Mengecek status server dan informasi model yang sedang aktif."""
    return {
        "status": "online",
        "model": MODEL_NAME,
        "classes": ml_models["yolo"].names,
    }


@app.post("/predict")
def predict(
    file: UploadFile = File(...),
    conf: float = Query(0.4, ge=0.0, le=1.0),
):
    """Menjalankan validasi, preprocessing, inferensi YOLO, dan pemformatan hasil JSON."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Input tidak valid. Harap unggah file gambar (JPG/PNG).",
        )

    processed_image = preprocess_image(file.file)

    model = ml_models["yolo"]
    results = model.predict(
        source=processed_image,
        conf=conf,
        imgsz=640,
        verbose=False,
    )

    boxes = results[0].boxes
    detections = [
        {
            "class_id": int(cls_id),
            "class_name": model.names[int(cls_id)],
            "confidence": round(score, 4),
            "bbox_xyxy": [round(coord, 2) for coord in coords],
        }
        for cls_id, score, coords in zip(
            boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist()
        )
    ]

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
