import os
import cv2
import numpy as np

from pydantic import BaseModel
from app.model_loader import ModelWrapper
from contextlib import asynccontextmanager
from fastapi.responses import Response, StreamingResponse
from fastapi import FastAPI, File, UploadFile, HTTPException


models = {}

AVAILABLE_MODELS = {
    "onnx": "models/best_model.onnx",
    "pytorch": "models/best_model.pth"
}

class ModelSwitchRequest(BaseModel):
    model_name: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Model Yükleniyor.")
    models["pothole"] = ModelWrapper(model_path="models/best_model.onnx")
    print("Model Yüklendi.")
    yield

    models.clear()

app = FastAPI(lifespan=lifespan)

@app.get("/")
def read_root():
    return {"mesaj": "Pothole Detection API Aktif!"}


@app.get("/model-info")
def get_model_info():
    if "pothole" not in models:
        raise HTTPException(status_code=500, detail="Model yüklenmedi.")
    
    model = models["pothole"]
    return {
        "model_type": model.model_type,
        "model_path": model.model_path,
        "device": model.device,
        "avg_fps": model.get_avg_fps() if hasattr(model, 'get_avg_fps') else 0,
        "available_models": AVAILABLE_MODELS
    }


@app.post("/switch-model")
def switch_model(request: ModelSwitchRequest):
    model_type = request.model_name
    if model_type.lower() not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400, 
            detail=f"Geçersiz model tipi. Erişilebilir modeller: {list(AVAILABLE_MODELS.keys())}"
        )
    
    model_path = AVAILABLE_MODELS[model_type.lower()]
    
    if not os.path.exists(model_path):
        raise HTTPException(
            status_code=404,
            detail=f"Model bulunamadı {model_path}"
        )
    
    try:
        print(f"Model değiştiriliyor: {model_type} -> {model_path}")
        new_model = ModelWrapper(model_path=model_path)
        
        # Replace old model
        models["pothole"] = new_model
        
        print(f"Model başarıyla değiştirildi: {model_type.upper()}")
        
        return {
            "success": True,
            "message": f"Model switched to {model_type.upper()}",
            "model_type": new_model.model_type,
            "model_path": new_model.model_path,
            "device": new_model.device
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")


@app.post("/predict")
def predict_media_type(file: UploadFile=File(...)):
    if "pothole" not in models:
        raise HTTPException(status_code=500, detail="Model yüklenmedi.")
    
    content_type = file.content_type
    filename = file.filename.lower()

    if "image" in content_type and filename.endswith((".jpeg", ".png", ".jpg")):
        return process_image(file)
    elif "video" in content_type and filename.endswith((".mp4", ".avi", ".mov")):
        return process_video(file)
    else:
        print(f"UYARI: Tanımsız dosya türü: {content_type} / {filename}")
        return HTTPException(status_code=400, detail="Veri desteklenen formatta değil.")


def process_image(file: UploadFile):
    contents = file.file.read()

    np_array = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Resim okunamadı.")

    model = models["pothole"]
    mask, fps = model.predict(frame)

    h, w = frame.shape[:2]
    mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    colored_mask = np.zeros_like(frame)
    colored_mask[:, :, 2] = mask_resized
    alpha = 0.5
    result = cv2.addWeighted(frame, 1-alpha, colored_mask, alpha, 0)

    _, encoded_img = cv2.imencode('.jpg', result)

    return Response(content=encoded_img.tobytes(), media_type="image/jpeg")


def process_video(file: UploadFile):
    import tempfile

    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(file.file.read())
    tfile.close()

    def iterfile():
        cap = cv2.VideoCapture(tfile.name)
        model = models["pothole"]

        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                break

            mask, fps = model.predict(frame)

            h, w = frame.shape[:2]
            mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

            colored_mask = np.zeros_like(frame)
            colored_mask[:, :, 2] = mask_resized
            alpha = 0.5
            result = cv2.addWeighted(frame, 1-alpha, colored_mask, alpha, 0)

            _, buffer = cv2.imencode('.jpg', result)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        cap.release()

    return StreamingResponse(iterfile(), media_type="multipart/x-mixed-replace; boundary=frame")