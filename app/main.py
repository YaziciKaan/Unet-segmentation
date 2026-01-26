import cv2
import numpy as np

from app.model_loader import ModelWrapper
from contextlib import asynccontextmanager
from fastapi.responses import Response, StreamingResponse
from fastapi import FastAPI, File, UploadFile, HTTPException


models = {}

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
    mask_resized = cv2.resize(mask, (h,w), interpolation=cv2.INTER_NEAREST)

    colored_mask = np.zeros_like(frame)
    colored_mask[:, :, 2] = mask_resized
    alpha = 0.5
    result = cv2.addWeighted(frame, 1-alpha, colored_mask, 0)

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