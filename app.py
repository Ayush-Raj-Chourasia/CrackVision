from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from crackvision_service import SERVICE, load_default_conf


app = FastAPI(
    title="CrackVision API",
    version="1.0.0",
    description="Road damage detection API for CrackVision.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict:
    return {
        "name": "CrackVision API",
        "status": "ready",
        "default_confidence": load_default_conf(),
        "model_path": str(SERVICE.model_path),
    }


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": SERVICE._model is not None,
        "sahi_enabled": SERVICE.enable_sahi,
    }


@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    mode: str = Form("accurate"),
    confidence: float = Form(load_default_conf()),
) -> JSONResponse:
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload a valid image file.")

    suffix = Path(image.filename or "upload.jpg").suffix or ".jpg"
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_path = Path(temp_file.name)
    try:
        temp_file.write(await image.read())
        temp_file.flush()
        temp_file.close()
        payload = SERVICE.predict_file(temp_path, mode=mode, confidence=confidence)
        return JSONResponse(payload)
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
