from fastapi import FastAPI, UploadFile, Form
import uvicorn
import whisper
import os
import threading
import time

app = FastAPI()
model = None
last_used = time.time()
INACTIVITY_TIMEOUT = 600  # 10 хвилин

@app.on_event("startup")
def load_model():
    global model
    model = whisper.load_model("medium")

def monitor_inactivity():
    while True:
        if time.time() - last_used > INACTIVITY_TIMEOUT:
            print("Inactivity timeout reached. Shutting down...")
            os._exit(0)
        time.sleep(30)

@app.post("/transcribe/")
async def transcribe(file: UploadFile, language: str = Form("uk"), timestamps: bool = Form(False), confidence: bool = Form(False)):
    global last_used
    last_used = time.time()
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())
    result = model.transcribe(temp_path, language=language)

    if not timestamps:
        return {"text": result["text"]}
    else:
        return {"segments": result.get("segments", []), "text": result["text"]}

@app.post("/shutdown")
async def shutdown():
    print("Shutdown requested")
    os._exit(0)

if __name__ == "__main__":
    threading.Thread(target=monitor_inactivity, daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=8000)
