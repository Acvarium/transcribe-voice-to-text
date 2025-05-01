from fastapi import FastAPI, UploadFile, Form
import uvicorn
import whisper
import os
import threading
import time
import json
import ssl

CONFIG_FILE = "config.json"


app = FastAPI()
model = None
last_used = time.time()
INACTIVITY_TIMEOUT = 10  # 5 хвилин

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

config = load_config(CONFIG_FILE)


is_expandable_segments = config.get("expandable_segments", False)
is_unverified_ssl_context = config.get("unverified_ssl_context", False)

if is_unverified_ssl_context:
    ssl._create_default_https_context = ssl._create_unverified_context
    print("[INFO] Disabled SSL certificate verification")

if is_expandable_segments:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    print("[INFO] Enabled PyTorch expandable segments")


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
async def transcribe(file: UploadFile, language: str = Form("uk")):
    global last_used
    last_used = time.time()
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())
    result = model.transcribe(temp_path, language=language)
    return result


@app.post("/shutdown")
async def shutdown():
    print("Shutdown requested")
    os._exit(0)


if __name__ == "__main__":
    threading.Thread(target=monitor_inactivity, daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=8000)
