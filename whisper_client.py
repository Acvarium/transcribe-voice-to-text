import argparse
import os
import requests
import subprocess
import time

SERVER_URL = "http://127.0.0.1:8000"

def is_server_running():
    try:
        r = requests.get(SERVER_URL + "/docs", timeout=1)
        return r.status_code == 200
    except:
        return False

def start_server():
    subprocess.Popen(["python3", "whisper_server.py"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("[INFO] Starting server...")
    for _ in range(10):
        if is_server_running():
            print("[INFO] Server is running.")
            return True
        time.sleep(1)
    print("[ERROR] Failed to start server.")
    return False


def send_file(file_path, language, timestamps, confidence):
    with open(file_path, "rb") as f:
        response = requests.post(
            SERVER_URL + "/transcribe/",
            files={"file": (os.path.basename(file_path), f, "audio/mpeg")},
            data={"language": language, "timestamps": str(timestamps).lower(), "confidence": str(confidence).lower()}
        )
    return response.json()

def stop_server():
    try:
        requests.post(SERVER_URL + "/shutdown")
        print("[INFO] Server shutdown requested.")
    except:
        print("[ERROR] Could not connect to server.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file", help="Path to input media file")
    parser.add_argument("-o", "--output_file", help="Path to output file")
    parser.add_argument("-l", "--language", default="uk")
    parser.add_argument("-t", "--timestamp", action="store_true")
    parser.add_argument("-c", "--confidence", action="store_true")
    parser.add_argument("--stop", action="store_true", help="Stop the running server")
    args = parser.parse_args()

    if args.stop:
        stop_server()
        return

    if not is_server_running():
        if not start_server():
            return

    result = send_file(args.input_file, args.language, args.timestamp, args.confidence)

    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            f.write(result["text"])
        print(f"[SUCCESS] Output saved to {args.output_file}")
    else:
        print(result["text"])

if __name__ == "__main__":
    main()
