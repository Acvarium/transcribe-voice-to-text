import argparse
import os
import json
import requests
import subprocess
import time

SERVER_URL = "http://127.0.0.1:8000"

verbose = True


def load_config():
    """Load and return configuration from config.json"""
    with open("config.json", "r", encoding="utf-8") as f:
        return json.load(f)


def format_timestamp(seconds):
    """Convert seconds to formatted timestamp"""
    return time.strftime('%H:%M:%S', time.gmtime(seconds))


def format_txt_output(result, include_timestamps, include_confidence):
    """Format transcription result as text"""
    if not include_timestamps or "segments" not in result:
        return result["text"]

    output = []
    for segment in result["segments"]:
        timestamp = f"[{format_timestamp(segment['start'])} - {format_timestamp(segment['end'])}]"
        
        if include_confidence:
            confidence = f"(confidence: {segment.get('confidence', 0):.2f})"
            output.append(f"{timestamp} {confidence}\n{segment['text']}\n")
        else:
            output.append(f"{timestamp}\n{segment['text']}\n")
    
    return "\n".join(output)


def format_srt_output(result):
    output = []
    for i, segment in enumerate(result["segments"], 1):
        start_time = time.strftime('%H:%M:%S', time.gmtime(segment['start']))
        start_milliseconds = "{:03d}".format(int((segment['start'] - int(segment['start'])) * 1000))
        formatted_start_time = f"{start_time},{start_milliseconds}"

        end_time = time.strftime('%H:%M:%S', time.gmtime(segment['end']))
        end_milliseconds = "{:03d}".format(int((segment['end'] - int(segment['end'])) * 1000))
        formatted_end_time = f"{end_time},{end_milliseconds}"

        output.extend([
            str(i),
            f"{formatted_start_time} --> {formatted_end_time}",
            segment['text'],
            ""
        ])
    return "\n".join(output)


def print_message(message):
    if verbose:
        print(message)


def is_server_running():
    try:
        r = requests.get(SERVER_URL + "/docs", timeout=1)
        return r.status_code == 200
    except:
        return False


def start_server():
    subprocess.Popen(
        ["python3", "whisper_server.py"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setpgrp  # ВАЖЛИВО: запускає у власній групі процесів
    )
    print_message("[INFO] Starting server...")
    for _ in range(10):
        if is_server_running():
            print_message("[INFO] Server is running.")
            return True
        time.sleep(1)
    print_message("[ERROR] Failed to start server.")
    return False


def send_file(file_path, language):
    with open(file_path, "rb") as f:
        response = requests.post(
            SERVER_URL + "/transcribe/",
            files={"file": (os.path.basename(file_path), f, "audio/mpeg")},
            data={"language": language}
        )
    return response.json()

def stop_server():
    try:
        requests.post(SERVER_URL + "/shutdown")
        print_message("[INFO] Server shutdown requested.")
    except:
        print_message("[ERROR] Could not connect to server.")


def main():
    global verbose 
    parser = argparse.ArgumentParser(description="Transcribe audio/video files using Whisper model")
    parser.add_argument('-i', '--input_file', help="Path to input media file")
    parser.add_argument('-o', '--output_file', help="Optional path for output text file")
    parser.add_argument('-l', '--language', help="Optional language: en, uk")
    parser.add_argument('-t', '--timestamp', help="Include timestamp into output file if true, and exclude if false")
    parser.add_argument('-c', '--confidence', help="Include confidence into output file if true, and exclude if false")
    parser.add_argument('-p', '--print', action='store_true', help="Print resulting text")
    parser.add_argument('-s', "--stop", action="store_true", help="Stop the running server")

    args = parser.parse_args()

    if args.stop:
        stop_server()
        return

    input_path = args.input_file
    output_path = args.output_file
    if input_path == None:
        input_path = input("Enter the path to the media file: ")


    if args.print:
        verbose = False


    if not is_server_running():
        if not start_server():
            return
    
    if not os.path.exists(input_path):
        print_message(f"[ERROR] Media file not found: {input_path}")
        return
    print_message(f"[INFO] Input file: {input_path}")

    config = load_config()
    language = config.get("language", "uk")
    if args.language != None:
        language = args.language

    result = send_file(input_path, language)

    output_format = config.get("output_format", {
        "type": "txt",
        "include_timestamps": False,
        "include_confidence": False
    })
    if args.timestamp != None :
        if args.timestamp == "true":
            output_format["include_timestamps"] = True 
        elif args.timestamp == "false":
            output_format["include_timestamps"] = False
    if args.confidence != "None":
        if args.confidence == "true":
            output_format["include_timestamps"] = True
        elif args.confidence == "false":
            output_format["include_timestamps"] = False


    output_ext = output_format["type"].lower()

    if output_path != None:
        extension = os.path.splitext(output_path)[1]
        if extension != "":
            output_ext = extension
    if output_ext not in ["txt", "json", "srt"]:
        output_ext = "txt"

    if output_path is None:
        output_path = os.path.splitext(input_path)[0] + f".{output_ext}"
    print_message(f"[INFO] Saving output to: {output_path}")

    if output_ext == "txt":
        output_content = format_txt_output(
            result,
            output_format.get("include_timestamps", False),
            output_format.get("include_confidence", False)
        )
    elif output_ext == "json":
        if not output_format.get("include_timestamps", False):
            result = {"text": result["text"]}
        output_content = json.dumps(result, ensure_ascii=False, indent=2)
    else:  # srt
        output_content = format_srt_output(result)

    if args.print:
        print(output_content)
    else:
        try:
            with open(output_path, "w", encoding="utf-8") as file:
                file.write(output_content)
            print_message(f"[SUCCESS] Transcription saved to {output_path}")
        except Exception as e:
            print_message(f"[ERROR] Failed to save file: {e}")


if __name__ == "__main__":
    main()
