import itertools
import json
import os
import threading
import time
import ssl
import argparse
import time
import whisper

verbose = True

def load_config():
    """Load and return configuration from config.json"""
    with open("config.json", "r", encoding="utf-8") as f:
        return json.load(f)


def spinner(stop_event):
    """Display a spinning progress indicator"""
    for char in itertools.cycle(['|', '/', '-', '\\']):
        if stop_event.is_set():
            break
        if verbose:
            print('\rTranscribing... ' + char, end='', flush=True)
        time.sleep(0.1)
    if verbose:
        print('\rTranscription completed!     ')


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


def main():
    global verbose 
    parser = argparse.ArgumentParser(description="Transcribe audio/video files using Whisper model")
    parser.add_argument('-i', '--input_file', help="Path to input media file")
    parser.add_argument('-o', '--output_file', help="Optional path for output text file")
    parser.add_argument('-m', '--model_name', help="Optional model name: tiny, base, small, medium, large-v3")
    parser.add_argument('-l', '--language', help="Optional language: en, uk")
    parser.add_argument('-t', '--timestamp', help="Include timestamp into output file if true, and exclude if false")
    parser.add_argument('-c', '--confidence', help="Include confidence into output file if true, and exclude if false")
    parser.add_argument('-p', '--print', action='store_true', help="Print resulting text")
    

    args = parser.parse_args()
    input_path = args.input_file
    output_path = args.output_file
    arg_model_name = args.model_name
    arg_language = args.language
    arg_timestamp = args.timestamp
    arg_confidence = args.confidence
    arg_print = args.print
    if arg_print:
        verbose = False
    if input_path == None:
        input_path = input("Enter the path to the media file: ")
        
    if not os.path.exists(input_path):
        print_message(f"[ERROR] Media file not found: {input_path}")
        return
    print_message(f"[INFO] Input file: {input_path}")
    if output_path:
        print_message(f"[INFO] Output file: {output_path}")

    config = load_config()
    
    language = config.get("language", "uk")
    if arg_language != None:
        language = arg_language
    model_name = config.get("model", "medium")
    if arg_model_name != None:
        model_name = arg_model_name
    is_expandable_segments = config.get("expandable_segments", True)
    is_unverified_ssl_context = config.get("unverified_ssl_context", True)
    output_format = config.get("output_format", {
        "type": "txt",
        "include_timestamps": False,
        "include_confidence": False
    })
    if arg_timestamp != None:
        if arg_timestamp == "true":
            output_format["include_timestamps"] = True 
        elif arg_timestamp == "false":
            output_format["include_timestamps"] = False
    if arg_confidence != "None":
        if arg_confidence == "true":
            output_format["include_timestamps"] = True
        elif arg_confidence == "false":
            output_format["include_timestamps"] = False

    if is_unverified_ssl_context:
        ssl._create_default_https_context = ssl._create_unverified_context

    if is_expandable_segments:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    print_message(f"[INFO] Loading model '{model_name}'...")
    model = whisper.load_model(model_name)

    stop_event = threading.Event()
    spinner_thread = threading.Thread(target=spinner, args=(stop_event,))
    spinner_thread.start()

    try:
        result = model.transcribe(input_path, language=language)
    finally:
        stop_event.set()
        spinner_thread.join()

    print_message("[INFO] Transcription finished.")

    if not result:
        print_message("[ERROR] Empty transcription result!")
        return

    output_ext = output_format["type"].lower()
    if input_path != None:
        extension = os.path.splitext(input_path)[1]
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

    try:
        with open(output_path, "w", encoding="utf-8") as file:
            file.write(output_content)
        print_message(f"[SUCCESS] Transcription saved to {output_path}")
    except Exception as e:
        print_message(f"[ERROR] Failed to save file: {e}")

    if arg_print:
        print(output_content)

if __name__ == "__main__":
    main()
