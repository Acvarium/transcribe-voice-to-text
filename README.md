# Voice to Text Transcription Tool

This utility allows you to transcribe audio or video files using OpenAI's Whisper model via a lightweight persistent local server. It automatically starts the server if it’s not running and sends files for transcription. The server remains in memory to avoid model reloading for each file, and it shuts down automatically after a configurable period of inactivity.

## Features

- Supports multiple languages (configurable via config.json)
- Uses OpenAI's Whisper model for high-quality transcription
- Multiple output formats (TXT, JSON, SRT)
- Optional timestamps and confidence scores
- Configurable model size and expandable segments

## Requirements

- Python 3.x
- OpenAI Whisper
- CUDA-capable GPU (recommended for better performance)

## Installation

1. Clone this repository

2. Copy `config.example.json` to `config.json` and setup

3. Create a virtual environment:
   
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Linux/Mac
   # or
   .venv\Scripts\activate  # On Windows
   ```

4. Install dependencies:
   
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Edit `config.json` to customize the transcription settings:

### Basic Settings

- `language`: Target language code (e.g., "uk" for Ukrainian, "en" for English)
- `model`: Whisper model size ("tiny", "base", "small", "medium", "large-v3")
- `expandable_segments`: Enable/disable expandable segments for better memory management
- `unverified_ssl_context`: Enable/disable verify ssl certificate if you got `CERTIFICATE_VERIFY_FAILED` error

### Output Format Settings

The `output_format` section allows you to customize how the transcription is saved:

```json
{
    "output_format": {
        "type": "txt",
        "include_timestamps": false,
        "include_confidence": false
    }
}
```

Available options:

- `type`: Output format type (choose one):
  - `"txt"`: Plain text format
  - `"json"`: JSON format with full transcription data
  - `"srt"`: SubRip subtitle format
- `include_timestamps`: Add timestamps to the output (true/false)
- `include_confidence`: Include confidence scores for each segment (true/false, only for txt format)

Example configurations:

1. Basic text output:
   
   ```json
   "output_format": {
    "type": "txt",
    "include_timestamps": false,
    "include_confidence": false
   }
   ```

2. Text with timestamps and confidence:
   
   ```json
   "output_format": {
    "type": "txt",
    "include_timestamps": true,
    "include_confidence": true
   }
   ```

3. Full JSON output:
   
   ```json
   "output_format": {
    "type": "json",
    "include_timestamps": true
   }
   ```

4. SRT subtitles:
   
   ```json
   "output_format": {
    "type": "srt"
   }
   ```

## Usage

#### 1. Transcribe a file

```bash
python transcribe.py -i path/to/audio.mp3 -o output.srt
```

Optional arguments:

- `-i`, `--input` — path to the input media file

- `-o`, `--output` — path to the output file 

- `-l`, `--language` — Override language

- `-t`, `--timestamp` — `true` / `false`, include timestamps

- `-c`, `--confidence` — `true` / `false`, include confidence scores

- `-p`, `--print` — Print transcription to terminal

If the server is not running, it will be launched automatically.

If the client is started without parameters, it will prompt you to enter the path to the input file and will save the resulting file alongside it

#### 2. Stop the server manually

```bash
python transcribe.py --stop
```

## How it works

* The first time you run the utility, it launches a lightweight HTTP server (whisper_server.py) in the background.

* The Whisper model is loaded once and stays in memory.

* Each transcribe.py run communicates with the server via HTTP and gets a transcription result.

* If no new files are processed for (e.g.) 5 minutes, the server will shut down automatically to free up resources.

## License

This project is open source and available under the MIT License. 