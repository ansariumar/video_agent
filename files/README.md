# 🎬 Video Editing Agent

An AI-powered video editing agent that accepts natural language commands and executes them using FFmpeg.
Built with **LangChain + Ollama + FastAPI**.

---

## Stack

| Component | Role |
|-----------|------|
| **Ollama** | Local LLM server (llama3.1, qwen2.5, mistral, etc.) |
| **LangChain** | Agent framework (`create_agent`, `@tool`) |
| **FFmpeg** | Video processing engine |
| **FastAPI** | HTTP API layer |

---

## Prerequisites

Make sure you have these installed:
- Python 3.11+
- FFmpeg (`ffmpeg --version` to verify)
- Ollama (`ollama --version` to verify)

---

## Setup

```bash
# 1. Clone / navigate to the project
cd video_agent

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate      # Linux/macOS
# venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Pull your preferred Ollama model
ollama pull llama3.1:8b
# or: ollama pull qwen2.5:7b
# or: ollama pull mistral:7b

# 5. Configure your model (optional)
# Edit config.py and change OLLAMA_MODEL to your preferred model

# 6. Start the server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

API docs will be available at: **http://localhost:8000/docs**

---

## Usage

### Put your video in the inputs folder
```
workspace/inputs/my_video.mp4
```

### Submit an editing task

```bash
curl -X POST http://localhost:8000/api/v1/task \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Trim the video from 00:00:05 to 00:00:20",
    "input_file": "my_video.mp4"
  }'
```

Response:
```json
{
  "job_id": "abc123-...",
  "status": "pending",
  "message": "Job created. Poll GET /status/abc123-... for updates."
}
```

### Check job status

```bash
curl http://localhost:8000/api/v1/status/abc123-...
```

### Download the output

```bash
curl -O http://localhost:8000/api/v1/download/abc123-...
```

### List all jobs

```bash
curl http://localhost:8000/api/v1/jobs
```

---

## Switching Models

You can switch models **per request** without restarting the server:

```bash
curl -X POST http://localhost:8000/api/v1/task \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Convert this video to grayscale",
    "input_file": "my_video.mp4",
    "model": "qwen2.5:7b"
  }'
```

Or change the default in `config.py`:
```python
OLLAMA_MODEL = "qwen2.5:7b"
```

---

## Available Tools (26 total)

### Inspection
- `get_video_info` — Get duration, resolution, codec, fps, file size

### Trimming & Merging
- `trim_video` — Cut a segment by start/end time
- `merge_videos` — Concatenate multiple videos
- `split_video` — Split at multiple timestamps

### Audio
- `extract_audio` — Rip audio track to mp3/aac/wav
- `mute_video` — Remove all audio
- `replace_audio` — Swap audio track
- `adjust_volume` — Make louder or quieter
- `add_background_music` — Mix in background music

### Visual Transformations
- `resize_video` — Scale to custom dimensions
- `crop_video` — Crop to a region
- `rotate_video` — Rotate 90/180/270 degrees
- `flip_video` — Horizontal or vertical flip
- `change_speed` — Slow motion or timelapse
- `reverse_video` — Play backwards

### Overlays
- `add_text_overlay` — Burn text/captions into video
- `add_image_watermark` — Add logo/watermark
- `add_subtitles` — Burn in SRT subtitles
- `generate_thumbnail` — Extract frame as image

### Effects
- `apply_color_filter` — Grayscale, sepia, blur, vignette, etc.
- `adjust_brightness_contrast` — Brightness/contrast/saturation
- `add_fade` — Fade in/out transitions
- `stabilize_video` — Reduce camera shake (requires libvidstab)

### Export & Format
- `convert_format` — Convert to mp4/mkv/webm/etc.
- `compress_video` — Reduce file size (high/medium/low/tiny)
- `change_resolution` — Set to 4K/1080p/720p/480p/etc.
- `create_gif` — Convert segment to animated GIF
- `extract_frames` — Dump frames as images

---

## Example Prompts

```
"Trim the video to keep only the first 30 seconds"
"Convert this to grayscale and compress it to medium quality"
"Add the text 'My Vacation 2024' at the top of the video"
"Speed up the video to 2x and convert to GIF"
"Mute the video and add background music from music.mp3"
"Extract the audio and save as MP3"
"Trim from 1:00 to 2:30, then add a fade in and fade out"
```

---

## Project Structure

```
video_agent/
├── main.py              # FastAPI app entry point
├── config.py            # All settings in one place
├── requirements.txt
├── agent/
│   ├── core.py          # LangChain agent + job runner
│   └── prompts.py       # System prompt
├── tools/
│   ├── registry.py      # Master tool list
│   ├── trim.py          # Trim, split, merge
│   ├── audio.py         # Audio tools
│   ├── transform.py     # Resize, crop, rotate, etc.
│   ├── overlay.py       # Text, watermark, subtitles
│   ├── effects.py       # Filters, fade, stabilize
│   └── export.py        # Format, compress, GIF, frames
├── jobs/
│   ├── manager.py       # Job lifecycle management
│   └── models.py        # Job data models
├── api/
│   ├── routes.py        # All API endpoints
│   └── schemas.py       # Request/response models
├── utils/
│   ├── ffmpeg_runner.py # FFmpeg subprocess wrapper
│   └── file_utils.py    # Path helpers
└── workspace/
    ├── inputs/          # Put source videos here
    └── outputs/         # Edited videos saved here
```
