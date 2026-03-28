import cv2
import base64
import json
import ollama
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional
import time


# ── Config ─────────────────────────────────────────────────────────────────────

MODEL        = "qwen3-vl:4b"   # or "llava", "gemma3", "minicpm-v", etc.
VIDEO_PATH   = "./workspace/inputs/vision_test.mp4"  # path to your test video
SAMPLE_EVERY = 1                   # seconds between sampled frames
MAX_WORKERS  = 1                  # concurrent VLM calls (tune to your hardware)

PROMPT = """
Look at this screenshot from a screen recording.

Answer ONLY with a valid JSON object (no markdown, no extra text):
{
  "flagged": true or false,
  "reason": "brief explanation",
  "confidence": "high" | "medium" | "low"
}

Flag as true if ANY of the following are visible:
- A .env file is open in an editor (look for lines like API_KEY=, SECRET=, TOKEN=, DB_PASSWORD=, etc.)
- A terminal showing contents of a .env file (e.g. cat .env)
- Environment variable secrets exposed in any form

Flag as false otherwise.
""".strip()


# ── Data model ─────────────────────────────────────────────────────────────────

@dataclass
class FrameResult:
    timestamp_sec:  float
    frame_number:   int
    flagged:        bool
    reason:         str
    confidence:     str
    raw_response:   str
    processing_sec: float = 0.0        # time spent on this single frame
    error:          Optional[str] = None


# ── Core: encode frame → ask VLM ───────────────────────────────────────────────

def frame_to_base64(frame_bgr) -> str:
    """Convert an OpenCV BGR frame to a base64-encoded JPEG string."""
    success, buffer = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not success:
        raise ValueError("Failed to encode frame as JPEG")
    return base64.b64encode(buffer).decode("utf-8")


def analyze_frame(frame_bgr, frame_number: int, timestamp_sec: float) -> FrameResult:
    """Send a single frame to the VLM and parse the result."""
    t0 = time.perf_counter()
    try:
        img_b64 = frame_to_base64(frame_bgr)

        response = ollama.chat(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": PROMPT,
                    "images": [img_b64],   # pass raw base64 string
                }
            ],
            think=False,
            options={"temperature": 0},    # deterministic — important for classifiers
        )

        raw = response.message.content.strip()

        # Strip markdown fences if the model wraps JSON in ```json ... ```
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        parsed = json.loads(raw)

        return FrameResult(
            timestamp_sec=timestamp_sec,
            frame_number=frame_number,
            flagged=bool(parsed.get("flagged", False)),
            reason=parsed.get("reason", ""),
            confidence=parsed.get("confidence", "unknown"),
            raw_response=raw,
            processing_sec=time.perf_counter() - t0,
        )

    except Exception as e:
        return FrameResult(
            timestamp_sec=timestamp_sec,
            frame_number=frame_number,
            flagged=False,
            reason="",
            confidence="unknown",
            raw_response="",
            processing_sec=time.perf_counter() - t0,
            error=str(e),
        )


# ── Video sampling ─────────────────────────────────────────────────────────────

def sample_frames(video_path: str, every_n_seconds: float):
    """
    Generator — yields (frame_bgr, frame_number, timestamp_sec) 
    for every Nth second of the video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps        = cap.get(cv2.CAP_PROP_FPS)
    hop        = max(1, int(fps * every_n_seconds))  # frames to skip
    frame_idx  = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % hop == 0:
                timestamp = frame_idx / fps
                yield frame, frame_idx, timestamp
            frame_idx += 1
    finally:
        cap.release()


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run_pipeline(video_path: str, stop_on_first_flag: bool = False):
    start = time.time()
    results: list[FrameResult] = []
    flagged_frames: list[FrameResult] = []

    frames = list(sample_frames(video_path, SAMPLE_EVERY))
    print(f"[INFO] Sampled {len(frames)} frames from '{video_path}' (1 frame/{SAMPLE_EVERY}s)")
    print(f"[INFO] Submitting to VLM '{MODEL}' with {MAX_WORKERS} workers...\n")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(analyze_frame, frame, fn, ts): (fn, ts)
            for frame, fn, ts in frames
        }

        for future in as_completed(futures):
            result = future.result()
            results.append(result)

            status = "🚨 FLAGGED" if result.flagged else "✅ OK"
            err    = f" [ERROR: {result.error}]" if result.error else ""
            print(f"  [{result.timestamp_sec:6.1f}s | frame {result.frame_number:5d}]  "
                  f"{status}  {result.reason}{err}"
                  f"  ⏱ {result.processing_sec:.2f}s")

            if result.flagged:
                flagged_frames.append(result)
                if stop_on_first_flag:
                    pool.shutdown(wait=False, cancel_futures=True)
                    break

    # ── Report ─────────────────────────────────────────────────────────────────
    elapsed = time.time() - start
    print(f"\n{'─'*60}")
    print(f"Total wall-clock time : {elapsed:.2f}s")
    print(f"Frames processed      : {len(results)}")
    if results:
        times = [r.processing_sec for r in results]
        print(f"Per-frame time        : avg {sum(times)/len(times):.2f}s  "
              f"min {min(times):.2f}s  max {max(times):.2f}s")
    print(f"Flagged               : {len(flagged_frames)} frame(s)\n")

    if flagged_frames:
        # Sort by timestamp for a clean report
        for r in sorted(flagged_frames, key=lambda x: x.timestamp_sec):
            print(f"  ⚠️  {r.timestamp_sec:.1f}s  — {r.reason}  (confidence: {r.confidence})")

    return results, flagged_frames


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results, flagged = run_pipeline(VIDEO_PATH, stop_on_first_flag=False)

    if flagged:
        print("\n❌ Video contains sensitive/flagged content — review before publishing.")
    else:
        print("\n✅ No sensitive content detected.")