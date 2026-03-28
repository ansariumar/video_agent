import cv2
import base64
import json
import ollama
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


# ── Config ─────────────────────────────────────────────────────────────────────

MODEL        = "qwen3-vl:4b"
VIDEO_PATH   = "./workspace/inputs/vision_test.mp4"
SAMPLE_EVERY = 1
MAX_WORKERS  = 2
LOG_FILE     = "pipeline.log"

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


# ── Logging setup ──────────────────────────────────────────────────────────────

def setup_logging(log_file: str) -> logging.Logger:
    logger = logging.getLogger("vlm_pipeline")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler — DEBUG and above (everything)
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    # Console handler — INFO and above (clean output)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

logger = setup_logging(LOG_FILE)


# ── Timer utility ──────────────────────────────────────────────────────────────

class Timer:
    """
    Context manager + manual timer.
    Usage:
        with Timer("my operation") as t:
            do_something()
        # t.elapsed is set after the block

        t = Timer("encode").start()
        ...
        t.stop()   # logs automatically
    """
    def __init__(self, label: str, log_level: int = logging.DEBUG):
        self.label     = label
        self.log_level = log_level
        self.elapsed:  float = 0.0
        self._start:   float = 0.0

    def start(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def stop(self) -> float:
        self.elapsed = time.perf_counter() - self._start
        logger.log(self.log_level, "⏱  %-40s  %.3fs", self.label, self.elapsed)
        return self.elapsed

    def __enter__(self) -> "Timer":
        return self.start()

    def __exit__(self, *_):
        self.stop()


# ── Data model ─────────────────────────────────────────────────────────────────

@dataclass
class FrameResult:
    timestamp_sec:   float
    frame_number:    int
    flagged:         bool
    reason:          str
    confidence:      str
    raw_response:    str
    encode_ms:       float = 0.0   # time to JPEG-encode the frame
    inference_ms:    float = 0.0   # time the VLM took to respond
    total_ms:        float = 0.0   # end-to-end for this frame
    error:           Optional[str] = None


# ── Core: encode frame → ask VLM ───────────────────────────────────────────────

def frame_to_base64(frame_bgr) -> tuple[str, float]:
    """Encode frame to base64 JPEG. Returns (b64_string, encode_time_ms)."""
    t = Timer("frame_to_base64").start()
    success, buffer = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    encode_ms = t.stop() * 1000
    if not success:
        raise ValueError("Failed to encode frame as JPEG")
    return base64.b64encode(buffer).decode("utf-8"), encode_ms


def analyze_frame(frame_bgr, frame_number: int, timestamp_sec: float) -> FrameResult:
    """Send a single frame to the VLM and parse the result."""
    frame_timer = Timer(f"frame {frame_number} end-to-end").start()
    logger.debug("→ Sending frame %d (%.1fs) to VLM", frame_number, timestamp_sec)

    try:
        img_b64, encode_ms = frame_to_base64(frame_bgr)
        logger.debug("  frame %d | encoded in %.1fms | payload ~%d KB",
                     frame_number, encode_ms, len(img_b64) // 1024)

        inference_timer = Timer(f"frame {frame_number} VLM inference").start()
        response = ollama.chat(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": PROMPT,
                    "images": [img_b64],
                }
            ],
            options={"temperature": 0},
        )
        inference_ms = inference_timer.stop() * 1000

        raw = response.message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        parsed    = json.loads(raw)
        total_ms  = frame_timer.stop() * 1000

        result = FrameResult(
            timestamp_sec=timestamp_sec,
            frame_number=frame_number,
            flagged=bool(parsed.get("flagged", False)),
            reason=parsed.get("reason", ""),
            confidence=parsed.get("confidence", "unknown"),
            raw_response=raw,
            encode_ms=encode_ms,
            inference_ms=inference_ms,
            total_ms=total_ms,
        )

        logger.debug("  frame %d | flagged=%s | inference=%.0fms | total=%.0fms",
                     frame_number, result.flagged, inference_ms, total_ms)
        return result

    except Exception as e:
        total_ms = frame_timer.stop() * 1000
        logger.error("  frame %d | ERROR after %.0fms: %s", frame_number, total_ms, e)
        return FrameResult(
            timestamp_sec=timestamp_sec,
            frame_number=frame_number,
            flagged=False,
            reason="",
            confidence="unknown",
            raw_response="",
            encode_ms=0.0,
            inference_ms=0.0,
            total_ms=total_ms,
            error=str(e),
        )


# ── Video sampling ─────────────────────────────────────────────────────────────

def sample_frames(video_path: str, every_n_seconds: float):
    """Generator — yields (frame_bgr, frame_number, timestamp_sec)."""
    logger.info("Opening video: %s", video_path)

    with Timer("video open + frame sampling", log_level=logging.INFO):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        fps       = cap.get(cv2.CAP_PROP_FPS)
        total_f   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration  = total_f / fps if fps else 0
        hop       = max(1, int(fps * every_n_seconds))
        frame_idx = 0

        logger.info("  FPS=%.2f | total_frames=%d | duration=%.1fs | sampling every %ds",
                    fps, total_f, duration, every_n_seconds)

        frames = []
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % hop == 0:
                    frames.append((frame.copy(), frame_idx, frame_idx / fps))
                frame_idx += 1
        finally:
            cap.release()

    logger.info("Sampled %d frames to analyse", len(frames))
    return frames


# ── Performance summary ────────────────────────────────────────────────────────

def log_performance_summary(results: list[FrameResult], wall_time: float):
    valid = [r for r in results if not r.error]
    if not valid:
        logger.warning("No valid results to summarise.")
        return

    inference_times = [r.inference_ms for r in valid]
    encode_times    = [r.encode_ms    for r in valid]
    total_times     = [r.total_ms     for r in valid]

    def stats(vals):
        avg = sum(vals) / len(vals)
        mn  = min(vals)
        mx  = max(vals)
        return avg, mn, mx

    inf_avg, inf_min, inf_max   = stats(inference_times)
    enc_avg, enc_min, enc_max   = stats(encode_times)
    tot_avg, tot_min, tot_max   = stats(total_times)
    throughput = len(valid) / wall_time if wall_time > 0 else 0

    summary = f"""
╔══════════════════════════════════════════════════════╗
║              PERFORMANCE SUMMARY                     ║
╠══════════════════════════════════════════════════════╣
║  Frames analysed : {len(valid):<5}  (errors: {len(results)-len(valid):<3})           ║
║  Wall time       : {wall_time:>7.2f}s                            ║
║  Throughput      : {throughput:>7.2f} frames/s                    ║
╠══════════════════════════════════════════════════════╣
║  JPEG encode     avg={enc_avg:>6.1f}ms  min={enc_min:>5.1f}  max={enc_max:>5.1f} ║
║  VLM inference   avg={inf_avg:>6.1f}ms  min={inf_min:>5.1f}  max={inf_max:>5.1f} ║
║  Frame total     avg={tot_avg:>6.1f}ms  min={tot_min:>5.1f}  max={tot_max:>5.1f} ║
╠══════════════════════════════════════════════════════╣
║  Flagged frames  : {len([r for r in valid if r.flagged])}                               ║
╚══════════════════════════════════════════════════════╝"""

    # Log to both console (INFO) and file (DEBUG gets the per-frame breakdown too)
    for line in summary.strip().splitlines():
        logger.info(line)

    # Per-frame timing table — file only (DEBUG)
    logger.debug("\nPer-frame timing breakdown:")
    logger.debug("%-8s %-10s %-12s %-12s %-12s %-8s",
                 "Frame", "Time(s)", "Encode(ms)", "Infer(ms)", "Total(ms)", "Flagged")
    logger.debug("-" * 70)
    for r in sorted(results, key=lambda x: x.frame_number):
        logger.debug("%-8d %-10.1f %-12.1f %-12.1f %-12.1f %-8s",
                     r.frame_number, r.timestamp_sec,
                     r.encode_ms, r.inference_ms, r.total_ms,
                     "YES" if r.flagged else "no")


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run_pipeline(video_path: str, stop_on_first_flag: bool = False):
    logger.info("=" * 60)
    logger.info("Pipeline start | model=%s | workers=%d | sample_every=%ds",
                MODEL, MAX_WORKERS, SAMPLE_EVERY)
    logger.info("=" * 60)

    pipeline_timer = Timer("total pipeline", log_level=logging.INFO).start()

    frames  = sample_frames(video_path, SAMPLE_EVERY)
    results: list[FrameResult] = []
    flagged: list[FrameResult] = []

    logger.info("Submitting %d frames to thread pool...", len(frames))

    with Timer("thread pool execution", log_level=logging.INFO):
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
                logger.info("  [%6.1fs | frame %5d]  %-12s  %s%s",
                            result.timestamp_sec, result.frame_number,
                            status, result.reason, err)

                if result.flagged:
                    flagged.append(result)
                    if stop_on_first_flag:
                        logger.info("stop_on_first_flag=True — shutting down pool early.")
                        pool.shutdown(wait=False, cancel_futures=True)
                        break

    wall_time = pipeline_timer.stop()

    # ── Final report ───────────────────────────────────────────────────────────
    logger.info("-" * 60)
    if flagged:
        logger.warning("%d flagged frame(s):", len(flagged))
        for r in sorted(flagged, key=lambda x: x.timestamp_sec):
            logger.warning("  ⚠️  %.1fs — %s (confidence: %s)", 
                           r.timestamp_sec, r.reason, r.confidence)
    else:
        logger.info("No flagged frames found.")

    log_performance_summary(results, wall_time)
    logger.info("Log written to: %s", Path(LOG_FILE).resolve())

    return results, flagged


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results, flagged = run_pipeline(VIDEO_PATH, stop_on_first_flag=False)

    if flagged:
        print("\n❌ Video contains sensitive content — review before publishing.")
    else:
        print("\n✅ No sensitive content detected.")