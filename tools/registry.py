"""
Tool Registry.
Single source of truth — all tools the agent can call, in one list.
To add a new tool: implement it in the relevant module, then import and add it here.
"""

from tools.trim import trim_video, merge_videos, split_video
from tools.audio import extract_audio, mute_video, replace_audio, adjust_volume, add_background_music
from tools.transform import resize_video, crop_video, rotate_video, flip_video, change_speed, reverse_video
from tools.overlay import add_text_overlay, add_image_watermark, add_subtitles, generate_thumbnail
from tools.effects import apply_color_filter, adjust_brightness_contrast, add_fade, stabilize_video
from tools.export import convert_format, compress_video, change_resolution, create_gif, extract_frames, get_video_info

ALL_TOOLS = [
    # ── Inspection ──────────────────────────────────────────────────────────
    get_video_info,

    # ── Trimming & Merging ──────────────────────────────────────────────────
    trim_video,
    merge_videos,
    split_video,

    # ── Audio ───────────────────────────────────────────────────────────────
    extract_audio,
    mute_video,
    replace_audio,
    adjust_volume,
    add_background_music,

    # ── Visual Transformations ───────────────────────────────────────────────
    resize_video,
    crop_video,
    rotate_video,
    flip_video,
    change_speed,
    reverse_video,

    # ── Overlays ─────────────────────────────────────────────────────────────
    add_text_overlay,
    add_image_watermark,
    add_subtitles,
    generate_thumbnail,

    # ── Effects ──────────────────────────────────────────────────────────────
    apply_color_filter,
    adjust_brightness_contrast,
    add_fade,
    stabilize_video,

    # ── Export & Format ───────────────────────────────────────────────────────
    convert_format,
    compress_video,
    change_resolution,
    create_gif,
    extract_frames,
]