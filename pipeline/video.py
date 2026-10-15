import os
import json
import subprocess
import cv2
import numpy as np

from pipeline import log


def load_video(video_path, output_folder, max_height=896, max_fps=60):
    """
    Read every frame of a video as (T,H,W,3) RGB, at most max_height tall (even width and
    height) and at most max_fps (by skipping frames). Returns (frames, fps, (width, height)
    of the video itself).
    """
    assert os.path.exists(video_path), f"{video_path} does not exist"
    os.makedirs(output_folder, exist_ok=True)

    if get_video_codec(video_path) == "av1":
        log.detail("AV1 video, converting to H.264")
        video_path = convert_av1_to_h264(video_path, output_folder)

    cap = cv2.VideoCapture(video_path)
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    input_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    assert input_fps > 0, f"input_fps is not positive: {input_fps}"

    target_fps = min(input_fps, max_fps)
    frame_skip = round(input_fps / target_fps)

    nframes, frames = video2frames(video_path, max_height=max_height, frame_skip=frame_skip)
    return frames, target_fps, input_size


def video2frames(vidfile, max_height=None, frame_skip=1):
    """Decode a video to RGB frames, keeping every frame_skip-th, resized by max_height."""
    frames = []
    count = 0
    frame_index = 0

    cap = cv2.VideoCapture(vidfile)
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_index += 1

            # Skip frames based on frame_skip parameter
            if frame_index % frame_skip != 0:
                continue

            # Process frame dimensions and resize if needed
            processed_frame = _process_frame_dimensions(frame, max_height)

            frames.append(processed_frame[..., ::-1])  # BGR -> RGB
            count += 1

    finally:
        cap.release()

    return count, np.array(frames)


def _process_frame_dimensions(frame, max_height=None):
    """Process frame dimensions to ensure even width/height and apply max_height constraint."""
    height, width = frame.shape[:2]

    # Ensure dimensions are even (divisible by 2)
    adjusted_width = width + (width % 2)
    adjusted_height = height + (height % 2)

    # Apply max_height constraint if specified
    if max_height is not None and adjusted_height > max_height:
        # Ensure max_height is even
        target_height = max_height - (max_height % 2)
        # Calculate proportional width and ensure it's even
        target_width = int(adjusted_width * target_height / adjusted_height)
        target_width = target_width - (target_width % 2)

        adjusted_width, adjusted_height = target_width, target_height

    # Resize frame if dimensions changed
    if adjusted_width != width or adjusted_height != height:
        return cv2.resize(frame, (adjusted_width, adjusted_height), interpolation=cv2.INTER_CUBIC)

    return frame


def get_video_codec(video_path):
    """Get the codec of a video file using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name",
        "-of", "json",
        video_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        codec_info = json.loads(result.stdout)
        return codec_info["streams"][0]["codec_name"]
    except (subprocess.CalledProcessError, KeyError, IndexError, json.JSONDecodeError):
        return None


def convert_av1_to_h264(video_path, output_dir=None):
    """Convert AV1 video to H.264 format using ffmpeg."""
    if output_dir is None:
        output_dir = os.path.dirname(video_path)

    output_path = os.path.join(output_dir, os.path.splitext(os.path.basename(video_path))[0] + "_h264.mp4")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-i", video_path,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        output_path
    ]
    try:
        subprocess.run(cmd, check=True)
        return output_path
    except subprocess.CalledProcessError:
        raise Exception(f"Failed to convert AV1 to H.264: {video_path}")
