import re
import json
import requests
import subprocess
import logging
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    CouldNotRetrieveTranscript,
)

# -----------------------------------------------------------
# LOGGER (silent by default; Streamlit can show via st.write)
# -----------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# -----------------------------------------------------------
# 🔹 Extract YouTube video ID robustly
# -----------------------------------------------------------
def extract_video_id(url: str) -> str:
    """
    Extracts the 11-char YouTube video ID from both long and short URLs.
    Handles: https://www.youtube.com/watch?v=..., https://youtu.be/..., or raw IDs.
    """
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", url)
    if match:
        return match.group(1)
    return url.strip()


# -----------------------------------------------------------
# 🔹 Fetch transcript (official API + yt-dlp fallback)
# -----------------------------------------------------------
def fetch_transcript(video_id: str):
    """
    Fetch YouTube transcript using multiple fallback strategies:
      1️⃣ Official YouTubeTranscriptApi (manual & auto-generated captions)
      2️⃣ yt-dlp fallback: parses JSON/VTT/SRT if API fails
    Returns: list of dicts like {"text": "...", "start": seconds}
    Raises: Exception with user-friendly message if all fail.
    """
    lang_priority = ["en", "en-US", "hi", "hi-IN"]

    # ---- 1️⃣ Try official transcript API ----
    try:
        list_result = YouTubeTranscriptApi.list_transcripts(video_id)

        # (a) Human-created transcripts
        for code in lang_priority:
            try:
                tr = list_result.find_transcript([code])
                data = tr.fetch()
                if data:
                    logger.info(f"✅ Official transcript ({code}) · {len(data)} segments")
                    return data
            except Exception:
                pass

        # (b) Auto-generated transcripts
        for code in lang_priority:
            try:
                tr = list_result.find_generated_transcript([code])
                data = tr.fetch()
                if data:
                    logger.info(f"✅ Auto-generated transcript ({code}) · {len(data)} segments")
                    return data
            except Exception:
                pass

    except (TranscriptsDisabled, NoTranscriptFound, CouldNotRetrieveTranscript) as e:
        logger.warning(f"⚠️ list_transcripts reported: {e}")
    except Exception as e:
        logger.warning(f"⚠️ Unexpected error from list_transcripts: {e}")

    # ---- 2️⃣ yt-dlp captions fallback ----
    try:
        logger.info("▶ Trying yt-dlp captions fallback...")
        result = subprocess.run(
            ["yt-dlp", "-J", f"https://www.youtube.com/watch?v={video_id}"],
            capture_output=True, text=True, check=False
        )
        if not result.stdout.strip():
            raise RuntimeError("yt-dlp returned empty stdout")

        info = json.loads(result.stdout)
        subs = info.get("subtitles") or info.get("automatic_captions") or {}
        if not subs:
            raise NoTranscriptFound("No captions found via yt-dlp")

        # Choose preferred language
        chosen_lang = next((code for code in lang_priority if code in subs), None)
        if not chosen_lang:
            chosen_lang = next(iter(subs.keys()))

        cap_url = subs[chosen_lang][0]["url"]
        txt = requests.get(cap_url, headers={"User-Agent": "Mozilla/5.0"}).text

        segs = []

        # WEBVTT parser
        if "WEBVTT" in txt:
            entries = re.findall(r"(\d{2}:\d{2}:\d{2}\.\d{3}) --> .*?\n(.*?)\n", txt, re.S)
            for start, line in entries:
                h, m, s = [float(x) for x in re.split("[:.]", start)]
                segs.append({"text": line.strip(), "start": h * 3600 + m * 60 + s})

        # JSON3 or list-like
        elif txt.strip().startswith("{") or txt.strip().startswith("["):
            try:
                data = json.loads(txt)
                events = data.get("events", data if isinstance(data, list) else [])
                for ev in events:
                    if ev.get("segs"):
                        text = "".join([s.get("utf8", "") for s in ev["segs"]]).strip()
                        if text:
                            segs.append({"text": text, "start": ev.get("tStartMs", 0) / 1000.0})
            except Exception as e:
                logger.warning(f"⚠️ JSON3 parse failed: {e}")

        # SRT format (rare)
        elif re.search(r"\d+\s+\d{2}:\d{2}:\d{2},\d{3}", txt):
            lines = re.findall(r"\d+\s+\d{2}:\d{2}:\d{2},\d{3} --> .*?\n(.*?)\n\n", txt, re.S)
            segs = [{"text": t.strip(), "start": i * 4.0} for i, t in enumerate(lines)]

        if segs:
            logger.info(f"✅ yt-dlp extracted {len(segs)} caption lines ({chosen_lang})")
            return segs
        else:
            logger.warning("⚠️ yt-dlp returned captions but could not parse content.")

    except Exception as e:
        logger.warning(f"❌ yt-dlp fallback failed: {e}")

    # ---- ❌ Nothing worked ----
    raise Exception(
        "❌ No transcript available via API or captions fallback.\n"
        "Please try a different video with captions enabled (English or Hindi)."
    )


# -----------------------------------------------------------
# 🔹 Utilities
# -----------------------------------------------------------
def transcript_to_text(transcript):
    """Flatten list of {text,start} into a single concatenated string."""
    return " ".join(
        [seg.get("text", "").strip() for seg in transcript if seg.get("text", "").strip()]
    )


def get_transcript_text(url: str) -> str:
    """
    One-shot convenience wrapper:
      - Extracts video ID
      - Fetches transcript
      - Converts to plain text
    Returns a single large string suitable for FAISS indexing.
    """
    video_id = extract_video_id(url)
    data = fetch_transcript(video_id)
    return transcript_to_text(data)
