# === youtube_scraper.py ===

# Requiere instalación previa:
# pip install youtube_transcript_api google-api-python-client pandas

import time
import random
import pandas as pd
import os
import logging
import sys
from datetime import datetime, timezone, timedelta
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

# === Configuración ===
OUTPUT_FILE = "youtube_videos.csv"
API_KEYS = [  # Agrega tus claves API aquí
]
CHANNELS = [
    "UC16niRr50-MSBwiO3YDb3RA", "UCupvZG-5ko_eiXAupbDfxWw", "UCknLrEdhRCp1aegoMqRaCZg",
    "UCQfwfsi5VrQ8yKZ-UWmAEFg", "UC7fWeaHhqgM4Ry-RMpM2YYw", "UCR0fZh5SBxxMNYdg0VzRFkg",
    "UCzW1oJMWo5BpHys5QGbpJrA", "UCph-M7aQqpi1fW0BQwn7XUA", "UCNye-wNBqNL5ZzHSJj3l8Bg",
    "UCvHDpsWKADrDia0c99X37vg", "UCawNWlihdgaycQpO3zi-jYg", "UCuxgEyMeaks7HS5gAw6Tt7w",
    "UC9jY5IcAA99wX8MZQ9K9r-Q", "UC7oQL6bXDLVlfRzeF8_KcGw", "UCLuNhKOeBYMt1KcfUmQdfWA",
    "UCfmSignFWkk1lw4015hCyyQ", "UCLLLdCANnMAdMyrXdYbSlxg", "UCKM3VQFIITaRegPDkLlLeKA",
    "UCnA2ZZ_6P7DZbmUVEolAp3g", "UCr3tVUYEqvVJTPqkggTFnyg", "UC1xAKR0vwwrHeorDesok6vw",
    "UCyytKFUbE6M7YY22XDlQHYw"
]
MIN_VIEWS = 1000
MIN_COMMENTS = 5
QUERY = "Gaza Israel Palestine conflict"
DEFAULT_START_DATE = "2023-10-07T00:00:00Z"
INTERVAL_DAYS = 1
MAX_VIDEOS_POR_CANAL = 5

quota_exceeded_count = 0
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_current_iso_date():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')

def cargar_csv():
    if os.path.exists(OUTPUT_FILE):
        df = pd.read_csv(OUTPUT_FILE)
        logging.info("CSV existente cargado con %d registros.", len(df))
        return df
    else:
        logging.info("No se encontró CSV existente. Se creará uno nuevo.")
        return pd.DataFrame(columns=["video_id", "title", "channel", "date", "views", "likes", "comments", "transcript", "url"])

def obtener_fecha_inicio(df):
    if df.empty:
        return DEFAULT_START_DATE
    else:
        ult_fecha = pd.to_datetime(df["date"]).max()
        return (ult_fecha + pd.Timedelta(seconds=1)).isoformat()

def get_youtube_service():
    api_key = API_KEYS.pop(0)
    API_KEYS.append(api_key)
    return build("youtube", "v3", developerKey=api_key)

def get_transcription(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([t["text"] for t in transcript])
    except (TranscriptsDisabled, NoTranscriptFound):
        return None
    except Exception as e:
        logging.debug("Error obteniendo transcripción para %s: %s", video_id, str(e))
        return None

def iso_to_datetime(iso_str):
    return datetime.fromisoformat(iso_str.replace("Z", "+00:00"))

def check_quota_exceeded():
    global quota_exceeded_count
    quota_exceeded_count += 1
    if quota_exceeded_count >= len(API_KEYS):
        logging.error("Todas las claves están agotadas. Terminando la ejecución.")
        sys.exit(1)

def get_videos_from_channel(youtube, channel_id, start_date, end_date, max_per_channel=20):
    global quota_exceeded_count
    videos = []
    next_page_token = None
    contador = 0

    while contador < max_per_channel:
        try:
            request = youtube.search().list(
                part="snippet", type="video", channelId=channel_id,
                q=QUERY, maxResults=50, order="date",
                publishedAfter=start_date, publishedBefore=end_date,
                pageToken=next_page_token
            )
            response = request.execute()
        except HttpError as e:
            if "quotaExceeded" in str(e):
                check_quota_exceeded()
                return get_videos_from_channel(get_youtube_service(), channel_id, start_date, end_date, max_per_channel)
            else:
                return [], youtube

        quota_exceeded_count = 0
        videos.extend(response.get("items", []))
        contador += len(response.get("items", []))
        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    videos.sort(key=lambda x: x["snippet"]["publishedAt"])
    return videos, youtube

def get_video_details(youtube, video_ids):
    global quota_exceeded_count
    details = {}
    try:
        response = youtube.videos().list(
            part="statistics,snippet", id=",".join(video_ids)
        ).execute()
        for item in response.get("items", []):
            video_id = item["id"]
            stats = item.get("statistics", {})
            snippet = item.get("snippet", {})
            details[video_id] = {
                "views": int(stats.get("viewCount", 0)),
                "likes": int(stats.get("likeCount", 0)),
                "comments": int(stats.get("commentCount", 0)),
                "date": snippet.get("publishedAt")
            }
    except HttpError as e:
        if "quotaExceeded" in str(e):
            check_quota_exceeded()
            return get_video_details(get_youtube_service(), video_ids)
    else:
        quota_exceeded_count = 0
    return details, youtube

def main():
    df_existente = cargar_csv()
    fecha_inicio_str = obtener_fecha_inicio(df_existente)
    fecha_inicio = iso_to_datetime(fecha_inicio_str)
    fecha_actual = iso_to_datetime(get_current_iso_date())
    nuevos_registros = []
    youtube = get_youtube_service()
    intervalo = timedelta(days=INTERVAL_DAYS)
    current_start = fecha_inicio

    while current_start < fecha_actual:
        current_end = min(current_start + intervalo, fecha_actual)
        start_str = current_start.isoformat(timespec='seconds').replace("+00:00", "Z")
        end_str = current_end.isoformat(timespec='seconds').replace("+00:00", "Z")

        for canal in CHANNELS:
            videos, youtube = get_videos_from_channel(youtube, canal, start_str, end_str, max_per_channel=20)
            if len(videos) > MAX_VIDEOS_POR_CANAL:
                videos = random.sample(videos, MAX_VIDEOS_POR_CANAL)

            video_ids = [item["id"]["videoId"] for item in videos
                         if item["id"]["videoId"] not in df_existente.get("video_id", pd.Series()).values]

            if not video_ids:
                continue

            detalles, youtube = get_video_details(youtube, video_ids)
            for item in videos:
                video_id = item["id"]["videoId"]
                if video_id not in detalles:
                    continue
                stats = detalles[video_id]
                if stats["views"] < MIN_VIEWS or stats["comments"] < MIN_COMMENTS:
                    continue
                transcript = get_transcription(video_id)
                if transcript is None:
                    continue

                registro = {
                    "video_id": video_id,
                    "title": item["snippet"]["title"],
                    "channel": item["snippet"]["channelTitle"],
                    "date": stats["date"],
                    "views": stats["views"],
                    "likes": stats["likes"],
                    "comments": stats["comments"],
                    "transcript": transcript,
                    "url": f"https://www.youtube.com/watch?v={video_id}"
                }
                nuevos_registros.append(registro)
                df_existente = pd.concat([df_existente, pd.DataFrame([registro])], ignore_index=True)
                time.sleep(random.uniform(1, 3))

        if nuevos_registros:
            pd.DataFrame(nuevos_registros).to_csv(OUTPUT_FILE, mode='a', header=not os.path.exists(OUTPUT_FILE), index=False)
            nuevos_registros = []

        current_start = current_end

    logging.info("Extracción completada.")

if __name__ == "__main__":
    main()
