# Instala antes en consola: pip install google-api-python-client pandas

import pandas as pd
from collections import defaultdict
from googleapiclient.discovery import build

# === API Key ===
API_KEY = "TU_API_KEY"
youtube = build("youtube", "v3", developerKey=API_KEY)

# === Palabras clave ===
keywords_international = [
    "Israel Palestine conflict", "Objective Middle East war coverage", 
    "Balanced report on Gaza situation", "UN official report on Israel Palestine",
    "International diplomatic response to Gaza", "Global perspective on Israel Palestine conflict",
    "World leaders statements on Israel Palestine", "Impartial media coverage Israel Gaza"
]

keywords_palestinian = [
    "Gaza latest news", "War crimes Israel Palestine", "Solidarity with Palestine",
    "Support for Palestinian rights", "Expose Israeli occupation", "Palestinian resistance narrative",
    "Critique Israeli policies", "Gaza under siege", "Palestinian struggle for justice",
    "Nakba remembrance", "Free Palestine", "Gaza under attack", "Israeli occupation",
    "Palestinian resistance", "Human rights in Gaza", "Israel war crimes in Palestine",
    "Nakba history", "Palestinian civilians killed", "Boycott Israel movement"
]

keywords_israeli = [
    "Israel latest news", "Support for Israeli defense", "Expose Hamas terrorism",
    "Israel under attack", "Israel self-defense narrative", "Critique of Palestinian extremism",
    "Israeli security first", "Defend Israeli sovereignty", "Praise IDF operations",
    "Hamas attack on Israel", "IDF response to Gaza", "Israel self-defense", "Hamas terrorism",
    "Hostages in Gaza", "Rockets fired at Israel", "Israeli settlements", "Iron Dome defense system"
]

channels_dict = defaultdict(lambda: {"mentions": 0, "category": None})

# === Función de búsqueda ===
def get_channels_from_youtube(query, category, max_results=50):
    request = youtube.search().list(
        q=query,
        part="snippet",
        maxResults=max_results,
        type="video",
        order="relevance",
        relevanceLanguage="en",
        publishedAfter="2023-10-07T00:00:00Z"
    )
    response = request.execute()

    for item in response["items"]:
        channel_title = item["snippet"]["channelTitle"]
        if channel_title in channels_dict:
            channels_dict[channel_title]["mentions"] += 1
            if channels_dict[channel_title]["category"] == "International":
                continue
            elif category == "International":
                channels_dict[channel_title]["category"] = "International"
            elif channels_dict[channel_title]["category"] is None or channels_dict[channel_title]["mentions"] < 5:
                channels_dict[channel_title]["category"] = category
        else:
            channels_dict[channel_title] = {"mentions": 1, "category": category}

# === Ejecutar ===
for keyword in keywords_international:
    get_channels_from_youtube(keyword, "International")

for keyword in keywords_palestinian:
    get_channels_from_youtube(keyword, "Palestinian")

for keyword in keywords_israeli:
    get_channels_from_youtube(keyword, "Israeli")

# === Crear DataFrame ===
df_channels = pd.DataFrame.from_dict(channels_dict, orient="index").reset_index()
df_channels.columns = ["Channel", "Mentions", "Category"]
df_channels = df_channels.sort_values(by=["Category", "Mentions"], ascending=[True, False])

# === Mostrar y guardar ===
print("\n🔹 Selected YouTube Channels:")
print(df_channels.to_string(index=False))
df_channels.to_csv("selected_youtube_channels_full.csv", index=False)
