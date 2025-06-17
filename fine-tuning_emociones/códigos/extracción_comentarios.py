# extraccion_comentarios_youtube.py

import os
import pandas as pd
import requests
import random
import time
from langdetect import detect
from tqdm import tqdm
from datetime import datetime

# 📦 Archivos por categoría
archivos = {
    "Palestina": "narrativas_palestina_clusterizadas.csv",
    "Israel": "narrativas_israel_clusterizadas.csv",
    "Internacional": "narrativas_internacional_clusterizadas.csv",
    "Internacional Medio Oriente": "narrativas_internacional_medio_oriente_clusterizadas.csv"
}

# 📊 Cargar resumen y normalizar columnas
df_resumen = pd.read_csv("resumen_topicos_por_categoria.csv")
df_resumen.columns = df_resumen.columns.str.strip()
columnas_topicos = [col for col in df_resumen.columns if col.startswith("Tópico_")]

# 📓 Cargar log incremental
try:
    log = pd.read_csv("log_comentarios_extraidos.csv")
    print("✅ Log cargado con", len(log), "comentarios.")
except FileNotFoundError:
    log = pd.DataFrame(columns=["comment_id", "video_id", "categoria", "topic", "fecha", "text"])
    print("📁 Log creado desde cero.")

# 🔐 Lista de claves
API_KEYS = [
    # Agrega aquí tus claves de API de YouTube
]

# 📥 Función para extraer comentarios
def extraer_comentarios(video_id, categoria, topic, max_results=40):
    global API_KEYS, log
    comentarios = []

    for api_key in API_KEYS:
        url = f"https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&videoId={video_id}&maxResults={max_results}&textFormat=plainText&key={api_key}"
        try:
            response = requests.get(url)
            data = response.json()
            if 'items' not in data:
                continue
            for item in data['items']:
                comment = item['snippet']['topLevelComment']['snippet']
                comment_id = item['id']
                text = comment['textDisplay']
                fecha = comment['publishedAt'][:10]

                if comment_id in log["comment_id"].values:
                    continue
                if len(text) < 20 or any(x in text.lower() for x in ['http', '@', '#', 'subscribe']):
                    continue
                try:
                    if detect(text) != "en":
                        continue
                except:
                    continue

                comentarios.append({
                    "comment_id": comment_id,
                    "video_id": video_id,
                    "categoria": categoria,
                    "topic": topic,
                    "fecha": fecha,
                    "text": text
                })

            break  # Si una key funcionó, no pruebes las demás
        except Exception:
            continue

    return comentarios

# 🔁 Parámetros
N_VIDEOS_POR_TOPICO = 50
MAX_COMENTARIOS_POR_VIDEO = 80
todos_comentarios = []

# 🔄 Iteración por categoría y tópicos existentes
for _, row in df_resumen.iterrows():
    categoria = row["Categoría"].strip()
    archivo = archivos[categoria]
    df_cat = pd.read_csv(archivo)

    for col in columnas_topicos:
        if col not in row or pd.isna(row[col]):
            continue
        try:
            numero_topico = int(col.split("_")[1])
        except:
            continue

        df_topico = df_cat[df_cat["topic"] == numero_topico]
        if len(df_topico) < N_VIDEOS_POR_TOPICO:
            continue

        videos_muestreados = df_topico.sample(N_VIDEOS_POR_TOPICO, random_state=random.randint(1, 10000))
        for _, video in videos_muestreados.iterrows():
            comentarios = extraer_comentarios(
                video["video_id"], categoria, numero_topico, max_results=MAX_COMENTARIOS_POR_VIDEO
            )
            todos_comentarios.extend(comentarios)

# ✅ Combinar nuevos con anteriores y eliminar duplicados
df_nuevos = pd.DataFrame(todos_comentarios)
df_total = pd.concat([log, df_nuevos], ignore_index=True).drop_duplicates(subset=["comment_id"])
df_total.to_csv("log_comentarios_extraidos.csv", index=False)

print(f"✅ Nuevos comentarios agregados: {len(df_nuevos)}")
print(f"🧾 Total acumulado: {len(df_total)} comentarios únicos")
