# === Librerías necesarias ===
# pip install -U seaborn kneed langdetect nltk wordcloud umap-learn transformers

import os
import re
import string
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from langdetect import detect
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
from kneed import KneeLocator
from umap import UMAP
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk

# === CONFIGURACIÓN GLOBAL ===
tqdm.pandas()
sns.set(style="whitegrid")
resultados_dir = "./resultados_topicos"
os.makedirs(resultados_dir, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-large-v2")
model = AutoModel.from_pretrained("intfloat/e5-large-v2").to(device)
lemmatizer = WordNetLemmatizer()

# === NLTK ===
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')

# === Stopwords Personalizadas ===
custom_stopwords = set(stopwords.words('english')).union({
    "um", "uh", "uhh", "uhm", "yeah", "you", "know", "ok", "okay", "s", "m", "ve", "re", "ll", "t",
    "gonna", "wanna", "thing", "stuff", "look", "also", "would", "get", "many", "need", "two", "first", "year", "lot", "kind", "make", "go", "say", "come", "said", "think", "want",
    "really", "like", "well", "actually", "just", "mean", "much", "still", "one", "course",
    "let", "video", "guy", "time", "see", "back", "even", "going", "way", "seen", "day", "come",
    "israel", "gaza", "israeli", "people", "palestinian", "right", "palestine", "music", "latest", "news", "sure", "subscribe", "thank", "joining", "channel",
    "little", "bit", "whats", "happening", "last", "week", "music music", "latest news", "thank joining", "channel latest", "little bit", "whats happening",
    "last week", "thats", "live", "stream", "since", "since october", "electronic net", "every single", "electronic", "net", "social", "medium", "social medium", "every", "single", "every single"
})

# === FUNCIONES DE LIMPIEZA Y PREPROCESAMIENTO ===
def clean_text(text):
    if pd.isna(text): return ""
    text = str(text).encode("ascii", "ignore").decode()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def lemmatize_text(text):
    return " ".join([lemmatizer.lemmatize(w) for w in nltk.word_tokenize(text)])

def filtrar_palabras(texto):
    return " ".join([w for w in texto.split() if w not in custom_stopwords and len(w) > 2])

def detectar_idioma(texto):
    try: return detect(texto)
    except: return "unknown"

def average_pool(last_hidden_states, attention_mask):
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def get_e5_embedding(text):
    prompt = f"passage: {text}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return average_pool(outputs.last_hidden_state, inputs['attention_mask']).squeeze().cpu().numpy()

# === FUNCIONES DE ANÁLISIS Y CLUSTERING ===
def seleccionar_k_optimo(reduced, categoria):
    silhouette_scores, sse = [], []
    k_range = list(range(2, 16))
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42).fit(reduced)
        sse.append(km.inertia_)
        silhouette_scores.append(silhouette_score(reduced, km.labels_))

    k_sil = k_range[silhouette_scores.index(max(silhouette_scores))]
    k_knee = KneeLocator(k_range, sse, curve='convex', direction='decreasing').elbow or k_sil

    if categoria == "Palestina": return 4, "Manual - k=4"
    elif categoria == "Israel": return 6, "Manual - k=6"
    elif categoria == "Internacional": return 4, "Manual - k=4"
    elif categoria == "Internacional Medio Oriente": return 3, "Manual - k=3"
    return k_sil, "Default"

def generar_ngrama_cloud(df, columna_texto, columna_tema, categoria, ngram_range=(2, 2)):
    num_topics = df[columna_tema].nunique()
    fig, axes = plt.subplots(1, num_topics, figsize=(5*num_topics, 5), squeeze=False)
    for topic_num in range(num_topics):
        textos = df[df[columna_tema] == topic_num][columna_texto]
        vectorizer = TfidfVectorizer(ngram_range=ngram_range, stop_words='english', min_df=2)
        X = vectorizer.fit_transform(textos)
        freqs = dict(zip(vectorizer.get_feature_names_out(), X.sum(axis=0).A1))
        wordcloud = WordCloud(width=800, height=600, background_color='white', max_words=100, colormap='tab10').generate_from_frequencies(freqs)
        ax = axes[0, topic_num]
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.set_title(f"Tópico {topic_num}", fontsize=18)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(resultados_dir, f"nube_bigrama_{categoria}.png"))
    plt.show()

# === FUNCIONES MODULARES ===
def limpiar_y_guardar_embeddings(df, categorias):
    for categoria in categorias:
        print(f"\n💾 Generando embeddings para: {categoria}")
        df_cat = df[df["categoria"] == categoria].copy()
        df_cat["transcript_clean"] = df_cat["transcript"].apply(clean_text).apply(lemmatize_text)
        df_cat["embedding"] = df_cat["transcript_clean"].progress_apply(get_e5_embedding)
        df_cat.to_pickle(os.path.join(resultados_dir, f"narrativas_clusterizadas_{categoria}.pkl"))

def analizar_topicos_desde_pkl(categorias):
    resumen = []
    for categoria in categorias:
        print(f"\n📊 Analizando tópicos para: {categoria}")
        df_cat = pd.read_pickle(os.path.join(resultados_dir, f"narrativas_clusterizadas_{categoria}.pkl"))
        df_cat["transcript_filtered"] = df_cat["transcript_clean"].apply(filtrar_palabras)

        embeddings = np.vstack(df_cat["embedding"].values)
        reduced = UMAP(n_components=5, n_neighbors=15, random_state=42).fit_transform(embeddings)

        k, criterio = seleccionar_k_optimo(reduced, categoria)
        km = KMeans(n_clusters=k, random_state=42)
        df_cat["topic"] = km.fit_predict(reduced)

        plt.figure(figsize=(6,4))
        sns.countplot(data=df_cat, x='topic', palette="tab10")
        plt.title(f"Distribución de tópicos - {categoria}")
        plt.tight_layout()
        plt.savefig(os.path.join(resultados_dir, f"distribucion_topicos_{categoria}.png"))
        plt.close()

        guardar_tf_idf(df_cat, categoria)
        generar_ngrama_cloud(df_cat, "transcript_filtered", "topic", categoria)

        df_cat.to_pickle(os.path.join(resultados_dir, f"narrativas_clusterizadas_{categoria}_con_topicos.pkl"))

        fila = {"Categoría": categoria, "k_seleccionado": k, "Criterio": criterio, "Total_videos": len(df_cat)}
        for t, count in df_cat["topic"].value_counts().sort_index().items():
            fila[f"Tópico_{t}"] = count
        resumen.append(fila)

    df_resumen = pd.DataFrame(resumen)
    df_resumen.to_csv(os.path.join(resultados_dir, "resumen_topicos_por_categoria.csv"), index=False)
    return df_resumen

def guardar_tf_idf(df, categoria):
    vect = TfidfVectorizer(stop_words=list(custom_stopwords), max_features=50, ngram_range=(2,2))
    tfidf = vect.fit_transform(df["transcript_filtered"])
    tfidf_df = pd.DataFrame({
        "Bigrama": vect.get_feature_names_out(),
        "Importancia": tfidf.mean(axis=0).A1
    }).sort_values(by="Importancia", ascending=False)
    tfidf_df.to_csv(os.path.join(resultados_dir, f"tfidf_bigrama_{categoria}.csv"), index=False)

# === EJECUCIÓN PRINCIPAL ===
if __name__ == "__main__":
    df = pd.read_csv("noticias_youtube_filtrado_min_25.csv")
    categorias = ["Palestina", "Israel", "Internacional", "Internacional Medio Oriente"]

    # Paso 1: Solo una vez - generar embeddings
    limpiar_y_guardar_embeddings(df, categorias)

    # Paso 2: Análisis y visualización de tópicos
    analizar_topicos_desde_pkl(categorias)
