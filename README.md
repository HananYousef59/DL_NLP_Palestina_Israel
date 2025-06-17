# Modelos de PLN y Deep Learning para el Análisis de Tópicos y Emociones en YouTube sobre el Conflicto Palestino-Israelí

**Hanan Yousef**  
*Universidad EAN*  
*Facultad de Ingeniería*  
*Maestría en Ciencias de Datos*

---

## Descripción del proyecto

Este repositorio contiene el código, los scripts y la documentación asociados al trabajo de grado orientado al análisis automatizado de narrativas y emociones en videos y comentarios de YouTube sobre el conflicto palestino-israelí. El periodo de estudio comprende del *7 de octubre de 2023* al *27 de marzo de 2025*.

Se aplicaron:

- **Técnicas no supervisadas**: embeddings semánticos (`e5-large-v2`) y agrupamiento con K-Means.
- **Técnicas supervisadas**: fine-tuning de modelos basados en Transformers para clasificación emocional.

El objetivo fue identificar diferencias discursivas según el origen del canal (Palestina, Israel, Internacional, Internacional Medio Oriente) y entrenar modelos capaces de capturar emociones contextualizadas como *ira, esperanza, tristeza, resistencia* y *sarcasmo*.

---

## Tecnologías utilizadas

- **Lenguaje**: Python 3.10  
- **Transformers**: `deberta-v3-base`, `roberta-base`, `electra-base-discriminator`, `xlnet-base-cased`  
- **Embeddings**: `intfloat/e5-large-v2`  
- **Clusterización**: KMeans, UMAP  
- **Visualización**: Matplotlib, Seaborn, WordCloud  
- **Métricas de evaluación**: F1-score, matriz de confusión, cohesión semántica  

---

## Descarga de modelos fine-tuneados

*Debido al tamaño de los archivos, los modelos entrenados no están incluidos directamente en este repositorio.*  
Puedes descargarlos en el siguiente enlace:

** **

---

## Resumen del estudio

El estudio explora narrativas de canales con distintas orientaciones sobre el conflicto palestino-israelí. Se recolectaron y categorizaron videos mediante scripts automáticos, se aplicaron embeddings `e5-large-v2` y K-Means para identificar tópicos, y posteriormente se extrajeron comentarios representativos para entrenar modelos clasificadores de emociones mediante *fine-tuning*.

Las emociones fueron etiquetadas manualmente en cinco categorías personalizadas: *Ira, Tristeza, Esperanza, Resistencia* y *Sarcasmo*. Los mejores resultados se lograron con **DeBERTa** y **RoBERTa**, alcanzando F1-macro de *0.89* y *0.87*, respectivamente.

Este trabajo contribuye al análisis computacional del discurso, particularmente en entornos digitales de alta sensibilidad geopolítica.

---

## Palabras clave

*Palestina-Israel, YouTube, análisis de discurso, clasificación de emociones, transformers, embedding semántico, topic modeling, PLN.*

---

## Resumen de resultados

La implementación del pipeline propuesto permitió recolectar y procesar datos de YouTube vinculados al conflicto palestino-israelí, asegurando representatividad discursiva entre octubre de 2023 y marzo de 2025. A través de *embeddings* semánticos (`e5-large-v2`), reducción dimensional con UMAP y agrupamiento mediante K-Means, se identificaron estructuras narrativas diferenciadas según el origen geopolítico de los canales analizados.

El entrenamiento supervisado de modelos Transformer para clasificación emocional evidenció la presencia de emociones fuertemente contextualizadas —como *resistencia, ira* y *esperanza*— en los comentarios de usuarios. El modelo **DeBERTa** obtuvo el mejor desempeño (F1-macro: *0.89*), superando a **RoBERTa**, **ELECTRA** y **XLNet**, gracias a su capacidad para capturar relaciones semánticas complejas en textos breves y emocionalmente ambiguos.

Estos resultados confirman la pertinencia de integrar técnicas de modelado temático y afectivo en el análisis computacional del discurso en conflictos geopolíticos digitales, y destacan el potencial de los modelos Transformer como herramientas analíticas en contextos informativos de alta sensibilidad.
