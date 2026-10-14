# TutorIA — Tutor académico con RAG (Avance 2)

Chatbot web (Flask) que responde preguntas de **Bases de Datos / MySQL** usando **únicamente** los PDF de la carpeta `pdfs/`.
Cada respuesta cita documento y página, y el chat muestra los fragmentos que la respaldan.

- **URL pública:** `https://<PEGA-AQUÍ-LA-URL-DE-RENDER>.onrender.com`
- **Repositorio:** `https://github.com/<usuario>/<repo>`

> El sistema conserva el control de la base de conocimientos y del índice: los PDF no se entregan a servicios de
> entrenamiento. Al LLM solo se le envían los fragmentos recuperados para cada consulta, nunca el corpus completo.

## Flujo RAG implementado

```
PDFs (pdfs/) ──pypdf──► texto por página
      │
      ▼
Chunking: RecursiveCharacterTextSplitter (500 caracteres, solape 50, separadores \n\n, \n, ".", " ")
      │  metadatos por fragmento: id, source (PDF), page
      ▼
Embeddings: gemini-embedding-001 (API de Gemini), 768 dimensiones, vectores normalizados
      │  taskType RETRIEVAL_DOCUMENT (fragmentos) / RETRIEVAL_QUERY (preguntas)
      ▼
Índice vectorial: embeddings.npy (matriz numpy) + chunks.json (texto + metadatos), en index_data/
      │
      ▼  pregunta → embedding → similitud coseno (producto punto) → top_k = 5
Prompt aumentado: system prompt + few-shot + <contexto><fragmento fuente pagina>…</contexto> + <pregunta>
      │
      ▼
LLM en Groq (openai/gpt-oss-120b, temperature 0) ──► respuesta con citas ──► interfaz Flask
```

### Decisiones técnicas y justificación

| Etapa | Decisión | Justificación |
|---|---|---|
| Selección de documentos | 3 PDF de apuntes de MySQL: *Fundamentos de MySQL*, *Creación y Búsqueda en MySQL*, *Programación en MySQL* | Cubren el dominio del tutor (Bases de Datos) de lo básico a lo avanzado y son texto digital extraíble. |
| Ingesta | `pypdf`, página por página | Conserva el número de página para poder citar. Solo se usan PDF porque es el formato del corpus; DOCX/TXT/HTML no hacen falta. |
| Chunking | 500 caracteres, solape 50, corte recursivo por párrafo → línea → punto → espacio | Fragmentos cortos = una idea por fragmento (mejor precisión de recuperación); el solape evita cortar una definición a la mitad; cortar por párrafo respeta la estructura de los apuntes. |
| Embeddings | `gemini-embedding-001` a 768 dim. | Multilingüe (el corpus está en español), se consume por API, así el servidor no carga ningún modelo y cabe en los 512 MB del plan gratuito de Render. |
| Base vectorial | Matriz numpy + JSON en disco | Con ~tens de fragmentos no hace falta un motor vectorial; ChromaDB superaba el límite de memoria de Render. La búsqueda sigue siendo vectorial (coseno). Metadatos `source` y `page` para citar. |
| Recuperación | Similitud coseno, `top_k = 5` | Suficiente contexto sin saturar el prompt. Preguntas muy cortas se combinan con la pregunta previa para mantener el hilo. |
| Generación | System prompt con reglas, few-shot de formato y delimitadores XML (`<reglas>`, `<contexto>`, `<pregunta>`) | Reutiliza el diseño del Avance 1. Regla explícita: si la respuesta no está en el contexto, responder el mensaje fijo de "no encontré información". |

Archivos: `rag.py` (pipeline), `prompts.py` (system prompt, few-shot, delimitadores, formato de salida), `app.py` (Flask), `build_index.py`.

## Evaluación con Ragas

- Conjunto de evaluación: `eval/eval_questions.json` (16 preguntas con ground truth; 12 dentro del corpus y 4 fuera del corpus/dominio para medir control de alucinaciones).
- Script: `eval/run_ragas.py` — métricas `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall` + tasa de abstención correcta fuera del corpus.

```bash
pip install -r requirements-eval.txt
python eval/run_ragas.py --label baseline
# iteración de mejora: cambia un parámetro y vuelve a evaluar
CHUNK_SIZE=800 CHUNK_OVERLAP=100 INDEX_DIR=index_v2 python eval/run_ragas.py --label v2
python eval/run_ragas.py --compare baseline v2
```

Los resultados quedan en `eval/results/` (CSV por pregunta y JSON resumen). Tabla y análisis: ver el PDF de la entrega.

## Ejecutar en local

```bash
python -m venv venv
source venv/bin/activate        # En Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # En Windows: copy .env.example .env
# edita .env: GROQ_API_KEY (console.groq.com) y GEMINI_API_KEY (aistudio.google.com/apikey)
python build_index.py           # indexa los PDFs (usa la API de Gemini)
python app.py                   # abre http://127.0.0.1:5000
```

## Configuración (variables de entorno)

| Variable | Para qué sirve | Por defecto |
|---|---|---|
| `GROQ_API_KEY` | Key de Groq (obligatoria) | — |
| `GEMINI_API_KEY` | Key de Gemini para embeddings (obligatoria) | — |
| `GROQ_MODEL` | Modelo de Groq | `openai/gpt-oss-120b` |
| `TUTOR_MATERIA` | Materia en la interfaz y en el prompt | `Bases de Datos` |
| `RAG_TOP_K` | Fragmentos recuperados por pregunta | `5` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | Parámetros de chunking | `500` / `50` |
| `INDEX_DIR` | Carpeta del índice | `index_data` |

Las claves **nunca** van en el repositorio: `.env` está en `.gitignore` y en Render se pegan en *Environment*.

## Desplegar en Render

1. Sube el proyecto a GitHub (con `pdfs/` incluida).
2. En [render.com](https://render.com): **New → Blueprint** y elige el repositorio (usa `render.yaml`). Alternativa manual: **New → Web Service**
   - Build Command: `pip install -r requirements.txt && python build_index.py`
   - Start Command: `gunicorn app:app --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:$PORT`
   - Instance type: Free
3. En **Environment** agrega `GROQ_API_KEY`, `GEMINI_API_KEY` y `PYTHON_VERSION` = `3.11.9`.
4. Espera el deploy. La URL pública queda como `https://<tu-servicio>.onrender.com`.

> El plan gratuito de Render duerme el servicio tras ~15 min sin uso; la primera visita tarda 30–60 s en despertar.

## Estructura

```
├── app.py  rag.py  prompts.py  build_index.py
├── eval/            # eval_questions.json, run_ragas.py, results/
├── pdfs/            # corpus
├── templates/  static/
├── requirements.txt  requirements-eval.txt  render.yaml  .env.example
```

## Autores
Jose Luis Patiño & Jordi Alexis Madrid | Desarrollo de Aplicaciones con IA - Facultad de matemáticas e ingenierías
