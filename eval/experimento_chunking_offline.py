"""Experimento OFFLINE de chunking (antes/después) sin llamar a ninguna API.

Usa el mismo corpus (pdfs/), el mismo splitter y las mismas anotaciones de evaluacion_manual.py,
pero la recuperación es TF-IDF (coseno) en vez de embeddings de Gemini. Por eso NO reemplaza una corrida
de Ragas: sirve para comparar el efecto del chunking en context_recall / context_precision con un mismo
recuperador. Faithfulness y answer_relevancy dependen del LLM y no se miden aquí.
"""
import sys, re, unicodedata, json
from pathlib import Path
import numpy as np
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
sys.path.insert(0, str(Path(__file__).parent))
from evaluacion_manual import REF, norm, ap_precision

ROOT = Path(sys.argv[1])
# Q14 (CHAR vs VARCHAR) se excluye: "CHAR" no aparece en ningún PDF del corpus, así que no es respondible con él.
EXCLUIR = {"Q14"}
Q = {q["id"]: q["question"] for q in json.loads((ROOT/"eval/eval_questions.json").read_text(encoding="utf-8")) if q["id"] not in EXCLUIR}
pages = []
for pdf in sorted((ROOT/"pdfs").glob("*.pdf")):
    for n, pg in enumerate(PdfReader(str(pdf)).pages, 1):
        t = (pg.extract_text() or "").strip()
        if t: pages.append(t)

def run(size, overlap, k):
    sp = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap, separators=["\n\n","\n",".", " "])
    chunks = [c for t in pages for c in sp.split_text(t)]
    vec = TfidfVectorizer(preprocessor=norm, ngram_range=(1,2), sublinear_tf=True)
    M = vec.fit_transform(chunks)
    rec, prec = [], []
    for q, text in Q.items():
        sims = (M @ vec.transform([text]).T).toarray().ravel()
        top = [norm(chunks[i]) for i in np.argsort(-sims)[:k]]
        allt = " ".join(top)
        rec.append(np.mean([bool(re.search(p, allt)) for p in REF[q]]))
        prec.append(ap_precision([any(re.search(p, c) for p in REF[q]) for c in top]))
    return len(chunks), round(float(np.mean(prec)),3), round(float(np.mean(rec)),3), np.round(rec,2)

for cfg in [(500,50,5),(800,100,5),(800,100,4),(300,30,5)]:
    n, p, r, per = run(*cfg)
    print(f"chunk={cfg[0]}/{cfg[1]} top_k={cfg[2]}  fragmentos={n}  context_precision={p}  context_recall={r}")
