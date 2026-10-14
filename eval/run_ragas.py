"""Evaluación del RAG de TutorIA con Ragas.

Uso (desde la raíz del repo, con GROQ_API_KEY y GEMINI_API_KEY en el entorno o en .env):

    python eval/run_ragas.py --label baseline
    CHUNK_SIZE=800 CHUNK_OVERLAP=100 RAG_TOP_K=4 INDEX_DIR=index_v2 python eval/run_ragas.py --label v2
    python eval/run_ragas.py --compare baseline v2

Genera en eval/results/:  <label>_detalle.csv (por pregunta) y <label>_resumen.json.
Métricas Ragas: faithfulness, answer_relevancy, context_precision, context_recall.
Métrica extra (control de alucinaciones): % de preguntas fuera del corpus donde el
sistema se abstiene correctamente (responde el mensaje NO_INFO o declina amablemente).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

RESULTS = ROOT / "eval" / "results"
QUESTIONS = ROOT / "eval" / "eval_questions.json"
METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
ABSTENTION_MARKERS = ["no encontré información", "solo puedo ayudar", "solo puedo ayudarte"]


def run_pipeline(questions, pause=2.0):
    """Ejecuta el RAG real (rag.py) sobre cada pregunta, sin historial."""
    from rag import RAGEngine

    engine = RAGEngine()
    rows = []
    for i, item in enumerate(questions, 1):
        result = engine.answer(item["question"])
        rows.append(
            {
                "id": item["id"],
                "user_input": item["question"],
                "response": result["respuesta"],
                "retrieved_contexts": [f["text"] for f in result["fuentes"]],
                "reference": item["ground_truth"],
                "in_corpus": item["in_corpus"],
            }
        )
        print(f"[{i}/{len(questions)}] {item['id']} listo")
        time.sleep(pause)  # evita límites de la capa gratuita
    return rows


def evaluate_with_ragas(rows):
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain_groq import ChatGroq
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
    from ragas.run_config import RunConfig

    judge = LangchainLLMWrapper(
        ChatGroq(model=os.getenv("RAGAS_JUDGE_MODEL", "openai/gpt-oss-120b"), temperature=0)
    )
    embeddings = LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001", google_api_key=os.environ["GEMINI_API_KEY"]
        )
    )
    dataset = EvaluationDataset.from_list(
        [
            {
                "user_input": r["user_input"],
                "response": r["response"],
                "retrieved_contexts": r["retrieved_contexts"],
                "reference": r["reference"],
            }
            for r in rows
        ]
    )
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge,
        embeddings=embeddings,
        run_config=RunConfig(max_workers=2, timeout=240, max_retries=6),
    )
    return result.to_pandas()


def summarize(df, rows):
    summary = {"n_preguntas": len(df)}
    in_mask = df["in_corpus"].astype(bool)
    for name in METRIC_NAMES:
        if name in df:
            summary[name] = round(float(df[name].mean()), 3)
            summary[f"{name}__solo_en_corpus"] = round(float(df.loc[in_mask, name].mean()), 3)
    outs = [r for r in rows if not r["in_corpus"]]
    if outs:
        ok = sum(any(m in r["response"].lower() for m in ABSTENTION_MARKERS) for r in outs)
        summary["abstencion_correcta_fuera_de_corpus"] = f"{ok}/{len(outs)}"
    summary["config"] = {
        "CHUNK_SIZE": os.getenv("CHUNK_SIZE", "500"),
        "CHUNK_OVERLAP": os.getenv("CHUNK_OVERLAP", "50"),
        "RAG_TOP_K": os.getenv("RAG_TOP_K", "5"),
        "EMBEDDING_MODEL": os.getenv("EMBEDDING_MODEL", "gemini-embedding-001"),
        "GROQ_MODEL": os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
    }
    return summary


def compare(a, b):
    sa = json.loads((RESULTS / f"{a}_resumen.json").read_text(encoding="utf-8"))
    sb = json.loads((RESULTS / f"{b}_resumen.json").read_text(encoding="utf-8"))
    print(f"{'métrica':<28}{a:>12}{b:>12}{'Δ':>10}")
    for name in METRIC_NAMES:
        va, vb = sa.get(name), sb.get(name)
        if va is not None and vb is not None:
            print(f"{name:<28}{va:>12}{vb:>12}{vb - va:>+10.3f}")
    print("abstención fuera de corpus:", sa.get("abstencion_correcta_fuera_de_corpus"), "->",
          sb.get("abstencion_correcta_fuera_de_corpus"))
    print("config", a, sa["config"])
    print("config", b, sb["config"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = parser.parse_args()
    if args.compare:
        return compare(*args.compare)

    RESULTS.mkdir(parents=True, exist_ok=True)
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    rows = run_pipeline(questions)
    df = evaluate_with_ragas(rows)
    df.insert(0, "id", [r["id"] for r in rows])
    df["in_corpus"] = [r["in_corpus"] for r in rows]
    df["response"] = [r["response"] for r in rows]
    df.to_csv(RESULTS / f"{args.label}_detalle.csv", index=False, encoding="utf-8-sig")
    summary = summarize(df, rows)
    (RESULTS / f"{args.label}_resumen.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
