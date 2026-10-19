# RAG over the GDPR: Baseline Retrieval vs. Cross-Encoder Re-Ranking

This repo compares two retrieval pipelines over the same chunked document, the EU General Data Protection Regulation (Regulation (EU) 2016/679, official 88-page English PDF):

1. **Baseline**: a Bi-Encoder (`sentence-transformers/all-MiniLM-L6-v2`) retrieves the Top 10 chunks by cosine similarity (FAISS). The Top 3 go to the LLM and produce **Answer 1**.
2. **Re-ranking**: the same Top 10 are re-scored by a Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`). The new Top 3 go to the LLM with the identical prompt and produce **Answer 2**.

The LLM is Gemini (`gemini-3.5-flash`) at temperature 0.

## Files

| Path | Contents |
|---|---|
| `rag_rerank.ipynb` | The full, commented notebook, with outputs saved |
| `data/gdpr.pdf` | Source document (EU Publications Office) |
| `llm_cache.json` | Cached LLM responses, so the notebook can be re-run without using API quota |
| `requirements.txt` | Pinned dependencies |

## Notebook sections

1. Setup & configuration
2. PDF loading and cleaning (running headers and footnotes removed)
3. Structure-aware chunking. Each chunk belongs to exactly one recital or article, has at most 150 words, and is labelled, e.g. `Article 17 – Right to erasure`. This gives 527 chunks.
4. Bi-Encoder embeddings + FAISS `IndexFlatIP`
5. LLM client, the shared prompt template and the response cache
6. The two pipelines
7. Sanity-check query
8. Failure mode analysis:
   - rank-shift summary for 13 stress-test queries (keyword traps, exceptions, negations, buried chunks)
   - Answer 1 vs Answer 2 for 6 of them
   - a counter-example where re-ranking hurts
   - a limitation where the gold chunk is never retrieved
9. Latency profiling, including the Cross-Encoder run over the whole corpus

## How to run

```bash
python3 -m venv .venv
.venv/bin/pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt

echo "GEMINI_API_KEY=your_key_here" > .env     # free key: https://aistudio.google.com/apikey
.venv/bin/jupyter notebook rag_rerank.ipynb     # or open it in VS Code and select the .venv kernel
```

Everything runs on CPU. The two MiniLM models (about 90 MB each) download automatically on first use.

LLM answers are cached in `llm_cache.json`, keyed by model + prompt. A re-run reuses them, so it needs no API calls. Delete the file to generate fresh answers. The Gemini free tier allows only about 20 requests per model per day.
