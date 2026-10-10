# VulDuet

## Create the Conda environment

```bash
conda env create -f environment.yml
conda activate vulduet
cp src/config/config.local.example.yml src/config/config.local.yml
```

Add your own API key to `src/config/config.local.yml`. This local file is
excluded from Git and must never be committed.

Faiss is installed only by Conda. Do not install `faiss` or `faiss-cpu` with
pip in this environment.

The historical path uses Vul-RAG-style multi-view BM25 retrieval. Function
purpose, security-relevant behavior, and vulnerable source code are searched
independently. Their ranks are fused before the unchanged LLM reranker,
historical reasoner, evidence aggregator, and arbiter run.

## One-sample smoke test

```bash
python -m src.main_primevul \
  --profile deepseek_v3_2 \
  --rule-top-k 5 \
  --knowledge-view-top-k 20 \
  --knowledge-retrieval-top-k 5 \
  --knowledge-reasoning-top-k 3 \
  --minimum-scenario-match 0.2 \
  --minimum-confidence 0.5 \
  --workers 1 \
  --reasoning-workers 3 \
  --limit 1
```

## Rebuild the enhanced knowledge base

```bash
python -m src.build_historical_knowledge \
  output/step1/historical_knowledge_documents.json \
  /path/to/primevul_train_clean.json \
  output/knowledge/historical_knowledge.json
```

The builder validates `source_id`, CVE, and commit mappings before writing.

## Evaluate results

```bash
python -m src.evaluation.evaluation_dual_path RESULT.json
```

Every run saves both the complete `*_result.json` file and a smaller
`*_compact_result.json` companion. To compact an existing result manually:

```bash
python -m src.evaluation.compact_results FULL_RESULT.json
```

