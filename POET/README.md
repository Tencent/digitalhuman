# From Poetry Parties to Poetry Critics
Code companion for:

> *From Poetry Parties to Poetry Critics: Benchmarking Classical Chinese Poetry Generation and Evaluation*

Two tracks:

- **Generation** — build Honglou-Poem, the 160-entry dataset, Digital Poetry Party + expert-persona ranking.
- **Poetry-Tree** — build trees, the 153-root tree JSON, train/test parquet, ranking along tree paths.


## Layout

```text
data/construct/honglou/                   build Honglou-Poem from 《Dream of the Red Chamber》
data/construct/poetry_tree/build_tree.py  build Poetry-Tree (needs private critiques; see Data)
data/honglou/honglou_poems.json           160-entry generation set (no expert critiques)
data/poetry_tree/                         download Poetry-Tree JSON + parquet here (see Data)
experiments/pipeline/                     generate poems and rank them
experiments/tree_rank/                    rank along Poetry-Tree paths
common/llm_client.py
.env.example
```

## Setup

```bash
cd classical-chinese-poetry-benchmark
python -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env
```


## Calling models

All LLM calls go through `common/llm_client.py` (`call_LLM`), which uses the official [OpenAI Python SDK](https://github.com/openai/openai-python) (`openai`) **Chat Completions** API (`client.chat.completions.create`). Any OpenAI-compatible HTTP endpoint works: set the base URL to that provider.

| Variable | Meaning |
|----------|---------|
| `POET_API_KEY` | API key (required). Falls back to `OPENAI_API_KEY`. |
| `POET_API_BASE` | Chat Completions base URL. Falls back to `OPENAI_BASE_URL`. Official OpenAI: `https://api.openai.com/v1`. For a gateway, use the URL in that provider’s docs. |
| `POET_TEMPERATURE` | Sampling temperature (default `1.0`). |
| `POET_RETRY_LIMIT` | Retries on request failure (default `5`). |
| `POET_LOG_DIR` | Directory for request logs (default `output/logs`). |
| `POET_DISABLE_MODEL_LOG` | Set to `1` to turn off request logs. |

`--model` / `--models` is the model id your endpoint expects (the Chat Completions `"model"` field). For several models, use a comma-separated list.

```bash
# .env
POET_API_KEY=sk-...
POET_API_BASE=https://api.openai.com/v1
```

```bash
python experiments/pipeline/generate_poems.py --models gpt-4o --workers 1
```

Defaults used in the generation + ranking pipeline:

| Role | Default models |
|------|----------------|
| Generators | `glm-4.6`, `kimi-k2`, `gemini-2.5-pro`, `o3`, `gpt-5.2`, `gemini-3`, `claude-4.5`, `hunyuan-turbos`, `deepseek-reasoner`, `v3.1-think`, `gpt-4o`, `grok-4` |
| Judges | `deepseek-reasoner`, `kimi-k2`, `gemini-2.5-pro`, `glm-4.6` |

### Pipeline flags (`experiments/pipeline`)

| Flag | Meaning |
|------|---------|
| `--tasks` | Prompt JSON (default: `data/honglou/honglou_poems.json`). |
| `--models` | Comma-separated generator ids. |
| `--output` | Where to write generated poems. |
| `--workers` | Parallel **tasks** (poems in one meeting still run in order). |
| `--seed` | Shuffle generator order; omit for a random order. |
| `--show-previous` / `--hide-previous` | Whether later generators see earlier poems in the same meeting (hidden by default). |
| `--poems` | Ranking input (generation JSON). |
| `--model` | Judge model id. |
| `--rank-mode` | `batch`: rank the whole group in one call (default). `separate`: review one poem at a time. |
| `--gold` | Optional expert-critique JSON for in-context learning. Not in this repo. |
| `--gold-samples` | Extra ICL critiques besides the background poem (default `4`, so n=5 with the background). |

`./run_pipeline.sh` wraps both scripts. Extra flags: `--poetry-models`, `--rank-model`, `--skip-poetry`, `--skip-rank`.

### Tree ranking flags (`experiments/tree_rank/rank_tree.py`)

| Flag | Meaning |
|------|---------|
| `--input` | Poetry-Tree JSON (default via `run_eval.sh`: `data/poetry_tree/poetry_tree.json`). |
| `--model` | Ranker model id. |
| `--rank-mode` | Same as above (`batch` / `separate`). |
| `--workers` | Parallel tree paths. |
| `--gold` / `--use-gold` | Optional expert-critique JSON for ICL. |

```bash
./run_eval.sh gpt-4o
```

### Tree construction flags (`data/construct/poetry_tree/build_tree.py`)

| Flag | Meaning |
|------|---------|
| `--source` | Honglou-Poem JSON **with** expert critiques. The public `honglou_poems.json` is not enough. |
| `--model` | Model used to extract merits and generate children. |
| `--max-depth` | Maximum tree depth (default `5`). |
| `--max-children` | Children sampled per node (default `3`). |
| `--workers` | Parallel poems. |
| `--poem-limit` | Debug: only the first N poems. |

## Data

**Honglou-Poem** (`data/honglou/honglou_poems.json`, 160 entries) is the default generation input:

```json
{
  "id": 1,
  "title": "...",
  "poet": "...",
  "author": "...",
  "background": "..."
}
```

`poet` is the poem text. This file has no expert critiques.

Rebuild from a novel you obtain yourself:

```bash
export PYTHONPATH=$PWD
cd data/construct/honglou
mkdir -p novel
python extract_from_novel.py --data-dir novel --model gpt-4o
```

`--data-dir` is a folder of `.txt` files; `--force` rebuilds cached extracts.

**`--gold` (optional, not in the repo).** Ranking can use in-context learning to imitate an expert (1 background poem + 4 random critiques, n=5). That JSON holds `title` / `poet` / `poet_analyse` and is copyrighted, so it is not shipped. Pass `--gold /path/to/your.json` only if you have it locally. Omit it and ranking still runs, without expert demonstrations.

**Poetry-Tree construction (`data/construct/poetry_tree/build_tree.py`) cannot be reproduced from the public files.** Building trees needs the full Honglou-Poem records with expert critiques (`poet_analyse`, etc.). The released `data/honglou/honglou_poems.json` omits those fields for copyright reasons, so the default `--source` is not sufficient. Download `poetry_tree.json` from Hugging Face (below) for ranking. For the complete critique-bearing data, contact the authors: [yizh6@mail2.sysu.edu.cn](mailto:yizh6@mail2.sysu.edu.cn).

**Poetry-Tree JSON and parquet** are hosted on Hugging Face (too large for GitHub): [Zihao1/Poet-4B_training_data](https://huggingface.co/datasets/Zihao1/Poet-4B_training_data)

| File | Role |
|------|------|
| `poetry_tree.json` | 153-root Poetry-Tree (no expert critiques) |
| `poet_tree_train.parquet` | ranking training set (verl; 7560 rows) |
| `poet_tree_test.parquet` | ranking test set (verl; 990 rows) |

```bash
pip install huggingface_hub
huggingface-cli download Zihao1/Poet-4B_training_data --repo-type dataset --local-dir data/poetry_tree
```

Scripts still read `data/poetry_tree/poetry_tree.json` (and the parquet files) after the download. Parquet is verl ranking format (`data_source=poetry_rank`); columns: `prompt`, `reward_model` (`gold_ranks`), `extra_info` (`group_id`, `root_id`, `root_title`, `num_poems`). Expert-critique few-shot text has been removed from the public parquet; poems and `gold_ranks` remain. For full ICL training prompts, contact the authors.

## Reproduce

**Generate and rank** — 12 models write independently on `data/honglou/honglou_poems.json` (previous poems hidden by default), then four judges rank in `batch` mode.

```bash
cd experiments/pipeline
./run_pipeline.sh --help
# ./run_pipeline.sh --gold /path/to/your_expert_critiques.json
```

**Rank along tree paths** — compare model order vs depth-order labels (Spearman and accuracy):

```bash
cd experiments/tree_rank
./run_eval.sh gpt-4o
```

## Poet-4B

Two options:

1. **Use the released model:** [huggingface.co/Zihao1/poet](https://huggingface.co/Zihao1/poet)
2. **Train your own:** download `poet_tree_train.parquet` / `poet_tree_test.parquet` from [Zihao1/Poet-4B_training_data](https://huggingface.co/datasets/Zihao1/Poet-4B_training_data) and run [verl](https://github.com/volcengine/verl) (official scripts; none are vendored here). Reward is ranking correlation vs `gold_ranks` (e.g. Spearman).