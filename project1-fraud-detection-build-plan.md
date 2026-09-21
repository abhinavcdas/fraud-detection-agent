# Project 1: Real-Time Fraud Detection with LLM Investigation Agent
## In-Depth Build Plan for Antigravity

This is a task-by-task implementation guide, expanded from the roadmap, written so you can hand chunks of it directly to Antigravity as work orders. Each phase has: goals, repo structure, concrete tasks, acceptance criteria, and suggested prompts for the agent.

---

## 0. Before You Start

### 0.1 Repo structure (set this up first — Antigravity works best with clear scaffolding)

```
fraud-detection-agent/
├── docker-compose.yml
├── .env.example
├── README.md
├── data/
│   ├── raw/                    # Kaggle CSV lands here
│   └── processed/
├── producer/
│   └── stream_producer.py      # Replays dataset into Kafka
├── consumer/
│   ├── feature_engineering.py
│   └── consumer.py             # Kafka consumer -> Postgres
├── db/
│   ├── schema.sql
│   └── init.py
├── model/
│   ├── train.py
│   ├── evaluate.py
│   ├── shap_explain.py
│   └── registry/               # MLflow local store, gitignored
├── agent/
│   ├── tools.py                # get_customer_history, check_known_patterns, get_merchant_risk_score
│   ├── agent_loop.py           # LangGraph or hand-rolled loop
│   ├── guardrails.py           # fact-checking / faithfulness validation
│   └── prompts.py
├── eval/
│   ├── eval_model.py
│   ├── eval_agent.py
│   └── eval_set.csv
├── governance/
│   ├── model_card.md
│   ├── fairness_audit.md
│   └── audit_log_schema.sql
├── api/
│   └── main.py                 # FastAPI service
├── dashboard/
│   └── app.py                  # Streamlit (or /frontend for React)
├── monitoring/
│   └── drift_report.py         # Evidently AI
├── tests/
│   └── ...
└── .github/workflows/
    └── ci.yml
```

Create this skeleton (empty files + folders) as your very first Antigravity task — it gives the agent a map to work against instead of inventing structure ad hoc each session.

### 0.2 Local environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install xgboost lightgbm scikit-learn pandas numpy shap mlflow \
            kafka-python fastapi uvicorn psycopg2-binary sqlalchemy \
            streamlit evidently langgraph anthropic python-dotenv pytest
```

`docker-compose.yml` should stand up: Redpanda (or Kafka), Postgres, and MLflow's tracking server (or just run `mlflow ui` locally to start).

### 0.3 Get the data

Download the **Kaggle Credit Card Fraud Detection** dataset (`creditcard.csv`) into `data/raw/`. It has `Time`, `Amount`, `V1`–`V28` (PCA-anonymized features), and `Class` (1 = fraud). Note upfront in your README that real identity/location fields aren't present here — you'll simulate a `customer_id` and geo field for the agent-tooling phase (say so explicitly in the model card later; this is exactly the kind of "know the gap" honesty the roadmap flags as a seniority signal).

---

## Phase 1 — Data Pipeline (Week 1)

**Goal:** a working stream from CSV → Kafka → feature engineering → Postgres.

### Tasks

1. **Docker Compose for Redpanda + Postgres**
   - Redpanda console on `localhost:8080` for visually inspecting topics — very useful for debugging with Antigravity, since you can screenshot/describe topic state instead of guessing.
   - Postgres with a persisted volume.

2. **`db/schema.sql`**
   - `raw_transactions` (all original columns + injected `customer_id`, `merchant_id`, `lat`, `lon`)
   - `engineered_features` (transaction_id FK, rolling stats)
   - `audit_log` (see Phase 5 — create the table now even if unused yet)

3. **`producer/stream_producer.py`**
   - Reads `creditcard.csv` row by row (sorted by `Time`)
   - Injects a synthetic `customer_id` (e.g., hash of a few features so the same "customer" recurs — needed for rolling velocity features to mean anything) and synthetic merchant/geo fields
   - Pushes each row as a JSON message to a `transactions` Kafka topic, with a small `time.sleep()` or scaled delay to simulate real time

4. **`consumer/feature_engineering.py`**
   - Rolling features per `customer_id`, computed from Postgres history at consume-time:
     - transaction count in last 5 / 60 minutes
     - deviation of current `Amount` from customer's rolling mean/std
     - time-since-last-transaction
     - geographic jump distance from last known location (haversine)
   - Write both raw and engineered rows to Postgres

5. **`consumer/consumer.py`**
   - Kafka consumer loop tying producer output → feature engineering → DB write

### Acceptance criteria
- Running `producer` then `consumer` populates `engineered_features` with sane, non-null values.
- You can query Postgres and see rolling features change plausibly as more of a synthetic customer's history accumulates.

### Suggested Antigravity prompt
> "Using `db/schema.sql` as the target schema, write `consumer/feature_engineering.py`. It should accept a single transaction dict, query Postgres for that customer's prior transactions, and return a dict of the five rolling features described in [paste feature list]. Include unit tests with a fake in-memory customer history."

---

## Phase 2 — Fraud Model (Week 1–2)

**Goal:** a tuned, tracked XGBoost/LightGBM model registered in MLflow.

### Tasks

1. **`model/train.py`**
   - Load from `engineered_features` (not raw CSV — dogfood your own pipeline)
   - Train/test split with **stratification** (fraud is <1%, so a naive split can zero out the minority class in test)
   - Compare three imbalance strategies in the same script, logged as separate MLflow runs: class-weighting (`scale_pos_weight`), SMOTE, and focal loss (if using LightGBM with a custom objective, or `imbalanced-learn` + XGBoost)
   - Cross-validated hyperparameter search (Optuna or `GridSearchCV`) for the winning strategy

2. **MLflow logging**
   - Log params, metrics (precision, recall, F1, ROC-AUC, PR-AUC — **PR-AUC matters more than ROC-AUC here** given the class imbalance), and the model artifact for every run
   - Register the best run's model as `fraud-xgb-v1` in the MLflow Model Registry, stage it `Staging`

### Acceptance criteria
- MLflow UI shows ≥6 runs (3 strategies × at least 2 hyperparameter configs each) with comparable metrics tables.
- Best model has recall you can defend (fraud detection should bias toward recall) and you can articulate the precision tradeoff in one paragraph for the model card later.

### Suggested Antigravity prompt
> "Write `model/train.py`. It loads features from Postgres table `engineered_features`, does a stratified 80/20 split, and trains three XGBoost variants (class-weighted, SMOTE-resampled, focal-loss via LightGBM) each with a small Optuna search over max_depth/learning_rate/n_estimators. Log every run to MLflow with metrics precision/recall/F1/ROC-AUC/PR-AUC. At the end, register the run with the best PR-AUC to the MLflow registry as `fraud-xgb-v1`."

---

## Phase 3 — LLM Investigation Agent (Week 2–3)

**Goal:** an agent that, given a flagged transaction, calls tools, reasons over the results, and emits a structured, fact-checked report.

### Tasks

1. **`agent/tools.py`** — three tools, each a plain Python function with a clear docstring/schema (this doubles as the tool spec you hand to LangGraph or the raw tool-calling loop):
   - `get_customer_history(customer_id)` → last N transactions + rolling stats from Postgres
   - `check_known_patterns(transaction)` → rule-based checks (e.g., amount just under a common reporting threshold, transaction at an odd hour, rapid-fire count) against a small hardcoded pattern list
   - `get_merchant_risk_score(merchant_id)` → a synthetic lookup table you seed (merchant category → base risk) since you don't have real merchant risk data

2. **`agent/agent_loop.py`**
   - If using LangGraph: define a graph with a reasoning node and tool nodes, looping until the model emits a final structured answer.
   - If hand-rolling: a `while` loop that (a) sends the transaction + system prompt to Claude, (b) parses tool-call requests, (c) executes them, (d) feeds results back, (e) repeats until a final JSON answer is returned or a max-turn limit hits.
   - **Force structured output**: system prompt should require exactly `{risk_level, evidence: [...], recommendation, cited_facts: [...]}` — no free text outside the JSON.

3. **`agent/guardrails.py`**
   - For each `cited_facts` entry the agent claims, check it actually appears (string-match or a light semantic check) in the tool outputs collected that turn.
   - If a claim doesn't check out, flag the report as `needs_review` rather than silently passing it through — this is the guardrail the roadmap is pointing at, and it's also a good metric to report in Phase 4.

### Acceptance criteria
- Feeding a known-fraud transaction from the eval set produces a JSON report where every cited fact is traceable to a tool call output.
- The agent handles a "boring" (non-fraud-looking) flagged transaction without inventing dramatic evidence.

### Suggested Antigravity prompt
> "Implement `agent/agent_loop.py` as a hand-rolled tool-calling loop against the Anthropic API. Tools are defined in `agent/tools.py` — convert their docstrings into the Anthropic tool-use JSON schema. System prompt: [paste]. Cap at 4 tool-call turns, then force a final structured JSON answer matching this schema: [paste schema]. Write 3 test transactions (clear fraud, borderline, clear non-fraud) and print the resulting reports."

---

## Phase 4 — Evaluation Harness (Week 3)

**Goal:** two separate scorecards — one for the ML model, one for the agent.

### Tasks

1. **`eval/eval_model.py`**
   - Score the registered model on a held-out slice: precision, recall, F1, ROC-AUC, PR-AUC, and a confusion matrix
   - Add SHAP (`model/shap_explain.py`): global summary plot + at least 3 individual-prediction force plots, saved as images for the dashboard/report

2. **`eval/eval_agent.py` + `eval/eval_set.csv`**
   - Build a small (30–50 row) hand-labeled slice: for each, note whether it's a "should flag" case and what a reasonable human investigator would look for
   - Run the agent over all of them, then manually (or with a second LLM-as-judge pass, clearly labeled as such — don't let it silently replace human review) score:
     - **Faithfulness rate**: % of `cited_facts` that pass the guardrail check
     - **Recommendation reasonableness**: did it recommend escalate/clear/monitor sensibly given the evidence?
   - Report both numbers plainly, including failure examples — a portfolio piece that shows *where* the agent fails is more credible than one claiming it never does

### Acceptance criteria
- A single markdown or notebook output summarizing: model metrics table, 3 SHAP plots, agent faithfulness rate, 2–3 annotated example reports (one good, one flawed).

---

## Phase 5 — Model Risk & Fairness Layer (Week 3–4)
*(This is the section that differentiates the project — don't compress it.)*

### Tasks

1. **`governance/model_card.md`** — follow Google's Model Card template sections:
   - Intended use / out-of-scope uses (be explicit: this is a portfolio demonstration on public anonymized data, not a production-validated model)
   - Training data description (source, size, class balance, known limitations — note the PCA-anonymized features limit true explainability of *what* V1–V28 represent)
   - Performance, overall and by segment (see fairness audit below)
   - Monitoring plan (tie to Phase 6's drift monitor)

2. **Fairness audit**
   - Since this dataset has no demographic fields, segment by proxies you *do* have: `Amount` bands, synthetic merchant category, time-of-day. Explicitly note in the write-up that these are **not** protected characteristics and that this is a methodology demo, not a real fair-lending analysis (save the real protected-class fairness work for Project 2, which has more appropriate data).
   - Compute flag-rate disparity ratios across segments; report the largest gaps.

3. **`governance/audit_log_schema.sql` + wiring it up**
   - Every model prediction and every agent decision writes a row: timestamp, input hash, model version, prediction/score, agent report (if any), guardrail status.
   - This should be live by Phase 6 so the dashboard can query it.

### Acceptance criteria
- `model_card.md` reads like something you could hand to a manager, not a README.
- Audit log table has real rows after a test run through the full pipeline.

---

## Phase 6 — Deployment & Monitoring (Week 4)

### Tasks

1. **`api/main.py`** (FastAPI)
   - `POST /score` — takes a transaction, returns model score + (if above threshold) triggers the agent and returns its report
   - `GET /transactions/flagged` — recent flagged transactions for the dashboard
   - `GET /audit-log/{id}` — reconstruct a past decision

2. **Dockerize** the API; add it as a service in `docker-compose.yml`.

3. **`monitoring/drift_report.py`** (Evidently AI) — compare a recent window of scored transactions against the training distribution; schedule it (cron, or a GitHub Action on a schedule) and store reports for the dashboard to link to.

4. **`dashboard/app.py`** (Streamlit to start) — flagged transactions table, agent report viewer, SHAP plot viewer, drift report link.

5. **`.github/workflows/ci.yml`**
   - On push: run `pytest`, run `eval/eval_model.py` against a fixed small fixture and fail if metrics regress past a threshold, lint.
   - On merge to `main`: build + push Docker image, deploy (Render/Fly.io/Railway — pick one with a generous free tier for a Postgres + API + dashboard trio).

### Acceptance criteria
- End-to-end: push a transaction through the producer → see it scored, possibly flagged and investigated, and visible on the deployed dashboard within a few seconds.

---

## Working With Antigravity Across Phases — Practical Tips

- **One phase = one task list, one PR.** Antigravity does better with a bounded scope per session; use this doc's phase boundaries as your PR boundaries.
- **Point it at real files, not descriptions.** When asking it to write the consumer, give it the actual `schema.sql` and an example Kafka message, not a paraphrase.
- **Ask it to write tests alongside code, not after.** For the guardrail and feature-engineering logic especially, tests double as your own spec-check.
- **Have it write the model card and fairness audit *from* the eval outputs**, not from vibes — feed it the actual metrics tables and SHAP summaries so it's summarizing real numbers, and review its wording carefully since this section is where credibility is won or lost.
- **Re-run the eval harness after any model or agent change** and diff the numbers — treat regressions the same way you'd treat a failing test.

---

## Milestone Checklist

- [ ] Repo skeleton + Docker Compose up
- [ ] Kaggle data landing in `data/raw/`
- [ ] Producer → Kafka → Consumer → Postgres working end to end
- [ ] 3 imbalance strategies trained and logged in MLflow; best model registered
- [ ] Agent tools implemented and unit-tested
- [ ] Agent loop producing structured, guardrail-checked reports
- [ ] Model eval report (precision/recall/F1/ROC-AUC/PR-AUC + SHAP)
- [ ] Agent eval report (faithfulness rate + annotated examples)
- [ ] Model card written
- [ ] Fairness audit written (with proxy-field caveat stated explicitly)
- [ ] Audit log live and queryable
- [ ] FastAPI service + Docker
- [ ] Dashboard live
- [ ] Drift monitoring wired up
- [ ] CI/CD pipeline green on a fresh clone

When this checklist is done, you have Project 1's resume bullets fully earned rather than aspirational — and a clean base to start Project 2 in parallel with applying.
