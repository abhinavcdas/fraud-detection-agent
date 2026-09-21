# Engineering & Architecture Rules for Fraud Detection Agent

These architectural rules MUST be strictly followed across all development phases.

---

## 1. Modularity & Pluggable Operators (Strategy Pattern)
* **Never hardcode backends** directly in business logic. All components must adhere to the contracts defined in [`core/interfaces.py`](file:///d:/Personal/fraud-detection-agent/core/interfaces.py).
* Every operational component must have a corresponding factory and support runtime swapping via environment configuration:
  - **Storage**: Must support `postgres`, `sqlite`, and `memory`.
  - **Streaming**: Must support `kafka` and `memory`.
  - **LLM Provider**: Must support `groq` and `mock`.
  - **Model Scorer**: Must support `xgboost` and `heuristic`.
* Adding a new provider must only require adding a class implementing the interface and updating the factory.

---

## 2. Fault Tolerance & Failure Isolation
* **One failing component must never bring down the system or stop the event stream**:
  - **LLM/API Outages**: If Groq or an external model API experiences rate limits (429), timeouts, or 503 errors, wrap calls with `core.resilience.retry_external_call` and error boundaries. On failure, emit a `degraded_fallback` audit log entry, alert via structured logger, and proceed. Never drop the transaction.
  - **Poison Pills**: Malformed or unparseable messages in the stream must be routed to the Dead-Letter Queue (`core.resilience.global_dlq`), logged as errors, and the consumer loop must continue without crashing.
  - **Database Resilience**: Operations should handle transient connection failures gracefully.

---

## 3. Structured Observability & Tracing
* **No `print()` statements** in production modules.
* Use `core.logger.get_logger(__name__)` everywhere.
* Always bind contextual transaction metadata (`bind_tx_context(logger, transaction_id, customer_id)`) to enable end-to-end distributed tracing across streaming, scoring, and agent investigations.
* Logs must write to both human-readable console and rotating structured JSON files (`logs/fraud_pipeline.log`).

---

## 4. Asynchronous & High-Throughput Design
* The streaming ingestion and consumer pipeline must be non-blocking (`async`/`await`).
* Heavy synchronous calls must be delegated to background worker threads / executors (`loop.run_in_executor`).
* Support batch processing and async queues.

---

## 5. Documentation & Test Driven Discipline
* Every phase must maintain and update `README.md`, `walkthrough.md`, and relevant `.md` files.
* Accompany every new module with automated unit and resilience tests (`tests/`).
