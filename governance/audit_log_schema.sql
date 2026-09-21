-- Model Risk Management - Immutable Audit Trail Schema
-- Reconstructs past model scores and agent decisions for compliance verification.

CREATE TABLE IF NOT EXISTS audit_log (
    audit_id SERIAL PRIMARY KEY,
    transaction_id VARCHAR(64) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    input_hash VARCHAR(64) NOT NULL,
    model_version VARCHAR(64) NOT NULL,
    fraud_score DOUBLE PRECISION NOT NULL,
    is_flagged BOOLEAN NOT NULL,
    agent_decision VARCHAR(32), -- 'APPROVE', 'ESCALATE', 'DECLINE', 'MONITOR'
    agent_report JSONB,
    guardrail_status VARCHAR(32), -- 'PASSED', 'NEEDS_REVIEW', 'SKIPPED'
    faithfulness_score DOUBLE PRECISION,
    latency_ms DOUBLE PRECISION,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_audit_log_tx ON audit_log (transaction_id);
CREATE INDEX IF NOT EXISTS idx_audit_log_timestamp ON audit_log (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_audit_log_flagged ON audit_log (is_flagged) WHERE is_flagged = TRUE;
