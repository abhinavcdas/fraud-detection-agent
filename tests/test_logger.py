"""Unit tests for production structured logging."""

from pathlib import Path
from core.logger import get_logger, bind_tx_context, LOG_FILE

def test_logger_instance():
    log = get_logger("test_module")
    assert log is not None
    log.info("Test log message from test_logger_instance")

def test_contextual_binding():
    log = get_logger("test_tx_module")
    tx_log = bind_tx_context(log, transaction_id="TX_12345", customer_id="CUST_001")
    assert tx_log is not None
    tx_log.info("Test transaction log with context binding")

def test_log_file_created():
    assert LOG_FILE.parent.exists()

def test_pan_masking():
    from core.logger import PAN_REGEX, mask_pan
    raw_text = "Processing payment with card 4111119912345678 at checkout"
    masked = PAN_REGEX.sub(mask_pan, raw_text)
    assert "411111******5678" in masked
    assert "4111119912345678" not in masked
