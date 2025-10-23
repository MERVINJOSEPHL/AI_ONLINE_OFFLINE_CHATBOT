import logging
import sys
import httpx
import json

# --- 1. Logging Setup ---
def setup_logging(name='rag_app'):
    """Sets up a robust, custom logger with try-catch."""
    try:
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            ch = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
            ch.setFormatter(formatter)
            logger.addHandler(ch)
            logger.info("Application logging initialized.")
        return logger
    except Exception as e:
        print(f"FATAL ERROR in setup_logging: {e}")
        sys.exit(1)

logger = setup_logging()

