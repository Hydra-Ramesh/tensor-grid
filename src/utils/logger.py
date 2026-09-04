import sys
from loguru import logger

# Remove default logger
logger.remove()

# Add a JSON structured logger to stdout
logger.add(
    sys.stdout,
    format="{time:YYYY-MM-DDTHH:mm:ss.SSSZ} | {level} | {module}:{function}:{line} | {message}",
    serialize=True,  # This forces JSON output for enterprise ingest (Datadog/ELK)
    level="INFO",
)

# Export the configured logger
__all__ = ["logger"]
