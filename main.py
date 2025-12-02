from fastapi import FastAPI
import logging

from app.api.routes import router as review_router

# Configure root logging once
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
)

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] %(levelname)s - %(name)s - %(message)s"
)

# Make sure root logger isn’t stuck on WARNING
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

logger = logging.getLogger(__name__)
logger.info("EA Review service starting up")

app = FastAPI(title="EA Review BE Service - MAF")

# Register routers
app.include_router(review_router)

logger.info("Routers registered and FastAPI app ready")
