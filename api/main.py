"""
FastAPI entrypoint — exposes the hallucination detection pipeline over HTTP.
"""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.pipeline import HallucinationDetectionPipeline
from app.schemas import CheckRequest, CheckResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Lifespan: load pipeline once at startup ───────────────────────────────────

pipeline: HallucinationDetectionPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    logger.info("Loading pipeline...")
    pipeline = HallucinationDetectionPipeline()
    logger.info("Pipeline ready.")
    yield
    logger.info("Shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Hallucination Detection API",
    description="Real-time LLM hallucination detection middleware",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "pipeline_loaded": pipeline is not None}


@app.post("/check", response_model=CheckResponse)
async def check(request: CheckRequest):
    """
    Verify an LLM-generated text for hallucinations.

    - Extracts factual claims from each sentence
    - Searches the web for evidence
    - Returns colour-coded verdict per sentence with sources
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready")

    try:
        # Run blocking pipeline in a thread pool — keeps the event loop free
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, pipeline.run, request)
        return result
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        raise HTTPException(status_code=500, detail=str(e))