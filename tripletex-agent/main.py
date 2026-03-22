"""FastAPI endpoint for the Tripletex AI Accounting Agent."""

import logging
import logging.handlers
import os
import time
import uuid

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from agent import solve_task

# ── Logging setup ─────────────────────────────────────────────────
# File log: full detail, no truncation, rotated at 50MB, keep 5 backups
LOG_DIR = os.environ.get("LOG_DIR", "/tmp/tripletex-agent-logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FORMAT = "%(asctime)s %(levelname)-5s [%(name)s] %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Root logger — captures everything from all modules
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)

# Console handler — INFO level, concise
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
root_logger.addHandler(console_handler)

# File handler — DEBUG level, full detail, 50MB rotation
file_handler = logging.handlers.RotatingFileHandler(
    os.path.join(LOG_DIR, "agent.log"),
    maxBytes=50 * 1024 * 1024,  # 50MB per file
    backupCount=5,
    encoding="utf-8",
)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
root_logger.addHandler(file_handler)

# Quiet down noisy libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

logger = logging.getLogger("main")

logger.info("=" * 80)
logger.info("Tripletex AI Agent starting — logs at %s", LOG_DIR)
logger.info("=" * 80)

# ── FastAPI app ───────────────────────────────────────────────────
app = FastAPI(title="Tripletex AI Agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(">> %s %s", request.method, request.url.path)
    response = await call_next(request)
    logger.info("<< %s %s -> %d", request.method, request.url.path, response.status_code)
    return response

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/solve")
@app.post("/")
async def solve(request: Request):
    task_id = uuid.uuid4().hex[:8]
    start = time.time()
    body = await request.json()

    prompt = body.get("prompt", "")
    files = body.get("files", [])
    creds = body.get("tripletex_credentials", {})
    base_url = creds.get("base_url", "")
    session_token = creds.get("session_token", "")

    logger.info("=" * 80)
    logger.info("[%s] NEW TASK", task_id)
    logger.info("[%s] Prompt (%d chars):\n%s", task_id, len(prompt), prompt)
    logger.info("[%s] Files: %d, Base URL: %s", task_id, len(files), base_url)
    for f in files:
        logger.info("[%s]   File: %s (%s, %d bytes base64)",
                    task_id, f.get("filename"), f.get("mime_type"), len(f.get("content_base64", "")))
    logger.info("=" * 80)

    if not base_url or not session_token:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "Missing tripletex_credentials"},
        )

    if not OPENROUTER_API_KEY:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": "OPENROUTER_API_KEY not configured"},
        )

    try:
        result = solve_task(
            prompt=prompt,
            files=files,
            base_url=base_url,
            session_token=session_token,
            openrouter_api_key=OPENROUTER_API_KEY,
            task_id=task_id,
        )
        elapsed = time.time() - start
        logger.info("[%s] Task completed in %.1fs, iterations: %s", task_id, elapsed, result.get("iterations"))
        return {"status": "completed"}
    except Exception as e:
        elapsed = time.time() - start
        logger.error("[%s] Task FAILED after %.1fs: %s", task_id, elapsed, str(e), exc_info=True)
        # Always return 200 with completed — competition requires it
        return {"status": "completed"}
