"""
Gunicorn configuration for production deployment.
"""

import multiprocessing
import os

# ── Server socket ────────────────────────────────────────────────────────
bind = f"{os.getenv('API_HOST', '0.0.0.0')}:{os.getenv('API_PORT', '8000')}"

# ── Worker processes ─────────────────────────────────────────────────────
# For ML workloads, fewer workers is better — each loads the model into memory.
# Default: 1 worker.  Override via API_WORKERS env var.
workers = int(os.getenv("API_WORKERS", "1"))
worker_class = "uvicorn.workers.UvicornWorker"

# ── Timeouts ─────────────────────────────────────────────────────────────
timeout = 120          # seconds before a worker is killed
graceful_timeout = 30  # seconds to finish in-flight requests on shutdown
keepalive = 5

# ── Preload ──────────────────────────────────────────────────────────────
# Preloading shares the model across workers via copy-on-write memory.
preload_app = True

# ── Logging ──────────────────────────────────────────────────────────────
loglevel = os.getenv("LOG_LEVEL", "info")
accesslog = "-"
errorlog = "-"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# ── Process naming ───────────────────────────────────────────────────────
proc_name = "openclip-api"
