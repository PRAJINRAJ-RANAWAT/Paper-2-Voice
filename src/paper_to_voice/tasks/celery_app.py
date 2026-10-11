"""
Celery configuration and application factory for Paper to Voice distributed task processing.
"""

import os
from celery import Celery

# Broker and backend configuration from environment variables
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")
TASK_TIME_LIMIT = int(os.getenv("CELERY_TASK_TIME_LIMIT", "120"))
TASK_SOFT_TIME_LIMIT = int(os.getenv("CELERY_TASK_SOFT_TIME_LIMIT", "90"))

# Initialize Celery app
celery_app = Celery(
    "paper_to_voice",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
)

# Celery app configuration
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=TASK_TIME_LIMIT,
    task_soft_time_limit=TASK_SOFT_TIME_LIMIT,
    task_acks_late=True,  # Supports safe redelivery if worker crashes
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,  # Fair dispatch among distributed workers
    task_default_queue="default",
    task_routes={
        "solve_substeps_task": {"queue": "research_steps"},
        "generate_dialog_task": {"queue": "dialog_tasks"},
    },
)

# Auto-discover tasks in the paper_to_voice package
celery_app.autodiscover_tasks(["src.paper_to_voice.tasks"])
