"""
Task status tracking and idempotency control for distributed Celery tasks.
"""

import json
import logging
import os
import time
from enum import Enum
from typing import Any, Dict, Optional, Tuple
from pydantic import BaseModel, Field

try:
    import redis
except ImportError:
    redis = None

logger = logging.getLogger("paper_to_voice.tasks.tracking")


class TaskStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


class TaskRecord(BaseModel):
    task_id: str
    workflow_id: str = ""
    stage: str = ""
    step_index: int = 0
    status: TaskStatus = TaskStatus.PENDING
    error: Optional[str] = None
    result_metadata: Optional[Dict[str, Any]] = None
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)


class TaskTracker:
    """
    Manages task lifecycle states and idempotency.
    Uses Redis as primary backing store, with automatic in-memory fallback.
    Never stores large audio blobs or PDFs in Redis; only stores lightweight metadata.
    """

    def __init__(self, redis_url: Optional[str] = None):
        self.redis_url = redis_url or os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")
        self._redis_client: Optional[Any] = None
        self._local_cache: Dict[str, Dict[str, Any]] = {}
        self._idempotency_results: Dict[str, Dict[str, Any]] = {}
        self._init_redis()

    def _init_redis(self) -> None:
        if redis is None:
            logger.info("Redis library not available; using in-memory TaskTracker.")
            return

        try:
            client = redis.Redis.from_url(
                self.redis_url,
                socket_timeout=2.0,
                socket_connect_timeout=2.0,
                decode_responses=True,
            )
            # Test connectivity
            client.ping()
            self._redis_client = client
            logger.info("Connected TaskTracker to Redis at %s", self.redis_url)
        except Exception as exc:
            logger.info("TaskTracker could not connect to Redis (%s); using in-memory storage fallback.", exc)
            self._redis_client = None

    def set_task_status(
        self,
        task_id: str,
        status: TaskStatus,
        workflow_id: str = "",
        stage: str = "",
        step_index: int = 0,
        error: Optional[str] = None,
        result_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record or update task lifecycle status."""
        record = TaskRecord(
            task_id=task_id,
            workflow_id=workflow_id,
            stage=stage,
            step_index=step_index,
            status=status,
            error=error,
            result_metadata=result_metadata,
            updated_at=time.time(),
        )
        data = record.model_dump()

        if self._redis_client:
            try:
                key = f"task:status:{task_id}"
                self._redis_client.set(key, json.dumps(data), ex=86400)
                return
            except Exception as exc:
                logger.warning("Redis write failed for task %s (%s); falling back to memory", task_id, exc)

        self._local_cache[task_id] = data

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve the current recorded state of a task."""
        if self._redis_client:
            try:
                key = f"task:status:{task_id}"
                raw = self._redis_client.get(key)
                if raw:
                    return json.loads(raw)
            except Exception as exc:
                logger.warning("Redis read failed for task %s (%s); checking memory", task_id, exc)

        return self._local_cache.get(task_id)

    def is_idempotent_complete(self, task_id: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Check if a task with the given stable task_id has already succeeded.
        Returns: (is_complete, cached_result)
        """
        if self._redis_client:
            try:
                key = f"task:result:{task_id}"
                raw = self._redis_client.get(key)
                if raw:
                    logger.info("Idempotency hit in Redis for task %s", task_id)
                    return True, json.loads(raw)
            except Exception as exc:
                logger.debug("Redis idempotency read failed for task %s: %s", task_id, exc)

        if task_id in self._idempotency_results:
            logger.info("Idempotency hit in memory for task %s", task_id)
            return True, self._idempotency_results[task_id]

        return False, None

    def store_idempotent_result(self, task_id: str, result_data: Dict[str, Any], ttl: int = 86400) -> None:
        """
        Store result metadata for idempotent retrieval upon redelivery.
        Only stores JSON-serializable structured results (no raw file bytes).
        """
        if self._redis_client:
            try:
                key = f"task:result:{task_id}"
                self._redis_client.set(key, json.dumps(result_data), ex=ttl)
            except Exception as exc:
                logger.warning("Redis idempotency write failed for task %s: %s", task_id, exc)

        self._idempotency_results[task_id] = result_data


# Global default tracker singleton
tracker = TaskTracker()
