"""长历时点列的后台作业生命周期。

作业在有界线程池里跑 :func:`model.hydrograph.run_hydrograph`，状态机为

    queued -> running -> completed
                       \\-> cancelled   （中途取消，只交完整=false 的元信息）
                       \\-> failed      （收敛失败等）

取消语义：取消信号在每两个点之间被检查；取消时 :class:`HydrographCancelled`
带出的半截点列**不会**作为结果返回，响应里只放 ``complete=false``、
``points`` 一律为空，已算点数只作为诊断计数。每份作业各自持有自己的
工况与推进临时量，彼此完全隔离。
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Deque

from .errors import ServiceError
from .hydrograph import HydrographCancelled, run_hydrograph


@dataclass
class _Job:
    id: str
    params: dict[str, Any]
    series: dict[str, Any]
    state: str = "queued"
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    points: list[dict[str, Any]] | None = None
    ponding: dict[str, Any] | None = None
    phase_switches: list[dict[str, Any]] | None = None
    progress_done: int = 0
    progress_total: int = 0
    error: dict[str, Any] | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)

    def snapshot(self, *, include_points: bool = True) -> dict[str, Any]:
        body: dict[str, Any] = {
            "job_id": self.id,
            "state": self.state,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "progress": {"done": self.progress_done, "total": self.progress_total},
            "params": self.params,
            "series": self.series,
        }
        if self.state == "completed":
            if include_points:
                body["points"] = self.points
                body["phase_switches"] = self.phase_switches
            body["ponding"] = self.ponding
            body["complete"] = True
        elif self.state == "cancelled":
            # 取消的作业永远不交半截点列
            body["complete"] = False
            body["points"] = []
            body["points_computed_before_cancel"] = self.progress_done
            body["ponding"] = self.ponding
        elif self.state == "failed":
            body["complete"] = False
            body["error"] = self.error
        return body


class JobManager:
    """进程内作业管理器。"""

    def __init__(self, workers: int = 4, retention: int = 500) -> None:
        self._pool = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="ga-hydrograph"
        )
        self._jobs: dict[str, _Job] = {}
        self._recent: Deque[str] = deque(maxlen=retention)
        self._lock = threading.Lock()
        # 默认推进器；测试可替换为确定性替身（如立刻抛 HydrographCancelled）
        self.runner: Callable[..., Any] = run_hydrograph

    def set_runner(self, runner: Callable[..., Any]) -> None:
        self.runner = runner

    def submit(
        self,
        params: dict[str, Any],
        series: dict[str, Any],
        *,
        job_id: str | None = None,
        runner: Callable[..., Any] | None = None,
    ) -> str:
        job = _Job(
            id=job_id or uuid.uuid4().hex,
            params=dict(params),
            series=dict(series),
        )
        self._pool.submit(self._run, job, runner or self.runner)
        with self._lock:
            self._jobs[job.id] = job
            self._recent.append(job.id)
        return job.id

    def _run(self, job: _Job, runner: Callable[..., Any]) -> None:
        job.started_at = time.time()
        job.state = "running"

        def on_progress(done: int, total: int) -> None:
            job.progress_done = done
            job.progress_total = total

        try:
            result = runner(
                job.params["Ks"],
                job.params["psi"],
                job.params["delta_theta"],
                job.series["duration"],
                n_points=job.series["n_points"],
                i=job.params.get("i"),
                already_ponded=job.series.get("already_ponded", False),
                abs_tol=job.params.get("abs_tol", 1e-10),
                rel_tol=job.params.get("rel_tol", 1e-12),
                max_iter=job.params.get("max_iter", 50),
                should_cancel=job.cancel_event.is_set,
                progress=on_progress,
            )
        except HydrographCancelled as exc:
            job.state = "cancelled"
            job.finished_at = time.time()
            job.progress_done = exc.last_index + 1
            job.points = None
            return
        except ServiceError as exc:
            job.state = "failed"
            job.finished_at = time.time()
            job.error = {"code": exc.code, "reason": exc.reason, "details": exc.details}
            return
        except Exception as exc:  # pragma: no cover - 防御性
            job.state = "failed"
            job.finished_at = time.time()
            job.error = {"code": "internal_error", "reason": str(exc)}
            return

        job.points = result.points
        job.ponding = result.ponding
        job.phase_switches = result.phase_switches
        job.progress_done = result.n_points
        job.progress_total = result.n_points
        job.state = "completed"
        job.finished_at = time.time()

    def get(self, job_id: str) -> _Job:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise ServiceError("job_not_found", f"作业 {job_id!r} 不存在",
                               status_code=404)
        return job

    def status(self, job_id: str, *, include_points: bool = True) -> dict[str, Any]:
        return self.get(job_id).snapshot(include_points=include_points)

    def cancel(self, job_id: str) -> dict[str, Any]:
        job = self.get(job_id)
        job.cancel_event.set()
        return {"job_id": job_id, "cancel_requested": True, "state": job.state}

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
