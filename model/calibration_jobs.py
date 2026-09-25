"""标定（反演）作业的后台生命周期。

与 :mod:`model.jobs` 的点列作业同构但各自独立账册：标定在有界线程池里跑
:func:`model.calibration.run_calibration`，状态机同样为

    queued -> running -> completed
                       \\-> cancelled   （中途取消，绝不交没收敛的半成品）
                       \\-> failed      （迭代不收敛、数据无约束等）

每份作业各自持有自己的观测序列、迭代轨迹与临时参数，全部只活在本作业
的栈帧与作业对象里，多份标定并发互不串账。取消信号在每个 LM 迭代步前
被检查；取消时迭代轨迹只作为诊断计数出现，**标定参数一律不外吐**。
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Deque

from .calibration import CalibrationCancelled, run_calibration
from .errors import ServiceError


@dataclass
class _CalJob:
    id: str
    spec: dict[str, Any]
    state: str = "queued"
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    progress_iterations: int = 0
    progress_sse: float | None = None
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    save_error: dict[str, Any] | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)

    def snapshot(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "job_id": self.id,
            "state": self.state,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "progress": {
                "iterations": self.progress_iterations,
                "sse": self.progress_sse,
            },
            "spec": self.spec,
        }
        if self.state == "completed":
            body["complete"] = True
            body["result"] = self.result
            if self.save_error is not None:
                body["save_error"] = self.save_error
        elif self.state == "cancelled":
            # 取消的作业永远不交半成品参数；轨迹只给诊断计数
            body["complete"] = False
            body["iterations_before_cancel"] = self.progress_iterations
        elif self.state == "failed":
            body["complete"] = False
            body["error"] = self.error
        return body


class CalibrationJobManager:
    """进程内标定作业管理器，与点列 JobManager 完全独立。"""

    def __init__(self, workers: int = 4, retention: int = 500) -> None:
        self._pool = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="ga-calibrate"
        )
        self._jobs: dict[str, _CalJob] = {}
        self._recent: Deque[str] = deque(maxlen=retention)
        self._lock = threading.Lock()
        self._app: Any = None
        # 默认推进器；测试可替换为确定性替身
        self.runner: Callable[..., Any] = run_calibration

    def bind_app(self, app: Any) -> None:
        """注入 Flask 应用，供作业完成后在 app context 内落工况档案。"""
        self._app = app

    def set_runner(self, runner: Callable[..., Any]) -> None:
        self.runner = runner

    def submit(self, spec: dict[str, Any], *,
               runner: Callable[..., Any] | None = None,
               job_id: str | None = None) -> str:
        spec = dict(spec)
        spec.setdefault("n_observations", len(spec.get("observations", ())))
        job = _CalJob(id=job_id or uuid.uuid4().hex, spec=spec)
        self._pool.submit(self._run, job, runner or self.runner)
        with self._lock:
            self._jobs[job.id] = job
            self._recent.append(job.id)
        return job.id

    def run_inline(self, spec: dict[str, Any],
                   runner: Callable[..., Any] | None = None) -> dict[str, Any]:
        """同步执行（HTTP /calibrate 用）：异常原样抛出由路由层转错误结构。"""
        run = runner or self.runner
        result = run(
            spec["observations"],
            i=spec.get("i"),
            fixed=spec.get("fixed"),
            initial=spec.get("initial"),
            max_iter=spec.get("max_iter", 200),
        )
        return result.to_dict()

    def _run(self, job: _CalJob, runner: Callable[..., Any]) -> None:
        job.started_at = time.time()
        job.state = "running"

        def on_progress(it: int, sse: float, params: dict[str, float]) -> None:
            job.progress_iterations = max(job.progress_iterations, it)
            job.progress_sse = sse

        try:
            result = runner(
                job.spec["observations"],
                i=job.spec.get("i"),
                fixed=job.spec.get("fixed"),
                initial=job.spec.get("initial"),
                max_iter=job.spec.get("max_iter", 200),
                should_cancel=job.cancel_event.is_set,
                on_progress=on_progress,
            )
        except CalibrationCancelled as exc:
            job.state = "cancelled"
            job.finished_at = time.time()
            job.progress_iterations = max(job.progress_iterations, exc.iterations)
            job.result = None
            return
        except ServiceError as exc:
            job.state = "failed"
            job.finished_at = time.time()
            job.error = {"code": exc.code, "reason": exc.reason,
                         "details": exc.details}
            return
        except Exception as exc:  # pragma: no cover - 防御性
            job.state = "failed"
            job.finished_at = time.time()
            job.error = {"code": "internal_error", "reason": str(exc)}
            return

        result_dict = result.to_dict()

        # 先落工况档案（在 app context 内），再把状态置为 completed：
        # 保证外部一旦观察到 completed，建档副作用已经发生，不存在竞态。
        save = job.spec.get("save_profile")
        save_error = None
        if save and self._app is not None:
            try:
                with self._app.app_context():
                    from .calibration_routes import save_profile_if_requested
                    save_profile_if_requested(job.spec, result_dict)
            except ServiceError as exc:
                save_error = {"code": exc.code, "reason": exc.reason}

        job.result = result_dict
        if save_error is not None:
            job.save_error = save_error
        job.progress_iterations = result.total_iterations
        job.progress_sse = result.sse
        job.finished_at = time.time()
        job.state = "completed"  # 最后一步发布：此前所有结果与副作用均已就绪

    def get(self, job_id: str) -> _CalJob:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise ServiceError("calibration_job_not_found",
                               f"标定作业 {job_id!r} 不存在", status_code=404)
        return job

    def status(self, job_id: str) -> dict[str, Any]:
        return self.get(job_id).snapshot()

    def cancel(self, job_id: str) -> dict[str, Any]:
        job = self.get(job_id)
        job.cancel_event.set()
        return {"job_id": job_id, "cancel_requested": True, "state": job.state}

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
