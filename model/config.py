"""运行配置：持久化位置等可由环境变量覆盖。"""

from __future__ import annotations

import os
from pathlib import Path


def _default_profiles_dir() -> Path:
    # 容器内持久化位置；允许通过环境变量改到挂载卷。
    env = os.environ.get("GA_PROFILES_DIR")
    if env:
        return Path(env)
    return Path(os.environ.get("DATA_DIR", "/data/profiles"))


PROFILES_DIR = Path(_default_profiles_dir())

# 后台点列作业的线程数
JOB_WORKERS = int(os.environ.get("GA_JOB_WORKERS", "4"))
# 后台标定作业的线程数
CAL_JOB_WORKERS = int(os.environ.get("GA_CAL_JOB_WORKERS", "2"))
# 作业结果在内存中保留的数量上限（防止长跑服务无限堆积）
JOB_RETENTION = int(os.environ.get("GA_JOB_RETENTION", "500"))
