"""pytest 公共夹具：临时持久化目录 + Flask 测试客户端。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.app import create_app
from model.jobs import JobManager
from model.profiles import ProfileStore, seed_defaults  # noqa: E402


@pytest.fixture()
def tmp_profiles(tmp_path):
    store = ProfileStore(tmp_path / "profiles")
    seed_defaults(store)
    return store


@pytest.fixture()
def app(tmp_path):
    application = create_app(profiles_dir=str(tmp_path / "profiles"))
    application.config.update(TESTING=True)
    yield application
    application.extensions["ga_jobs"].shutdown()
    application.extensions["ga_calibration_jobs"].shutdown()


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def job_manager():
    mgr = JobManager(workers=4)
    yield mgr
    mgr.shutdown()


# 典型壤土（cm、h 单位制）
LOAM = dict(Ks=1.09, psi=11.01, delta_theta=0.434)
