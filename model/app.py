"""Flask 应用装配：错误处理、持久化目录、作业管理器与路由注册。"""

from __future__ import annotations

import json
import logging

from flask import Flask, jsonify
from werkzeug.exceptions import HTTPException

from . import config
from .caljobs import CalibrationManager
from .errors import ServiceError
from .jobs import JobManager
from .profiles import ProfileStore, seed_defaults
from .routes import api
from .routes_calibrate import cal_api


def create_app(*, profiles_dir: str | None = None) -> Flask:
    app = Flask(__name__)
    app.json.sort_keys = False

    store = ProfileStore(profiles_dir or config.PROFILES_DIR)
    seed_defaults(store)
    mgr = JobManager(workers=config.JOB_WORKERS, retention=config.JOB_RETENTION)
    cal_mgr = CalibrationManager(
        store, workers=config.CAL_JOB_WORKERS, retention=config.JOB_RETENTION
    )

    app.extensions["ga_profiles"] = store
    app.extensions["ga_jobs"] = mgr
    app.extensions["ga_caljobs"] = cal_mgr
    app.register_blueprint(api)
    app.register_blueprint(cal_api)

    @app.errorhandler(ServiceError)
    def _on_service_error(exc: ServiceError):
        return jsonify({"error": exc.to_dict()}), exc.status_code

    @app.errorhandler(HTTPException)
    def _on_http_error(exc: HTTPException):
        return jsonify({
            "error": {"code": exc.name.lower().replace(" ", "_"),
                      "reason": exc.description or exc.name},
        }), exc.code

    @app.errorhandler(json.JSONDecodeError)
    def _on_bad_json(exc: json.JSONDecodeError):
        return jsonify({"error": {"code": "invalid_body",
                                  "reason": f"JSON 解析失败: {exc.msg}"}}), 400

    @app.errorhandler(Exception)
    def _on_unexpected(exc: Exception):  # pragma: no cover - 防御性
        logging.getLogger("ga").exception("unhandled error: %s", exc)
        return jsonify({"error": {"code": "internal_error", "reason": str(exc)}}), 500

    return app
