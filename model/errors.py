"""统一的错误结构与异常类型。

所有在校验或求解阶段被挡住的请求，都转成 ``{"error": {...}}`` 的 JSON 结构，
``error`` 中带机器可读的 ``code`` 与人能读懂的 ``reason``。
"""

from __future__ import annotations

from typing import Any


class ServiceError(Exception):
    """可直接回报给调用方的服务错误。

    Attributes:
        code: 机器可读错误码。
        reason: 失败原因（人话）。
        status_code: 建议的 HTTP 状态码。
        details: 附加上下文（如迭代次数、最终残差）。
    """

    status_code = 400

    def __init__(
        self,
        code: str,
        reason: str,
        status_code: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(reason)
        self.code = code
        self.reason = reason
        if status_code is not None:
            self.status_code = status_code
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        body: dict[str, Any] = {"code": self.code, "reason": self.reason}
        if self.details:
            body["details"] = self.details
        return body


class ValidationError(ServiceError):
    """参数在进入数值过程之前未通过校验。"""

    status_code = 400


class ConvergenceError(ServiceError):
    """隐式方程迭代到最大步数仍未把残差压进阈值。

    发生时绝不把没解出来的迭代初值/中间量当作结果吐出去。
    """

    status_code = 422
