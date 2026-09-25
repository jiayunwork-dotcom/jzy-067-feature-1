"""土壤工况建档持久化。

工况按名字存成容器内持久化位置的 JSON 文件（配置见
:data:`model.config.PROFILES_DIR`），日后凭名字取回同名参数复算。
所有读写经过同一把进程内互斥锁，点列作业之间的临时量只活在
:mod:`model.hydrograph` 的局部栈帧里，与本模块互不串账。
"""

from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .errors import ServiceError, ValidationError
from .validation import validate_soil_params

_NAME_RE = re.compile(r"^[A-Za-z0-9_\-.一-鿿]{1,64}$")


@dataclass(frozen=True)
class SoilProfile:
    """一份土壤工况档案。"""

    name: str
    Ks: float
    psi: float
    delta_theta: float
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_name(name: str) -> str:
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise ValidationError(
            "invalid_name", "工况名字长度需在 1~64，只允许字母数字、中文、_ - ."
        )
    return name


class ProfileStore:
    """以 JSON 文件为后端的工况仓库。"""

    def __init__(self, directory: str | os.PathLike[str]) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, name: str) -> Path:
        return self.dir / f"{_safe_name(name)}.json"

    def save(self, profile: SoilProfile, *, overwrite: bool = True) -> None:
        path = self._path(profile.name)
        with self._lock:
            if path.exists() and not overwrite:
                raise ServiceError(
                    "profile_exists",
                    f"工况 {profile.name!r} 已存在",
                    status_code=409,
                )
            tmp = path.with_suffix(".json.tmp")
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(profile.to_dict(), fh, ensure_ascii=False, indent=2)
            os.replace(tmp, path)

    def get(self, name: str) -> SoilProfile:
        path = self._path(name)
        with self._lock:
            if not path.exists():
                raise ServiceError(
                    "profile_not_found", f"工况 {name!r} 不存在", status_code=404
                )
            data = json.loads(path.read_text(encoding="utf-8"))
        return SoilProfile(
            name=data["name"],
            Ks=float(data["Ks"]),
            psi=float(data["psi"]),
            delta_theta=float(data["delta_theta"]),
            description=data.get("description", ""),
        )

    def delete(self, name: str) -> None:
        path = self._path(name)
        with self._lock:
            if not path.exists():
                raise ServiceError(
                    "profile_not_found", f"工况 {name!r} 不存在", status_code=404
                )
            path.unlink()

    def list_names(self) -> list[str]:
        with self._lock:
            return sorted(p.stem for p in self.dir.glob("*.json"))

    def create(
        self,
        name: str,
        Ks: Any,
        psi: Any,
        delta_theta: Any,
        description: str = "",
        *,
        overwrite: bool = True,
    ) -> SoilProfile:
        _safe_name(name)
        Ks_f, psi_f, dtheta_f = validate_soil_params(Ks, psi, delta_theta)
        profile = SoilProfile(
            name=name, Ks=Ks_f, psi=psi_f, delta_theta=dtheta_f,
            description=str(description),
        )
        self.save(profile, overwrite=overwrite)
        return profile


# 预置壤土算例：Ks 单位 cm/h，psi 单位 cm。
# 一小时积水入渗 F 满足 F - A ln(1+F/A) = Ks*1，因吸力项明显大于 Ks*t=1.09。
LOAM = SoilProfile(
    name="loam",
    Ks=1.09,
    psi=11.01,
    delta_theta=0.434,
    description=(
        "预置壤土算例（Mein-Larson 典型参数）：Ks=1.09 cm/h，"
        "psi=11.01 cm，delta_theta=0.434。积水 1 小时累积入渗因湿润锋"
        "吸力项明显大于 Ks*t=1.09 cm，可用于拉起服务后的核对。"
    ),
)


def seed_defaults(store: ProfileStore) -> None:
    """写入预置工况（已存在且参数一致则不动）。"""
    try:
        existing = store.get(LOAM.name)
    except ServiceError as exc:
        if exc.code != "profile_not_found":
            raise
        store.save(LOAM, overwrite=False)
        return
    if (
        existing.Ks != LOAM.Ks
        or existing.psi != LOAM.psi
        or existing.delta_theta != LOAM.delta_theta
    ):
        store.save(LOAM, overwrite=True)
