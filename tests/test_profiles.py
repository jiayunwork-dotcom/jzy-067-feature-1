"""工况建档持久化与预置壤土算例。"""

from __future__ import annotations

import pytest

from model.errors import ServiceError, ValidationError
from model.profiles import LOAM, ProfileStore, seed_defaults
from model.solver import solve_cumulative


def test_seed_loam_and_fetch(tmp_profiles):
    p = tmp_profiles.get("loam")
    assert p.Ks == LOAM.Ks
    assert p.psi == LOAM.psi
    assert p.delta_theta == LOAM.delta_theta


def test_loam_one_hour_F_visibly_above_Ks_t(tmp_profiles):
    p = tmp_profiles.get("loam")
    sol = solve_cumulative(p.Ks, p.psi, p.delta_theta, 1.0)
    # 吸力项作用：一小时累积入渗明显大于 Ks*t=1.09 cm
    assert sol.F > 1.5 * p.Ks
    assert abs(sol.residual) <= sol.tolerance


def test_create_get_delete(tmp_profiles):
    tmp_profiles.create("sand", Ks=8.25, psi=4.95, delta_theta=0.417,
                        description="砂土")
    p = tmp_profiles.get("sand")
    assert p.Ks == 8.25 and p.description == "砂土"
    assert "sand" in tmp_profiles.list_names()
    tmp_profiles.delete("sand")
    with pytest.raises(ServiceError) as exc:
        tmp_profiles.get("sand")
    assert exc.value.code == "profile_not_found"


def test_no_overwrite_without_flag(tmp_profiles):
    tmp_profiles.create("clay", Ks=0.2, psi=31.63, delta_theta=0.476)
    with pytest.raises(ServiceError) as exc:
        tmp_profiles.create("clay", Ks=0.3, psi=31.63, delta_theta=0.476,
                            overwrite=False)
    assert exc.value.code == "profile_exists"


def test_invalid_profile_params_rejected(tmp_profiles):
    with pytest.raises(ValidationError):
        tmp_profiles.create("bad", Ks=-1.0, psi=10.0, delta_theta=0.4)
    with pytest.raises(ValidationError):
        tmp_profiles.create("bad2", Ks=1.0, psi=10.0, delta_theta=1.5)


def test_unsafe_names_rejected(tmp_profiles):
    for name in ("../etc/passwd", "a/b", "x" * 65, ""):
        with pytest.raises(ValidationError):
            tmp_profiles.create(name, Ks=1.0, psi=10.0, delta_theta=0.4)


def test_persistence_across_store_instances(tmp_path):
    d = tmp_path / "persist"
    s1 = ProfileStore(d)
    s1.create("loess", Ks=1.5, psi=8.0, delta_theta=0.35)
    s2 = ProfileStore(d)  # 模拟服务重启后重新挂载同一目录
    assert s2.get("loess").Ks == 1.5


def test_seed_idempotent(tmp_profiles):
    seed_defaults(tmp_profiles)
    seed_defaults(tmp_profiles)
    assert tmp_profiles.get("loam").Ks == LOAM.Ks
