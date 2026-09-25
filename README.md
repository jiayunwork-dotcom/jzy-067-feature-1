# Green–Ampt 入渗核算服务

把 Green–Ampt 湿润锋入渗的集中参数解算封装成 HTTP 后端：给定土壤参数
（饱和导水率 `Ks`、湿润锋基质吸力 `psi`、锋前后含水量差 `delta_theta`）
与历时，回报累积入渗量 `F`、瞬时入渗率 `f`、隐式方程残差，以及给定降雨
强度下的积水时刻判定；也支持把一串实测 `(时刻, 累积入渗量)` 反过来标定
土壤参数，并老实交代这次标定信不信得过。纯 JSON API，无网页界面。

## 模型

积水情形下（`A = psi·delta_theta`）：

```
F − A·ln(1 + F/A) = Ks·t          （隐式，牛顿迭代 + 二分括号兜底求解）
f = Ks·(1 + A/F)                   （F→0 时封顶到约定初值 1e6）
```

对数项是**减号**：真解恒有 `F > Ks·t`（t>0），测试专门盯这条。

降雨强度 `i` 下的积水判定（Mein–Larson）：

- `i <= Ks`：入渗能力始终高于供水，**永不积水**，全程自由入渗 `F = i·t`；
- `i > Ks`：自由段 `F = i·t` 直到能力降到供水强度，
  `Fp = Ks·A/(i−Ks)`，`tp = Fp/i`；之后以等效积水历时
  `te = [Fp − A·ln(1+Fp/A)]/Ks` 为偏移接积水隐式。

单位自洽即可（如 Ks、i 用 cm/h，psi 用 cm，F 用 cm，t 用 h）。

## 参数反演（标定）

吃进一串实测 `(t, F)`，在参数空间最小化各点模型值与实测值之差的平方和
（SSE）。被反演的是 `Ks` 与组合量 `A = psi·delta_theta`——正问题只通过
乘积 `A` 接触 `psi`/`delta_theta`，求解时以 `psi=A、delta_theta=1` 复用
既有正向解算，不另写方程。

- 目标函数：SSE = Σ(F_model − F_obs)²；
- 迭代：对数参数空间的 Levenberg–Marquardt（阻尼正规方程 + 信赖域拒绝），
  启发式多起点兜底；收敛判据为步长与目标下降都走不动，且停点在参数域
  内部。压不进判据、撞边界、跑飞或被取消一律报错，**绝不交半路参数**；
- 可辨识性（最关键）：只给**积水段**时，(Ks, A) 沿残差脊可同时挪动，
  结果判成 `combination_only`——只回报 SVD 强方向上的可辨识组合与弱
  方向，顶层 `fitted.Ks/A` 给 `null`，脊线上的点仅作 `ridge_point`
  诊断。要分别定参：固定其一（`fixed.Ks`/`fixed.A`），或给降雨强度
  `i` 且观测覆盖自由段与积水后段（积水时刻 tp 参与约束，判成
  `individual`）。全部落在自由段 F=i·t 的观测对土壤参数无约束，直接报错；
- 回报含 `sse`、`rmse`/`nrmse` 与拟合质量、逐点残差、迭代步数、函数
  评估次数与逐步迭代轨迹，供复核；
- 观测入箱校验（点数 <3、时刻不严格递增、F 为负、非有限值）在迭代前
  挡下；F 非单调不在此拦，交给 SSE 诚实暴露成一个很差的拟合。

## 运行

```bash
docker build -t green-ampt .          # 构建期自动跑 pytest，不过则构建失败
docker run -p 8000:8000 -v ga_data:/data green-ampt
curl localhost:8000/health            # 内含预置壤土自检
```

本地开发：`pip install -r requirements.txt pytest && python -m pytest && python wsgi.py`

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 健康检查 + 预置壤土一小时自检 |
| POST | `/infiltrate` | 单点核算：`{Ks, psi, delta_theta, t, [i], [already_ponded], [abs_tol], [rel_tol], [max_iter]}` 或 `{"profile": 名字, ...}` |
| POST | `/ponding` | 积水判定：`{Ks, psi, delta_theta, i}`，回 `will_pond` 与 `tp` |
| POST | `/series` | 长历时点列作业：`{..., duration, n_points, i 或 already_ponded:true}`，202 回 `job_id` |
| GET | `/series/<job_id>` | 作业状态/完整点列 |
| POST | `/series/<job_id>/cancel` | 取消；取消后只回 `complete:false`，**绝不交半截点列** |
| POST | `/calibrate` | 同步反演标定：`{observations:[{t,F}...], [i], [fixed], [initial]/[initial_profile], [max_iter], [save_profile]}` |
| POST | `/calibrations` | 标定后台作业（观测多、迭代长），202 回 `job_id` |
| GET | `/calibrations/<job_id>` | 标定作业状态/完整结果（含残差、轨迹、可辨识性） |
| POST | `/calibrations/<job_id>/cancel` | 取消标定；取消后**绝不交没收敛的半成品参数** |
| GET/PUT/DELETE | `/profiles[/<name>]` | 工况建档（预置 `loam`），持久化在 `DATA_DIR`（默认 `/data/profiles`） |

标定的可辨识性结果：

- `identifiability.status = "individual"`：`fitted.Ks`、`fitted.A` 可用，
  可凭 `save_profile`（补一个 `delta_theta` 或 `psi` 把 A 拆开）落成命名
  工况，与手工建档走同一套存取；已有工况也能用 `initial_profile` 当初值；
- `combination_only`：`fitted.Ks/A` 为 `null`，只给
  `identifiability.identifiable_combination` 与 `weak_direction`，不能建档。

错误统一为 `{"error": {"code", "reason", "details?"}}`：参数不合法 400、
工况/作业不存在 404、组合不可辨却要建档 409、迭代不收敛/数据无约束 422
（**不会**把没解出来的迭代值当结果）。

## 代码结构

```
model/
  validation.py         参数校验（迭代之前挡下坏输入）
  solver.py             隐式 F 求解（牛顿 + 括号兜底，残差阈值内才返回）
  infiltration.py       入渗率、积水时刻判定、自由/积水分段
  hydrograph.py         历时点列分段推进（可取消）
  jobs.py               点列后台作业生命周期
  calibration.py        反演标定：观测校验、LM 最小二乘、SVD 可辨识性
  calibration_jobs.py   标定后台作业（独立账册、可取消、取消不交半成品）
  calibration_routes.py 标定 HTTP 路由（含结果落工况档案）
  profiles.py           工况建档持久化（JSON 文件 + 预置壤土）
  routes.py             正向 HTTP 路由（只做入参与序列化）
  app.py                Flask 装配
tests/                  残差达标、符号陷阱、单调走向、积水判定、取消语义、
                        并发隔离、正反演还原、简并识别、掺噪连续、病态报错
```

## 关于 Δθ 的趋势（与常见直觉相反，按数学实现）

由隐式式可证 `dt/dA = −[ln(1+F/A) − F/(A+F)]/Ks < 0`：固定时刻 F 随
`delta_theta` 增大而**增大**，渗到同一 F 所需时间随 `delta_theta` 增大而
**缩短**（入渗能力 `f = Ks(1+A/F)` 随 A 增大）。测试按此钉牢。
