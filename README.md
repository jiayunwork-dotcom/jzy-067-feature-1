# Green–Ampt 入渗核算服务

把 Green–Ampt 湿润锋入渗的集中参数解算封装成 HTTP 后端，**正反两个方向**：

- **正问题**：给定土壤参数（饱和导水率 `Ks`、湿润锋基质吸力 `psi`、锋前后
  含水量差 `delta_theta`）与历时，回报累积入渗量 `F`、瞬时入渗率 `f`、
  隐式方程残差，以及给定降雨强度下的积水时刻判定；
- **反问题（标定）**：吃进一串实测 `(时刻, 累积入渗量)` 观测，反演出最能
  解释数据的 `Ks` 与 `A = psi·delta_theta`，并老实交代这次标定可不可信
  （残差、逐点残差、迭代步数、可辨识性判定）。

纯 JSON API，无网页界面。

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

## 反演（标定）

目标函数：各观测点模型值与实测值之差的平方和。求解用
Levenberg–Marquardt（Nielsen 阻尼更新 + 多起点兜底），在对数参数空间
`(ln Ks, ln A)` 迭代，正向求值复用正问题的隐式求解器。收敛判据为梯度 /
步长 / 目标下降三者之一，其中后两条必须搭配梯度佐证（防止被大阻尼压出
假收敛）。**压不进判据、顶到参数盒边界、或落入无信号区，一律报错，
绝不交半路参数。**

### 可辨识性（这套服务最核心的诚实义务）

只有积水段观测时，`Ks` 与 `A` 沿 `(ln Ks, ln A)` 的某个方向近似简并
（短历时 `F≪A` 极限下只剩乘积 `Ks·A` 可定）。服务按**数据本身**判定：
对解点处列归一化的对数雅可比算条件数
`κ = sqrt((1+|ρ|)/(1−|ρ|))`，`ρ` 为两列相关系数：

- `κ ≤ 50` → `both_identifiable`：两个参数分别可定；
- `κ > 50` → `combination_only`：只回报可辨识的组合（方向、取值、
  标准误）与简并方向，两个单参明确标记 `identifiable: false`，
  并给出 `resolution_hint`（用 `fix` 固定其一，或补含自由入渗段的
  观测并给 `i`，让积水起始时刻参与约束）；
- 用 `fix` 固定其一 → `identifiable_via_constraint`。

判据只看雅可比几何，不看模式名：降雨数据若积水面太浅同样会判
`combination_only`；跨多个时间尺度的积水观测也可以 `both_identifiable`。
降雨模式下若拟合出的积水时刻不早于最后观测的 98%，说明数据里根本没
有积水段，直接判 `no_parameter_signal` 失败而非假装标定成功。

### 标定请求

`POST /calibrate`（202 回 `job_id`，后台作业）：

```json
{
  "observations": [{"t": 0.1, "F": 0.83}, ...]   或 {"t": [...], "F": [...]},
  "i": 3.27,                  可选：降雨强度；不给则按自始积水
  "fix": {"Ks": 1.09},        可选：固定一个参数打破简并
  "initial": {"Ks": 1.0, "A": 5.0},   可选：初值（缺省用启发式）
  "initial_profile": "loam",  可选：用已建档工况当初值（与 initial 互斥）
  "save_as": "field-plot-3",  可选：把可辨识的结果建档为工况
  "delta_theta": 0.434,       save_as 时必填其一（或 psi），把 A 拆回档案参数
  "max_iter": 100, "gtol": 1e-10, "xtol": 1e-12, "ftol": 1e-14
}
```

观测校验挡在迭代之前（400）：点数不足（自由参数数+1）、时刻非严格递增、
入渗量为负、全零无信号、t=0 处 F≠0。

完成后的 `result` 含：`parameters`（含逐参 identifiable 标记与相对标准误）、
`sum_squared_residuals`、逐点 `residuals`、`iterations` /
`total_iterations` / `forward_evaluations`、`convergence_criterion`、
`identifiability`（状态、条件数、组合值等）、`attempts`（各起点结局）、
`profile_save`。取消的作业 `result` 恒为 `null`，绝不交半成品参数；
`save_as` 只在可辨识时真正建档（组合可辨的结果建档等于固化虚假精度，
会被拒绝并说明原因）。

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
| POST | `/calibrate` | 反演标定作业（见上），202 回 `job_id` |
| GET | `/calibrate/<job_id>` | 标定状态/完整结果（含可辨识性判定） |
| POST | `/calibrate/<job_id>/cancel` | 取消标定；取消后 `result` 恒为 `null` |
| GET/PUT/DELETE | `/profiles[/<name>]` | 工况建档（预置 `loam`），持久化在 `DATA_DIR`（默认 `/data/profiles`） |

错误统一为 `{"error": {"code", "reason", "details?"}}`：参数不合法 400、
工况/作业不存在 404、正向牛顿迭代不收敛 422、标定压不进判据或无信号
422（`calibration_not_converged` / `no_parameter_signal` /
`degenerate_solution`）——**不会**把没解出来的迭代值当结果。

## 代码结构

```
model/
  validation.py   参数校验（迭代之前挡下坏输入）
  solver.py       隐式 F 求解（牛顿 + 括号兜底，残差阈值内才返回）
  infiltration.py 入渗率、积水时刻判定、自由/积水分段
  hydrograph.py   历时点列分段推进（可取消）
  calibration.py  反演标定：观测校验、LM、可辨识性判定（独立成块，
                  内部反复调用的仍是上面的正向解算）
  caljobs.py      标定后台作业生命周期（与点列作业分账）
  jobs.py         点列后台作业生命周期
  profiles.py     工况建档持久化（JSON 文件 + 预置壤土）
  routes.py       正问题 HTTP 路由
  routes_calibrate.py 反演 HTTP 路由
  app.py          Flask 装配
tests/            残差达标、符号陷阱、单调走向、积水判定、取消语义、并发隔离、
                  标定还原精度、简并识别、噪声连续、病态输入落报错
```

## 关于 Δθ 的趋势（与常见直觉相反，按数学实现）

由隐式式可证 `dt/dA = −[ln(1+F/A) − F/(A+F)]/Ks < 0`：固定时刻 F 随
`delta_theta` 增大而**增大**，渗到同一 F 所需时间随 `delta_theta` 增大而
**缩短**（入渗能力 `f = Ks(1+A/F)` 随 A 增大）。测试按此钉牢。
