# Green–Ampt 入渗核算服务

把 Green–Ampt 湿润锋入渗的集中参数解算封装成 HTTP 后端：给定土壤参数
（饱和导水率 `Ks`、湿润锋基质吸力 `psi`、锋前后含水量差 `delta_theta`）
与历时，回报累积入渗量 `F`、瞬时入渗率 `f`、隐式方程残差，以及给定降雨
强度下的积水时刻判定。纯 JSON API，无网页界面。

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
| GET/PUT/DELETE | `/profiles[/<name>]` | 工况建档（预置 `loam`），持久化在 `DATA_DIR`（默认 `/data/profiles`） |

错误统一为 `{"error": {"code", "reason", "details?"}}`：参数不合法 400、
工况/作业不存在 404、牛顿迭代不收敛 422（**不会**把没解出来的迭代值当结果）。

## 代码结构

```
model/
  validation.py   参数校验（迭代之前挡下坏输入）
  solver.py       隐式 F 求解（牛顿 + 括号兜底，残差阈值内才返回）
  infiltration.py 入渗率、积水时刻判定、自由/积水分段
  hydrograph.py   历时点列分段推进（可取消）
  jobs.py         后台作业生命周期（queued/running/completed/cancelled/failed）
  profiles.py     工况建档持久化（JSON 文件 + 预置壤土）
  routes.py       HTTP 路由（只做入参与序列化）
  app.py          Flask 装配
tests/            残差达标、符号陷阱、单调走向、积水判定、取消语义、并发隔离
```

## 关于 Δθ 的趋势（与常见直觉相反，按数学实现）

由隐式式可证 `dt/dA = −[ln(1+F/A) − F/(A+F)]/Ks < 0`：固定时刻 F 随
`delta_theta` 增大而**增大**，渗到同一 F 所需时间随 `delta_theta` 增大而
**缩短**（入渗能力 `f = Ks(1+A/F)` 随 A 增大）。测试按此钉牢。
