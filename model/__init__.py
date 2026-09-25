"""Green-Ampt 入渗核算服务。

代码按职责切分：

- ``model.validation`` 土壤参数校验
- ``model.solver``    隐式累积入渗量求解（牛顿迭代）
- ``model.infiltration`` 入渗率与积水时刻判定
- ``model.hydrograph``  历时点列分段推进（可取消）
- ``model.profiles``    工况建档持久化
- ``model.jobs``        后台作业生命周期
- ``model.errors``      错误结构

HTTP 路由见 ``routes.py``，应用装配见 ``app.py``。

全程单位自洽即可：Ks 与 i 为 [长度/时间]，psi 为 [长度]，
F 为 [长度]，t 为 [时间]，delta_theta 无量纲。
"""

from __future__ import annotations
