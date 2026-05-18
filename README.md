# 0-456 主干渠 Saint-Venant 动态供配水原型

本仓库整理当前阶段性汇报所用的代码、参数文件、结果图和说明文档，便于老师检查模型假设、输入文件和图表来源。

## 1. 当前说明

- `lineParam.txt` 是渠道断面参数文件，不是单独的新算法。其字段包括线段属性、渠深 `D/H`、底宽 `bottomwidth`、边坡角 `Angle` 和 Manning 糙率。
- 当前程序为 Python/NumPy 自编一维 Saint-Venant 正演原型，用于验证 0-456 主渠段拓扑、断面参数、分水口-支渠网络耦合、供水过程和水位约束。
- 当前原型没有直接调用老师现有一维模型的 `exe` 或 `source code`，也没有直接调用 HEC-RAS、TELEMAC 等外部求解器。
- 当前正演程序的通量项采用 NumPy 向量化的有限体积 HLL 数值通量，摩阻项采用半隐式 Manning 修正；分水口汊点按主渠与支渠水位差形成的能量坡降计算入支流量，并在主渠与支渠间守恒转移水量。因此它的控制方程是一维 Saint-Venant 方程，但数值离散格式不等同于老师现有一维软件；若后续接入老师软件，应按其实际离散格式统一公式和代码。

## 2. 目录结构

```text
data/raw/
  input.txt             # 离散节点编号、坐标、高程、线段属性
  lineParam.txt         # 渠深、底宽、边坡角、Manning 糙率
  neighborId.txt        # 节点邻接关系
  Gates.ini             # 闸门/节点附加信息，当前原型暂未完整使用
data/configuration.json # Saint-Venant 配水模拟工况与数值参数

src/
  simulator.py                               # 统一模拟入口：运行 Saint-Venant 与 Pati/MC 验证模型
  postprocess.py                             # 统一后处理入口：生成图件、表格并同步最终结果目录
  dispatch.py                                # 当前 Saint-Venant 动态配水调度核心
  dispatch_postprocess.py                    # 配水出流过程图和配水结果表后处理
  saint_venant_legacy_dispatch_simulation.py # 早期 Saint-Venant 配水正演原型
  saint_venant_dispatch_topology_and_capacity.py # 图1与支渠能力检查
  cumulative_supply_postprocess.py           # 累计供水后处理
  key_node_depth_postprocess.py              # 关键节点水深后处理
  depth_envelope_postprocess.py              # 沿程最大水深包络后处理/安全校核
  pati_local_reach_applicability.py          # Pati RSR 局部适用性评价
  muskingum_cunge_stage1.py                 # 早期 MC 验证及公共数据解析函数

results/
  figures/              # 汇报使用的最终图件
  tables/               # 汇报使用的 CSV 结果表
  saint_venant_dispatch_results/            # Saint-Venant 脚本默认输出目录
  pati_local_reach_applicability_results/   # Pati 分段评价默认输出目录

docs/
  阶段性汇报_0-456渠段SaintVenant与PatiRSR适用性分析_按老师意见修订.docx
```

## 3. 环境依赖

建议使用 Python 3.10 及以上版本。

```bash
pip install -r requirements.txt
```

当前绘图主要使用 `Pillow` 直接生成 PNG；Word 文档生成脚本需要 `python-docx`。

## 4. 常用运行命令

在仓库根目录运行：

```bash
# 只运行当前 Saint-Venant 配水模拟，输出紧凑摘要
python src/simulator.py dispatch

# 只运行无分水 Saint-Venant 正演/Pati 局部适用性模拟摘要
python src/simulator.py no-diversion
python src/simulator.py pati-applicability

# 一次生成当前汇报所需图件、表格，并同步到 results/figures 与 results/tables
python src/postprocess.py

# 只重跑 Saint-Venant 配水后处理（图2、图3、图4及对应表格）
python src/postprocess.py dispatch

# 只重跑沿程最大水深包络
python src/postprocess.py envelope

# 只重跑 Pati RSR 局部适用性评价
python src/postprocess.py pati

# 只把各模块已有结果同步到最终汇报目录
python src/postprocess.py sync
```

## 5. 当前工况

- 渠首边界：流量由 0 平滑爬升至 80 m3/s。
- 数值工况参数从 `data/configuration.json` 读取；当前时间步长为 1 s，模拟时段为 0-14 h，输出间隔为 60 s。
- 配水口最大能力：71、89 口为 20 m3/s，287 口为 12 m3/s，150、194、349、383 口为 5 m3/s。
- 配水口控制：按最大能力放水，累计供水达到假设需水量后关闸。
- 分水口在 Saint-Venant 框架中作为主渠-支渠耦合汊点处理：支渠链参与动态正演，入支流量由主渠/支渠水位差、支渠安全能力、配水口能力、剩余需水量和局部可取水量共同限制，同时在主渠与支渠之间守恒转移质量并修正动量。

## 6. 结果说明

`results/figures/` 中的图件对应当前阶段汇报：

- 图1：0-456 主渠段和配水口位置图。
- 图2：各配水口放水过程图。
- 图3：各配水口累计供水量与需水量对比图。
- 图4：关键节点水深动态变化图。
- 图5：沿程最大水深包络线。
- Pati RSR 局部适用性分段评价图。
