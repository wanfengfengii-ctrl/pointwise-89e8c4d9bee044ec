# 观测网络联合校准

Python 3.11 标准库项目。维护一份可更正的独立差分观测账，联合拟合节点势值，并返回具名节点差值的精确估计及方差。所有数值计算使用 Fraction，无随机近似，无持久化或后台进程。

## 运行

在项目根目录执行：

```sh
python3 -B checks.py
python3 -B demo.py samples/network.json
python3 -B bench.py --smoke
python3 -B bench.py
```

checks.py 是完整业务回归，包括独立奇异系统判定器、固定样例、固定种子连续更正、撤桥重连及整批拒绝。demo 逐批回放有限 JSON 样例，输出真实查询结果。bench 默认运行完整负载；smoke 只检查小规模运行链，不作性能准入。

## 公开 API

```python
from ledger import Ledger, Observation
from calibration import Calibrator
from analysis import Analyzer

ledger = Ledger(["A", "B", "C"])
calibrator = Calibrator(ledger)
analyzer = Analyzer(calibrator)
ledger.apply({"ab": Observation("A", "B", 2, 1),
              "bc": Observation("B", "C", 3, 1)})
assert analyzer.query("A", "C") == (5, 2)
ledger.apply({"bc": None})
assert analyzer.query("A", "C") is None
```

Ledger(nodes) 固定节点全集。nodes 是可迭代的非空字符串名集合，允许空集合，不允许重复名；直接传字符串或 bytes 无效。节点名和观测 ID 都区分大小写，不作去空格或 Unicode 归一化。

Observation(u,v,d,w) 是不可变值。u、v 必须不同；d、w 接受非布尔 int、Fraction 或 Fraction 可解析的字符串，不接受 float；w 必须严格大于零。构造时规范成 Fraction。端点是否属于本账在 apply 时检查。

Ledger.apply(batch) 接受 Mapping，键是观测 ID，值是 Observation 或 None。存在 ID 替换，不存在 ID 新增；None 删除已有 ID，删除未知 ID 是错误。同一批只能为每个 ID 指定一个最终值；需要换端点时直接替换该 ID。整个批次验证后才发布，任一非法项抛 ValueError，账、修订和既有分析结果不变。成功返回新修订号，每次成功调用（含空批或相同值重放）修订加一。本接口没有最高修订或请求幂等协议。

Ledger.nodes、Ledger.revision 为只读属性；snapshot() 返回不可变 Snapshot(revision,nodes,rows)，rows 是按 ID 排序的 (ID,Observation) 元组。旧快照不随更正改变。

Calibrator(ledger).current() 返回不可变 State(revision,membership,fits)。membership 将每个节点映射到 fits 下标；每个 Fit 提供 nodes、positions、values 和 covariance。每个分量按节点名排序，以第一个节点固定为零，其余坐标由约化信息矩阵解得；固定坐标的协方差行列为零。该坐标选择不表示已知物理基准。保留的旧 State 不随更正改变。每次新修订均重新搜索全部分量、重新装配并消元，不跨批维护矩阵更新。

Analyzer(calibrator).query(a,b) 返回 (估计 Fraction, 方差 Fraction)，代表 x_b-x_a；同节点返回 (0,0)，不同连通分量返回 None。未知节点（包括未知同节点）或非法名字抛 ValueError。一个查询的估计、方差和可估计性来自同一个当前 State；一个已存在的 Analyzer 自动看到后续成功更正。账和查询均限单进程串行使用，无并发、恢复或持久化承诺。

## 数学语义

每条观测 d 是 x_v-x_u 的独立测量，已知方差为 1/w。联合拟合最小化 Σ w(x_v-x_u-d)²。平行观测各自贡献信息，不作去重；不一致环路按权值共同拟合，不当作输入错误。方差来自已知观测方差，不乘残差方差估计。

令有向关联向量为 b，信息矩阵为 Σ w b bᵀ，右端为 Σ w b d。每个连通分量的平移方向不可识别。固定一个坐标后，约化矩阵正定，逆矩阵是该坐标表示的协方差。差值方差为 C_aa+C_bb-C_ab-C_ba；差值结果不依赖固定节点。孤立节点只允许自身比较。

样例：AB=(2,1)，BC=(3,1) 时 AC=(5,2)；撤回 BC 后 AC 不可估计、AB 不变；加入 BC=(4,2) 后 AC=(6,3/2)。只改权值也必须更新方差。

## 可重复计量

reference.py 是全量数值参照，使用不同的分量算法、末节点定锚及前向消元/回代；不导入生产校准器或分析器。checks.py 还独立构建完整奇异信息系统，通过对比向量是否属于列空间判断可估计性，不依赖生产连通性。数值兼容修改应保留 reference.py、workloads.py、bench.py 和原回归；完整计量打印参照、负载及计量文件的 SHA256，以便绑定运行记录。

workloads.py 固定种子为 20261116+节点数。默认 80 和 160 节点，两个半区各为等权完全观测网，初始 d 为节点编号差；两区只有一条桥。每轮 20 批，每批更正一条区内观测的 d 和 w。stable 保留桥；split 在偶数批撤桥、奇数批以新 d/w 重连。每批后固定请求两个区内比较、跨区正反比较、桥端点比较及同节点比较。生成的输入在计量前完成，不预存答案。

每种规模及负载运行三轮全新的 target/reference。奇偶轮互换执行顺序。初始化含首轮校准，单独报告；周期计时包含全部 20 批 apply 与每批全部查询，以及结果物化。逐项精确比较各轮结果，计时外比较不会省略业务工作。另运行一次 tracemalloc 路径，包含初始化和完整周期，排除预生成输入；该路径结果也必须相同。报告原始耗时、各自中位数、比值、分配峰值及结果摘要。默认不以性能目标决定退出码，业务差异会失败。

性能目标模式：
```sh
python3 -B bench.py --gate
```
四个规模/负载单元各要求周期中位耗时不超过同进程全量参照的 1/4，Python 分配峰值不超过 96 MiB；任一未达标则非零退出。当前全量重算实现不承诺达到该目标。--smoke 仅运行 8 节点、4 批、1 轮，不能与 --gate 合作充当准入。性能输出是运行时实测值，仓库不提供预录通过记录。
