# 物理边界下的校准范围

这是现有项目的新增功能契约。README.md 中的原接口、数学语义、样例与全部原回归继续适用。当前代码尚无 Bound、apply_bounds 或 range；本文件及 bounds_examples.py 只规定待实现的行为，不包含求解器。

## 调用与输入

```python
from fractions import Fraction
from ledger import Ledger, Observation, Bound
from calibration import Calibrator
from analysis import Analyzer

账 = Ledger(["A", "B", "C"])
分析 = Analyzer(Calibrator(账))
账.apply({"ab": Observation("A", "B", 2, 1)})
账.apply_bounds({"bc": Bound("B", "C", 0, 3)})
assert 分析.range("A", "C") == (Fraction(2), Fraction(5))
```

Bound(u,v,lower,upper) 为不可变值，表示 lower ≤ x_v-x_u ≤ upper。节点名遵守现有 name 规则，u、v 不同；端点属于账的检查在 apply_bounds 时进行。有限上下端接受原 rational 支持的非布尔 int、Fraction 或有理数字符串，构造后为 Fraction，不接受 float。None 表示该侧无界；至少一侧有限，有限下端不得大于有限上端。非法构造抛 ValueError。

Ledger.apply_bounds(batch) 接受 Mapping，键是非空字符串边界 ID，值的精确类型为 Bound 或 None。边界与观测使用独立 ID 命名空间。已有 ID 替换，新 ID 新增，None 删除已有边界；删除未知边界或非法输入抛 ValueError。每个 ID 在批次内给一个最终值，整批候选共同参与可行性判断，不能逐项提交。

新增接口支持至多8个节点、40条当前观测、10个当前边界有限端点；一个双侧边界计两个端点，即使两端相等也计两个。apply_bounds 对候选边界及当前观测检查上述上限；range 对当前账检查上限。有非空边界账时，原 apply 还须检查候选观测数不超过40；没有边界时原 apply 保留原来的规模能力。超限调用抛 ValueError，不改变状态。原 query/current 不增加规模限制。

成功的 apply_bounds 与原 apply 共用一个 revision，每次成功包括空批和相同值重放都加一。共同不可行的边界批次抛 ValueError，账、修订及既有结果不变。旧 Snapshot、旧 State 继续保持不变；原 snapshot 的公开字段保持兼容，不规定新增边界的内部存储形式。边界可行性只取决于物理边界，观测残差不为零不是拒绝理由。正权观测更正不改变物理可行域。

## 输出与分析含义

Analyzer.range(a,b) 返回两个元素的 tuple。它在当前全部物理边界可行的节点势值中，先全局最小化原目标 Σ w(x_v-x_u-d)²，再在全部全局最优解上求 x_b-x_a 的最小值与最大值。有限端点为 Fraction，下端 None 表示负无穷，上端 None 表示正无穷。即使没有物理边界，也须按这个定义处理全部最优解；没有观测时目标恒为零。合法同节点返回 (Fraction(0), Fraction(0))；未知节点或非法名字抛 ValueError，包括未知同节点。范围不是置信区间或方差，不允许浮点容差近似端点。

边界替换、撤回和原观测更正后，已有 Analyzer 须自动得到同一修订的最优集合与范围，不能用任意定锚结果消除原有自由度。校准职责负责联合最优集合，范围分析职责负责具名差值在该集合上的投影；双方交接形式由实现者选择。可以扩展现有业务类及增加内部模块。原 query 与 Calibrator.current 继续返回原无约束估计和协方差语义，物理边界不能改变它们的含义。

允许每修订全量重算、有限活动集枚举及每查询重新求解，不要求增量性能或通用优化引擎。串行内存模型保持不变，不增加持久化、线程或恢复要求；bench.py 的历史性能 gate 不属于本题验收。

## 可手算的样例

1. AB、AC 的观测均为0、权重均为1，加入 Bound("C","B",2,None)。range("A","B") 为 (1,1)，range("A","C") 为 (-1,-1)，原 query("A","B") 仍为 (0,1)。同时将B裁到2而C留在0虽然可行，却不是联合最优。
2. 仅有 AB 观测2、权重1，加入 Bound("B","C",0,3)。range("A","C") 为 (2,5)，反向为 (-5,-2)。改成 Bound("B","C",0,None) 后为 (2,None)；撤回后为 (None,None)。原 query("A","C") 始终为 None。
3. 无观测，Bound("A","B",0,2) 与 Bound("B","C",1,3) 给出 range("A","C") 为 (1,5)。加入 Bound("A","C",None,0) 共同不可行，须整批拒绝。
4. 仅有 AB 观测2、权重1，Bound("B","C",0,3) 下，C=B 是一个最优点，但范围不能因此缩为单点。随后把 AB 观测改为4，原分析器的 AC 范围变为 (4,7)。

正文中的数值对以有理数理解，实际返回类型遵守上节。

## 核验与展示入口

现有正常入口仍为 python3 -B checks.py、python3 -B demo.py samples/network.json、python3 -B bench.py --smoke。保留所有原测试字节。

实现后可运行 python3 -B bounds_examples.py --verify 核对这些公开样例；python3 -B bounds_examples.py --demo 输出实际修订与范围。这些入口在功能实现前缺少目标接口，不能将其作为原项目缺陷。

另补必要的独立正确性核验，覆盖耦合最优、退化、冗余与相等边界、无观测、有限与单侧/双侧无界、反向查询、不可行整批拒绝及观测/边界修订联动，并说明交接为何覆盖全部最优解。只核对同一个求解器返回的点不构成独立参照。

沿现有 .pointwise-runtime.json 扩充 verify 以执行原回归与新增核验；demo 展示边界更正前后实际范围。仍用 version=1、kind=cli、setup 为 argv 数组列表，verify/demo 各为一个 argv 数组；可自行提供调用多个检查的入口。原 demo.py 与样例仍可单独调用。不规定新增实现文件名、算法或测试数量。
