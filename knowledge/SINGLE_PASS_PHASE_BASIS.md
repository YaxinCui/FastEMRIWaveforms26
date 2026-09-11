# 一次轨道计算，生成整组初相位基

<!-- 2026-09-11 17:46 CST (linux): Derive the implemented exp15 algebra from
pinned FEW and record bounded actual measurements; primary sources are indexes,
not downloaded assets. Local Linux evidence, no Mac-owned file modified. -->

本实验的进展：**不降低精度，而是消除初相位扫描中的重复计算。**
同期 CPU 对照中，准备同一组相位基从 1.575 秒降至 0.325 秒，约 4.84 倍。
这不是所有波形的通用倍率；它适用于其他源参数和探测器配置固定、需要扫描
初始方位相位的工作负载。完整证据、命令与边界见
[Linux 实验记录](../collaboration/linux/HANDOFF.md)。

## 为什么可以复用

FEW 将引力波写成许多模式的和。单个模式具有随时间缓慢变化的复振幅，乘上
由轨道相位决定的振荡因子。固定质量、自旋、初始轨道及观察方向后，改变初始
方位相位 delta，只在模式上引入 exp(-i m delta)；不需要重新积分轨道慢变量
与频率演化，相位只增加常量。[FEW Kerr 波形论文](https://arxiv.org/html/2506.09470v2)
给出了这种谐波结构。不过，重新运行数值 ODE 时，初始相位可能影响步长选择
的浮点细节，因此“数学上的复用”仍需与独立重新生成的波形对比。

FEW 的实现还包含共轭配对，不能给整个配对结果随意乘同一个复相位。令

```text
z_j = A_j(t) exp[-i(m_j Phi_phi + k_j Phi_theta + n_j Phi_r)]
U_j = Yplus_j z_j
V_j = 1_(m_j != 0) (-1)^ell_j Yminus_j conj(z_j)
```

则这个配对在初相位变化后的贡献是

```text
U_j exp(-i m_j delta) + V_j exp(+i m_j delta).
```

对 q=|m|>0 的模式分别合并，得到

```text
C_q = sum_(|m_j|=q) (U_j + V_j)
S_q = sum_(|m_j|=q) [-i sign(m_j) U_j + i sign(m_j) V_j]
h(delta) = B_0 + sum_(q=1..M) [C_q cos(q delta) + S_q sin(q delta)].
```

m=0 只贡献常数 B0，不补一个共轭项。这是对已选择模式的代数重排，未增加模式
截断，也未改成 FP32。配对约定以固定 master 的
[原生求和](https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms/blob/47e4fea4bb3e5fbe1ba34e5e399ed37c97814191/src/few/cutils/interpolate.cu)
和 [Python 包装](https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms/blob/47e4fea4bb3e5fbe1ba34e5e399ed37c97814191/src/few/summation/interpolatedmodesum.py)
为准；前者用 m!=0 判断配对，不能用 m>0 的另一条 Python 实现替代有符号模式验证。

## 怎么落地

[single_pass_basis.py](../collaboration/linux/single_pass_basis.py) 暂时接管
求和入口，在原版轨道、振幅、模式选择完成后，按 |m| 拆分输入列。Cq 保留原
权重，Sq 对正、共轭两支权重分别乘 -i sign(m)、+i sign(m)，再调用未修改的
原生插值求和。(-1)^ell 仍由原包装处理，不能乘两遍。基矩阵经过未修改的
[外层模型](https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms/blob/47e4fea4bb3e5fbe1ba34e5e399ed37c97814191/src/few/waveform/waveform.py)
完成距离缩放、极化旋转和逆行规范转换。所有临时入口均在 finally 中恢复。

本例 M=6，故仍有 13 个基函数。之前是生成 13 次完整波形，再反推出它们；
现在一次轨道和振幅计算就能构造它们。并不是将最终所有模式只求和一次。
正行本例的原生调用仍为 13 次，但每次处理的模式更少。

固定探测器响应对实数 hp、hc 是线性的，故可以先分别处理 B0、Cq、Sq，
再用实数正弦/余弦系数组合。这里不假定响应对复数任意倍乘也线性。实验真的
计算了 13 次 ESA 轨道下的第二代 A/E/T 响应，并非跳过探测器步骤。
响应方法背景见 [LISA GPU 响应论文](https://arxiv.org/html/2204.06633)。

对于固定高斯噪声度量，定义 r0=d-h(0)、Delta h=h(delta)-h(0)，有

```text
Delta log L = (r0 | Delta h) - 1/2 (Delta h | Delta h).
```

因此先计算数据投影和完整 Gram 矩阵，就能只用少量系数扫描相位。本例最终
Delta-logL 最高为 12 阶谐波，利用 Delta-logL(0)=0，保存 12 个余弦差系数
和 12 个正弦系数，共 24 个 FP64 数。所有不同谐波的干涉交叉项都保留，不能
假定它们在有限观测时间、实际响应下正交。谐波化似然属于已有研究方向，参见
[相关多谐波分析](https://arxiv.org/html/2404.02435v2)；本实验的贡献是这个
FEW 实现、数值边界和可复现实测，未证明全球首创。

## 本轮证据如何理解

两个短时源（正行、逆行，T=0.003 年、dt=15 秒）使用实际约 5 GB Kerr 表。
每个源重新生成 22 个相位（21 个非采样网格点与相位零点），在
SNR 20/100/1000 下共 132 次完整链路似然检查通过。
最大加权 TDI 全局相对 L2 为 2.702e-12，最大绝对 Delta-logL 误差为
4.122e-8。另有冻结相位样条检查、双数值环境的 5+6 项小测试，以及 680 项
来源/结果审计。误差足够小，支持这两个用例中的代数重用；不能据此承诺全年
观测、全部参数空间或真实噪声模型下的推断精度。

性能只对正行 CPU 测量，初始化和数据加载已完成。准备阶段五组交替对照给出
4.842 倍；含实际返回与清理的“准备+256 点扫描”为 4.803 倍。这些倍率的
基线都是本轮重新测量的旧式相位基准备，不能与上一轮 19.825 倍相乘。
如果已经缓存探测器基，新建统计量再扫描 256 点比直接计算快 8.479 倍，
但只算 1 或 16 点反而较慢。一次普通波形调用也不需要准备全部相位基。

新的主要耗时是响应、FFT、白化和拼接：约 0.190 秒，占准备时间约 58%。
下一步应先剖析这个阶段及已有缓存，再验证能否让多个实基共享计算或批处理。
不应未经验证把时变响应当成固定复数乘法，或删去某些基/交叉项来换取倍率。
CUDA、Apple GPU、长时序和生产 API 集成均不在本轮验证范围。

<!-- 2026-09-11 17:50 CST (linux): Peer wording review separates slow-orbit
reuse from changing physical phase and clarifies the exact 24-coefficient and
21-off-grid-plus-anchor counts; no experimental code/result changes. -->
