# FEW 多层优化路线图：优化什么、可能省多少、如何证明

<!-- 2026-09-11 11:14 CST (linux): Create the user-requested, source-grounded
multilayer research roadmap. Distinguish committed cross-branch measurements,
read-only HDF5 metadata, analytical bounds, literature results and untested
proposals. Linux root is the sole editor; research agents are read-only.
No production implementation, numerical acceptance or new benchmark is claimed. -->

## 1. 研究范围与主要判断

这份路线图面向 FEW 的算法与工程设计，不局限于把 FP64 改为 FP32。
目标是：在明确的物理模型、观测任务和误差预算下，用更少时间、内存和能量完成可信计算。
因此，“优化”至少有五种不同收益：单条波形延迟、连续任务吞吐量、冷启动/内存、参数估计效率、科学可信范围。它们不能混成一个加速倍数。

**本分支状态。** 本次审查位于 `deep-optimization`，起始提交
`23eef90c69f90b7ac6bd54f5af26ff36e2ff5f39`；其 `src/` 与
`47e4fea4bb3e5fbe1ba34e5e399ed37c97814191` 相同。
这里保存的 Mac/CUDA 文档包括其他分支的历史成果，并不代表这些实现已经合并到本分支。
下文的 5× 实测来自另一分支 `codex/cuda-mixed-precision-5x` 的提交
`b28696b424043a12b49382a20c03c54b11c03cd2`，不是本次重新运行的结果。[^evidence]

综合源码、已有实测和文献，我建议下一步优先研究：

1. **把实际不需要的数据与工作去掉**：按自旋管理振幅表、消除重复拷贝、共享样条矩阵、按依赖关系复用中间量。
2. **针对新瓶颈重新分配工作**：编译/预分配 FP64 轨道计算，多个轨道与 GPU 波形流水执行；不要默认单条稀疏轨道也应搬上 GPU。
3. **直接优化科学任务**：让 FEW、探测器响应和似然在设备上衔接；研究只计算少量频点、少量高价值候选，而不是每次都输出完整全年数组。
4. **把误差预算变成可执行策略**：依据参数域、模式、相位和下游用途选精度；保留严格回退和独立参考。

这些方向可以叠加，但收益必须重新测量；共享同一段工作量的优化不能把倍数直接相乘。

### 证据标签

| 标签 | 本文含义 |
|---|---|
| 实测 | 本项目已有原始结果文件，附分支、对象和测量范围；不等于整个参数域通过 |
| 元数据/推算 | 根据实际数组结构、运算关系或阶段时间计算；不是实际运行性能 |
| 文献 | 他人已发表工作的结果，明确模型和平台，不能外推成本机成绩 |
| 假设 | 有源码或数学依据的实验方向；没有数值依据时不编造加速倍数 |

## 2. 当前证据告诉我们：瓶颈已经移动

### 2.1 5× 实验究竟证明了什么

RTX 2080 Ti 上，同一组一年 Kerr 波形、预热后同步计时的中位数为：
FP64 **152.4895 ms**，`mixed32_intrinsic_fast` **26.1811 ms**，约 **5.8244×**。
它是代数整理、精度分配与快速数学函数等组合的结果，不能归因于“FP32 单独快了 5.8 倍”。
短波形的对应收益仅约 1.69–1.82×。该计时不包含 HDF5 初始化、完整 TDI/似然、训练或参数估计。[^evidence]

显存单位也需要校正：原始 CuPy pool `used` 是 **4950.059 → 2523.915 MiB**，
即 **4.834 → 2.465 GiB**，降低约 **49.01%**，不是“4.95 → 2.52 GiB”。
pool `reserved` 则为 5053.263 → 2627.118 MiB。
这两种计数都不等于含 CUDA context、库资源的进程总显存，更不等于全过程峰值。[^evidence][^cupy]

误差扫查的七组案例全部有限且保持所选模式一致；最大 flat mismatch 约
`3.23e-12`，但严格逐点门槛未全部通过，也没有完成探测器加权、梯度和参数偏差认证。
“工程候选表现很好”和“已经满足所有科学用途”必须分开。[^sweep]

### 2.2 用阶段占比限制预期，而不是继续猜倍数

另一份带阶段同步的诊断计时如下。其总耗时是 **30.116 ms**，
不能把下面分项直接从无插桩的 26.181 ms 中减去。[^profile]

| 阶段 | 中位耗时 | 在诊断总时间中的比例 |
|---|---:|---:|
| 轨道计算 | 15.483 ms | 51.4% |
| 振幅及模式选择 | 4.349 ms | 14.4% |
| 求和全阶段 | 9.261 ms | 30.8% |
| 其中：样条构建 | 2.567 ms | 8.5% |
| 其中：编译求和内核 | 6.397 ms | 21.2% |

最后两行包含在“求和全阶段”里，不能再次相加；另有约 1.023 ms 未归因。
此例只有 30 个稀疏轨道点，却有 2,103,877 个输出采样、124 个所选模式。

若占比为 \(f\) 的阶段快 \(s\) 倍，其余不变，则

\[
S_{\rm total}=\frac{1}{(1-f)+f/s}.
\]

由此可得条件性上界：轨道快 2×，整体约 **1.35×**；轨道快 3×，整体约 **1.52×**；
即使轨道免费，整体也最多约 **2.06×**。
反过来，即使编译求和内核免费，整体也最多约 **1.27×**。
这是针对该诊断案例的 Amdahl 推算，不是其他参数、CPU 或 Mac 的预测。[^cuda]

## 3. 十四个层级的机会总览

表中的“目标”都是待验证的实验目标；没有依据的地方只写收益方向。
编号用于后续拆分实验，不能理解为已经实施。

| 层级 | 优化点与思路 | 预计效果与主要边界 |
|---|---|---|
| L0 任务与误差契约 | **O01** 区分教学展示、候选搜索、精确后验、Fisher/梯度；分别指定模型域、观测时长、PSD 和允许误差 | 避免为低要求任务支付不必要成本，也防止高精度任务误用快速模式；不能直接报速度倍数。见第 6 节 |
| L1 物理模型 | **O02** 为已有模型增加经过验证的 1PA/环境模块；**O03** 搜索与精修采用多保真策略 | 前者扩大科学可信范围，可能更慢；后者减少昂贵模型调用。不能用“更低物理保真”冒充“相同模型算得更快”。[^ode][^nit] |
| L2 离线真值与网格 | **O04** 误差驱动加密网格；**O05** 分区残差、谱系数剪枝、低秩表示；分开验证通量与应变模式截断 | 可能减少昂贵 BHPT 数据点、表体积，或改善边界误差；新表示未必比局部样条快。先做小范围截面实验。[^systematics] |
| L3 数据存储与内存 | **O06** 自旋切片加载；**O07** 主机只读缓存 + 设备 LRU/预取；**O08** 按块精度存储并保留 FP64 原表 | 两个相邻自旋面的 FP64 系数原始载荷可从约 4.739 GiB 降至 **294.078 MiB**，约省 93.94%；不是进程峰值承诺。热计算不一定更快。见第 4.1 节 |
| L4 轨道数值计算 | **O09** FP64 DOP853 预分配与编译；**O10** 固定自旋不变量缓存/样条预收缩；**O11** 多系统批处理 | 第一轮以轨道阶段 2× 为实验目标；在现有优化案例中只对应约 1.35× 整体。批处理主要增加吞吐，不保证单条低延迟。[^trajectory][^flux] |
| L5 振幅与选模 | **O12** 消除全模式 gather/flatten 拷贝；**O13** 复用排序、频率与球谐量；**O14** 先预测候选模式，再进行有误差约束的精算 | 同 FP64 零拷贝已有局部约 1.14–1.27×、长波形整体约 1.02× 的证据。候选筛选空间更大，但漏模是科学风险。FEW 已支持 `sensitivity_fn`，不是从零引入 PSD 选模。[^zero-copy][^selector] |
| L6 插值与代数结构 | **O15** 公共时间网格一次分解、多 RHS 样条求解；**O16** 固定自旋预混合振幅系数；**O17** 按共同谐波相位聚合模式 | O15 减少重复存储/分解；即使样条构建免费，该案例整体上限也仅约 1.093×。O16/O17 是需要核验交换次序条件的代数实验，不保证逐位相同。见第 4.3 节 |
| L7 数值精度 | **O18** 按误差敏感性选择 FP64/FP32/补偿求和；**O19** 对可靠矩阵算子单独探索 FP16/TF32 | 现有组合有 5.82× 热波形实测，但新目标应是保留大部分收益并扩大可靠参数域；不能把硬件峰值比当应用倍数。[^evidence] |
| L8 GPU 执行与异构协作 | **O20** 复用 stream、减少全设备同步、限定形状使用 Graph；**O21** CPU 轨道与 GPU 后半程流水；按工作负载选择 CPU/CUDA/Metal | 调度改善对短请求更有意义；单 CPU 轨道阶段与其余阶段理想重叠，当前案例吞吐上限约 1.95×，不是单条延迟收益。实际还受缓存、队列、内存和争用限制。[^kernel][^graphs] |
| L9 波形表示与采样 | **O22** 按用途选 TD/FD；**O23** 下游直接消费稀疏频点/解调慢变量，减少全年稠密输出 | FEW 已有 FD。论文四年 Schwarzschild 示例中，稀疏 FD 相对 TD 的波形生成：CPU 约 226×，GPU 仅约 1.83×；不含完整 TDI/似然，也不能外推当前 Kerr。SPA 转折与端点需检查。[^fd] |
| L10 探测器与似然 | **O24** 波形→TDI→FFT/PSD→内积设备驻留；**O25** 局部逐模式 relative binning/heterodyne；**O26** 局部 ROQ | O24 减传输与中间数组；O25/O26 有望显著降低频点工作量，但 FEW 的完整似然倍数尚无本地实测，而且两者收益高度重叠。[^response][^binning][^roq] |
| L11 推断与搜索 | **O27** 合法延迟接受；**O28** 模块灵敏度/JVP/VJP；**O29** SBI 粗定位 + 精确 FEW 精修 | 评价 ESS/s、覆盖率、漏峰率，而不只看 waveform/s。延迟接受的示例成本可改善 3.3×，但有效样本效率未必同比。EMRI SBI 文献的百万倍是搜索体积缩小，不是计算加速。[^delayed][^autodiff][^sbi] |
| L12 服务与 Agent 接口 | **O30** 常驻实例、共享只读表、不可变请求、按依赖缓存；**O31** 资源预算、原子下载、异常恢复、可取消任务 | 将初始化成本摊到多次调用，避免多实例重复占表。改善冷启动、可用性、吞吐及资源上限；需要单独的服务基准，不能记作热内核加速。[^lifecycle] |
| L13 测试、复现与能耗 | **O32** 独立参考阶梯与真实 GPU CI；**O33** 参数域误差地图、发布门槛；**O34** 同精度下测能耗、持续负载和跨设备回退 | 主要降低错误结论和返工，提升可交付性。Mac 移动使用应看持续吞吐与能量/有效任务，不能仅比较一次峰值计时。当前测试存在需要先修正的覆盖缺口。[^tests] |

## 4. 值得优先展开的计算组织优化

### 4.1 5 GB 表：优化“什么时候、向哪里装载”，不必先改物理

本次对本机 `ZNAmps_l10_m10_n55_DS2Outer.h5` 做了只读结构检查：

| 项目 | 实际元数据/计算值 |
|---|---:|
| 文件总长度 | 5,089,095,248 bytes |
| Region A 系数形状，类型 | `(33, 6993, 2, 1089)`，float64 |
| Region B 系数形状，类型 | `(33, 6993, 2, 289)`，float64 |
| 两区域系数总载荷 | 5,087,994,912 bytes = 4.739 GiB |
| 一个自旋面，两区域合计 | 154,181,664 bytes = 147.039 MiB |
| 两个面，FP64 | 308,363,328 bytes = 294.078 MiB |
| 四个面，FP64 | 616,726,656 bytes = 588.156 MiB |
| 两个面，假设存成 FP32 | 154,181,664 bytes = 147.039 MiB |

文件位于本机 FEW 数据缓存目录；身份以文件名和
[数据注册表](../src/few/files/registry.yml) 的 SHA-256
`3236d8b5eff618242291e9eeb24638dbbbb82fac464ae575dc3f2ba158c54834`
为准。本轮没有重跑整文件哈希或读取所有系数；上述尺寸来自 HDF5 元数据。

当前振幅代码在构造时读完整系数，而单条固定自旋波形在线只使用一个或两个相邻自旋面。
因此，**切片加载是有明确访问依据的优化，不是发现了整张表“都没用”**。
论文也明确知道系数约 5 GB 并选择让其适配多数 GPU；可以推测 eager 有利于摊销重复访问，
但不能把“作者专门为 MCMC 全表常驻设计”当作已有直接证词。[^amplitude][^kerr]

建议把加载策略变为显式选项：

- `eager`：高复用、宽自旋搜索、显存充裕。
- `lazy`：一次调用/移动设备/显存紧张，仅装邻近面。
- `host-cache + device-cache`：主机保留更多只读数据，设备设可控缓存预算。

每个缓存 key 应含数据哈希、区域、自旋节点、模式布局和精度，不能只按文件名。
LRU 容量四面只是在局部搜索中可能有效；宽自旋跳转、并行温度链可能使其抖动。
两个区域的缓存都达到四面时，原始 FP64 载荷约 588 MiB，而不是一个区域的数值。

当前两个系数数据集都是 **contiguous、未压缩**，首维是自旋，因此现有布局已能按面读取，
无需先重新 chunk 才能实现 lazy。若以后按模式读取、压缩或并发访问，应再按实际访问块设计布局；
chunk 太大可能读入大量无用数据。[^hdf5]

必须同时测构造、数据校验、首条波形、后续波形、缓存 miss、CPU RSS、CuPy used/reserved、
进程显存峰值和主机—设备传输量。若首次校验仍扫描整个文件，lazy 不会自动消除这部分 I/O；
释放 Python 对象后内存池也可能继续保留显存。[^cupy]

**预期：** 两面 FP64 原始系数较 33 面少 93.94%，这是可靠算术；
首条延迟、实际峰值和连续任务是否更快，仍需实验。两面 FP32 较全表 FP64 的原始载荷少 96.97%，
同时包含访问裁剪与有损降精度，必须分别验收，不能包装成无损压缩。

### 4.2 轨道：先消除解释器和重复工作，再决定是否上 GPU

轨道决定长期相位。示意上，频率误差通过
\(\delta\Phi(T)=\int_0^T\delta\Omega(t)\,dt\) 累积，
所以“轨道输出只有几十个点”不意味着精度不重要，也不意味着内部只算几十次 RHS。
应记录真实 RHS 次数、接受/拒绝步数和连续输出误差。[^trajectory]

当前可从三步开始：

1. **FP64 预分配。** DOP853 步进中有多组临时数组；把 workspace 生命周期提升到一次积分，
   保持同一算法、容差和边界条件。
2. **固定参数特化。** 缓存固定自旋下的常数；对 tensor spline 研究先收缩固定维度，
   在线求二维值。做系数收缩而不是重新拟合，能避免引入额外拟合层。
3. **编译热循环。** 降低 Python、NumPy、RHS 回调之间的细粒度切换；先保持 FP64，
   将快速数学开关作为独立实验变量。[^trajectory][^flux]

保持算法也不保证逐位一致：运算顺序变化可能影响自适应拒绝步数。
验收要检查终止点、整个相位历程、Region A/B 接口、逆行、近 separatrix、长观测和失败重试。

多系统并行是另一条路线：对多个参数点分别积分，按难度分桶，
CPU 生成下一批稀疏轨道时 GPU 处理上一批振幅/求和。
单任务因果依赖并不禁止跨系统并行；不过不能把不同请求误放进同一个有可变状态的模型实例。[^lifecycle]

### 4.3 不改变物理的代数优化，也需要逐项证明

**公共样条矩阵。** 当前插值会复制公共时间网格，并对多路数据建立相同三对角系数。
矩阵只依赖时间节点，RHS 才依赖振幅/相位，因此可共享网格和一次分解，再解多个 RHS。
有关网格/矩阵存储可从 \(O(MN)\) 降为 \(O(N)\)，但振幅、RHS 和结果仍需 \(O(MN)\)。
必须保留相同样条端点条件，并测非均匀/病态网格。[^spline]

**固定自旋预混合。** 若两相邻面的二维基函数与坐标相同，线性性给出
\[
(1-\alpha)\mathcal I(C_0)+\alpha\mathcal I(C_1)
=\mathcal I((1-\alpha)C_0+\alpha C_1).
\]
本机表的每个区域都只有一套公共 `u/w_knots`：A 各 37 个、B 各 21 个，
并不随自旋面变化。当前代码也在请求的自旋处只做一次坐标映射，供上下两面使用，
中间没有非线性振幅因子，因此同一区域内有交换依据。注意必须使用代码的 **z 坐标权重**，
不是物理自旋 a 的线性权重；A/B 基底不同，不能跨区域混合。[^amplitude]

可尝试先混合系数，再在线只做一次二维求值。但全量预混需读两面、写一面，
仅此就至少产生 **462,544,992 bytes，约 441.12 MiB** 数据流量，
可能比在几十个轨道点上分别求值更贵。
优先考虑重复同自旋时缓存预混结果，或在求值内核中共享基函数并融合两个面的局部系数读取。
后者还要与当前整网格共享内存方案实测比较，不能假设访问更少一定更快。
精确命中自旋节点时原本只求一个面，没有两面融合收益。

**共同相位聚合。** 赤道谐波中的相位 \(m\Phi_\phi+n\Phi_r\) 不依赖 \(\ell\)。
在模式集合固定、角权重与时间无关且插值是同一线性算子的条件下，可以研究
先对同 \((m,n)\) 的不同 \(\ell\) 合并加权振幅，再做密集时间求和。
理论上把相位求值次数从“所选模式数”降至“不同相位组数”；并非所有参数都有大量可合并模式。
必须在**现有模式选择完成后**聚合，避免改变功率排序规则。每组一般要保留两条独立复振幅：
\[
B_g^+=\sum_{\ell\in g}Y_\ell^+A_\ell,\qquad
B_g^-=\sum_{\ell\in g}\widetilde Y_\ell^- A_\ell^*,\qquad
h_g=B_g^+e^{-i\psi_g}+B_g^-e^{+i\psi_g}.
\]
这里负支权重吸收现有符号和排除掩码；通常不能从一个聚合振幅整体共轭恢复另一个分支。
当前 kernel 对 \(m=0\) 不另加共轭负支，不能重复计入。
稀疏点聚合与当前公共 not-a-knot 样条线性相容；若改成逐模式不同 knots、非线性插值、
或时变角权重，则不能直接交换顺序。需要分别返回各 \(\ell\) 模式的接口也不能丢掉原模式信息。

设稀疏点数 \(N_s\)、密集点数 \(N_d\)、所选模式数 \(M\)、相位组数 \(G\)，
有关求和工作可由 \(O(N_dM)\) 变为 \(O(N_sM+N_dG)\)。
这只是算量关系：两条独立分支、预处理、内存布局及相消误差都会影响实际收益。
应先用小型 FP64 原型验证完整符号约定，再报告实际 \(M/G\)、盈亏平衡长度和完整波形时间。[^kernel][^selector]

### 4.4 混合精度：优化单位是算子与误差来源，不是整个数组的 dtype

一个可检验的分层是：FP64 动力学与长期相位；受控的 FP32 振幅插值与局部三角函数；
对敏感归约保留 FP64、分块或补偿求和。精度策略必须显式记录到输出元数据。
最终输出 `complex128` 不会恢复前面已经损失的有效数字。[^evidence]

还应研究“误差感知回退”：高偏心率、强场、相消严重或误差估计偏大的块，
选择更高精度路径。弱模式不必然不重要：大项相消之后，弱项可能控制残差。
回退门槛需要连续性与梯度测试，不能只用最终 mismatch 调一个常数。

FP16、BF16、TF32/Tensor Core 只对适合矩阵计算的算子值得单独探索，例如批量降阶重构。
它们不能直接加速一切 FP64 三角函数、控制流或 ODE。
先验证值域、缩放、舍入和累加精度，再谈吞吐；不存在“FP64 改 FP32，应用必然快 32 倍”的定律。

## 5. 更大的收益可能来自少生成波形、少评估候选

### 5.1 从输出全年数组，转为直接满足下游需求

很多最终任务只需要几个噪声加权内积，而不是在 CPU 上拿到数百万个复数。
建议在接口上区分“返回完整波形”“指定频点”“指定内积/似然”，并保持结果定义一致。
一个含 \(N\) 点的 complex128 数组大小为 \(16N\) bytes；
在这里的一年样例中约 32.10 MiB，多通道和批量请求会进一步放大中间内存与传输成本。

FEW 已有 FD 实现，不应重造已有功能。可以先比较 TD+FFT、完整 FD、稀疏 FD 的
**波形+响应+似然**总耗时，再选择路径。
《Fast and Fourier》四年 Schwarzschild 示例中 CPU 时间为
90.6/13.7/0.4 秒，GPU 为 64/48/35 毫秒。这里仅指论文 Table 2 的波形生成时间，
FFT 另列，未包含完整 TDI/似然，说明“表示改变的收益”极依赖平台和任务。[^fd]

TDI 也不能被简化为所有频点乘一个标量：只有常时延才对应简单指数相因子。
真实运动星座含时变延迟，需使用相应响应模型；GPU 驻留是计算组织优化，
不是丢掉星座运动的物理近似。缓存几何时必须区分观测时刻、星座轨道和天空方向依赖。[^response]

### 5.2 相对分箱与 ROQ：压缩似然，而不一定重建整条波形

相对分箱利用参考附近波形之比缓慢变化，预存数据与参考的统计量。
EMRI 有许多干涉模式，整体波形零点附近的比值未必平滑，因此优先试逐模式/模式簇版本，
并逐步增加分箱数验证 \(\Delta\log\mathcal L\)。
原始 GW170817 研究约 60 箱、相对朴素方法约 \(10^4\)×，这是 CBC 文献结果，不是 FEW 预测。[^binning][^mode-binning]

ROQ 则从训练族中选择基和求积节点，直接近似内积。
从 \(N\) 个频点变为 \(K\) 个节点只给出有关算子的工作量比 \(N/K\)，
不含训练、轨道、响应和域外处理；FEW 的长相位与高维空间可能需要很大的基。
优先做局部参数盒、分模式或解调后慢变量，而不是预设“100 个节点覆盖所有 EMRI”。[^roq]

这两类方法的前提都是能便宜地得到所需节点；若每次仍先完整生成稠密波形，收益会被前半程限制。
离线建立成本 \(C_{\rm setup}\) 的盈亏平衡调用次数约为
\(C_{\rm setup}/(C_{\rm full}-C_{\rm fast})\)，应与预期工作负载一同报告。

### 5.3 多保真、梯度与神经搜索：评估统计效率

**延迟接受。** 便宜模型先筛候选，通过者再算严格模型，并采用正确的两阶段接受率。
若便宜模型成本 \(C_c\)、完整模型成本 \(C_f\)、第一阶段通过率 \(p\)，
每提议平均成本近似 \(C_c+pC_f\)。假设 \(C_c=0.1C_f,p=0.2\)，
单提议成本可理想下降约 **3.3×**；但拒绝增多、多峰混合变差时，ESS/s 可能不升反降。
定期抽检便宜模型不等于保持精确后验，需要正确修正和支持集条件。[^delayed]

**可微模块。** 优先做轨道切线灵敏度、样条解析导数及原生内核的自定义 JVP/VJP。
中心差分对 \(p\) 个参数通常需要 \(2p\) 次正向计算，但自动微分也有计算/内存成本。
Python/C++ 边界不是数学上的不可微墙，没必要为“能微分”立即重写全项目。
已有 ripple 工作是 CBC 工程先例，不能把其 Fisher 加速直接当 FEW 成绩。[^autodiff][^ripple]

**SBI 搜索。** 最新核对的 Cole 等工作 v2（2026-06）在非自旋 EMRI 的 11 维搜索中
把候选体积缩小 \(10^6\)–\(10^7\)，输出一维边缘提议分布。
可研究它与精确 FEW 精修的组合，但这既不是运行时快百万倍，也不是通用联合后验问题已经解决。
训练成本、覆盖率、漏峰率、域外信号和噪声失配都是验收项。[^sbi]

## 6. 统一验收：更快必须相对于同一任务和明确误差预算

### 6.1 不能只看一个 mismatch

应分别建立物理模型、离线数据、轨道积分、插值、选模、浮点计算、探测器响应、似然压缩的误差账本。
与现有 FP64 一致只约束其中一部分；CPU/CUDA 同时继承错误表格，仍可能彼此完全一致。

在采用平稳高斯噪声似然、PSD 已知的近似下，探测器通道的内积可定义为
\[
(a|b)=4\operatorname{Re}\int_{f_{\min}}^{f_{\max}}
\frac{\tilde a(f)\tilde b^*(f)}{S_n(f)}\,df,
\quad \delta h=h_{\rm candidate}-h_{\rm reference}.
\]
多个独立通道求和；相关通道噪声则使用谱协方差的逆矩阵。
非平稳噪声或数据缺口通常还需要更完整的协方差/数据处理，不能只套这个对角 PSD 表达式。
对同一物理参数的绝对探测器波形误差，在相应参数测量条件下，
\((\delta h|\delta h)<1\) 是常用的保守充分性参考；未通过不等于必然产生不可接受偏差。
它不是所有模拟的万能合格线，还应依据 SNR、用途和其他误差来源分配数值预算。[^accuracy][^accuracy-use]

必须同时报告：最大绝对/归一化点误差、相对 \(L_2\)、直接误差向量范数、
明确是否优化时间/相位的 overlap/mismatch、PSD/TDI 加权误差。
归一化 overlap 会隐藏整体幅度缩放；极小的 `1-overlap` 还有相减消去问题。
多来源误差若没有独立性依据，不能简单把平方误差相加；保守做法是按范数三角不等式分配预算。

用于 Fisher/HMC 时还需独立的导数收敛测试：波形误差在有限差分中会除以步长，
模式阈值、终止事件、缓存/精度分支引入的不连续都可能破坏梯度。
最终参数估计还需注入恢复、偏差、覆盖率和 ESS/s，而不是只通过波形相似度。

### 6.2 最小可执行测试矩阵

| 维度 | 至少覆盖什么 | 为什么 |
|---|---|---|
| 物理域 | 低/高自旋、顺行/逆行、近圆/高偏心、不同质量比、两数据区域接口、近终止边界 | 随机平均测试容易遗漏最敏感区域 |
| 时间与模式 | 短片段、一年/四年；单模式、固定模式、自动选模；阈值附近连续扫描 | 分开测相位累积、模式相消/干涉和模式切换 |
| 数据访问 | 精确自旋节点、节点左右、重复同自旋、相邻自旋、全域跳转、不同 batch | 检查切片索引、缓存命中与抖动 |
| 数值层级 | 独立 BHPT 小参考→FP64 CPU→FP64 CUDA→候选 CUDA/Metal | 区分模型、实现和浮点误差 |
| 性能口径 | 构造/首调用/热调用；同步 wall time；p50/p95；吞吐；内存 used/reserved/进程峰值 | 不把异步提交时间、初始化推迟或缓存内存当实际收益 |
| 下游统计 | PSD/TDI、\(\Delta\log L\)、梯度、注入后验与 ESS/s | 检查科学任务真正关心的结果 |
| 服务故障 | 异常后重试、取消、OOM、下载中断、版本变更、并发和资源回收 | 长驻 Agent 服务不能一次失败后污染后续请求 |

已有 5× 结果采用两次预热、七次重复，适合初步候选筛选；正式比较应增加跨进程重复，
随机化 A/B 顺序，记录功耗/温度状态及噪声区间。
本轮没有重新运行性能测试，因此未把任何新方向标为“已加速”。

## 7. 对早期知识草稿的修正

以下修正适用于本分支早期 `NEXT_GEN_EMRI_SIMULATION_BLUEPRINT.md` 和信号处理指南；
保留原文作为历史记录，不把其中强断言继续当事实传播。

- **FEW 框架不等于一个固定物理模型。** `ODEBase.evaluate_rhs` 与额外参数接口允许扩展；
  环境项可以研究接入，但接口存在不等于环境模型已被验证。[^ode]
- **1PA 不只是“一阶保守自作用力”。** 它涉及一阶保守/振荡自作用力及二阶平均耗散等贡献；
  不能无条件断言所有系统几个月后完全脱相，或相位误差统一按 \(t^2\) 增长。[^self-force]
- **赤道模型不能被指控抹平径向—极角共振。** 这一共振讨论属于一般倾斜 Kerr；
  将来的 Full/Partial NIT 方案也应与对应动力学比较，而非宣称相对当前已平均 FEW 必然快百倍。[^nit]
- **越过 separatrix 不意味着轨道频率发散到无穷。** 束缚绝热模型需要 transition/plunge 的适用域处理；
  偏心 separatrix 极限中径向周期增大、径向频率趋零。[^geodesic]
- **ROMAN 不是学习轨道历史的网络。** 其职责是瞬时轨道参数到降阶振幅的映射；
  缺少 RNN/Mamba 记忆本身不能证明漏掉某种物理效应。[^roman]
- **高偏心与高角向多极不可混为一谈。** 通量的离线截断、应变库模式覆盖和在线选模是三层误差；
  圆 Kerr 通量文献的高 \(\ell_{\max}\) 要求不能直接套到这个应变库文件名上。[^systematics]
- **频率不总单调、TDI 不总是标量乘法、ROQ 不保证固定百节点或精确无误差。**
  分别需要处理多个驻相点、时变延迟以及训练域/误差控制。[^fd][^response][^roq]
- **不能断言整个 FEW 始终受显存带宽限制。** 当前优化案例中 CPU 轨道已占一半；
  应分算子测算力、带宽、启动和同步，而不是只用 GPU 理论峰值解释一切。[^profile][^cuda]

## 8. 推荐实验顺序与继续/停止条件

| 阶段 | 有界工作包 | 继续条件 / 停止或转向条件 |
|---|---|---|
| P0：建立可信基线 | 修正振幅参考测试循环覆盖；固定模型/数据/后端身份；补全过程与下游计时；小型独立参考 | 先明确默认 FP64 是否通过参考；若参考冲突，先定位，不放宽门槛掩盖 |
| P1a：低内存 | eager/lazy/主机+设备缓存，仅 FP64；包含首调用和局部/宽域参数序列 | 实际峰值显著下降且目标工作负载延迟可接受；若 cache miss 抵消收益，保留策略选择 |
| P1b：同精度少做事 | 公共样条分解、去冗余拷贝/排序、stream 生命周期；小原型验证共同相位聚合 | 单独消融有正收益、完整请求改善超出计时噪声；局部快而整体无效时降优先级 |
| P1c：轨道与批量 | FP64 workspace/编译/不变量缓存；batch=1/4/16/64 与 CPU/GPU 流水 | 先争取轨道阶段 2× 或明确吞吐改善；检查相位、步数、终止行为和线程争用 |
| P2：受控近似 | 参数域误差地图、选择性高精度回退、FD 与稀疏消费 | 在预先定义的科学门槛下比较速度—内存—误差；不能仅挑最快的安全点 |
| P3：任务级加速 | 设备驻留似然→局部分箱→延迟接受；再视瓶颈做 ROQ/可微/SBI | 以总耗时、ESS/s、覆盖率衡量；训练与预计算必须计入，逐项检查漏峰 |
| 独立科学路线 | 更完整自作用力、一般轨道与环境模型，离线误差驱动建表 | 以新增物理范围和独立精度验证为目标；不把当前 0PA 性能直接当公平基线 |

Ubuntu 适合承担 CUDA 原型、FP64 对照、显存与任务吞吐验证；Mac 负责相同输入下的 CPU/Metal 实验、
统一内存压力及持续移动负载测试。两端共享模型/数据哈希、参数清单、误差定义和证据格式，
不直接比较不同物理模型、不同精度或不同冷/热状态的“GPU 快多少”。

本轮只形成研究路线和文档勘误，没有改生产源码、切分支或提交/推送。
最值得推进的不是一个不受条件限制的“再快 N 倍”，而是依次证明：
**省去哪些工作、保持什么不变、在哪类任务中得到多少实际收益。**

## 9. 执行阶段补充：缓存依赖与任务边界

<!-- 2026-09-11 13:52 CST (linux): Add the first measured dependency-cache
experiment and a source-grounded next investigation, without revising earlier
unmeasured proposals into retrospective performance claims. -->

`codex/exp06-prepared-observer-scan` 在独立 master 分支实现“固定内参、改变观察方向”：
缓存完整轨迹、连续相位系数、频率及未乘角权重的全部振幅；每次仍重新计算角权重和选模。
如果带自定义选模 PSD，则冻结固定频率向量上的 PSD 数值，而不是保留可变回调引用。
源码依据是 master 的 `waveform/base.py` 与 `utils/modeselector.py` 的参数依赖关系。

首轮直接计入准备成本的四视角任务显示：CPU 短波形约 2.46–3.40×；CUDA 一年波形只有
1.02–1.14×。更重要的反例是 CPU 固定三个模式：约 0.10×，即慢约十倍，因为原路径只插值
这三个模式，而准备路径先插值全部模式。**缓存是否值得，取决于被复用的工作本来有多少，
以及准备成本能被摊销多少次。** 这些是同进程温态 pilot，不是所有 FEW 输入的结论。
原始记录、完整矩阵状态及精度门槛见[独立实验索引](../collaboration/linux/EXPERIMENTS_2026-09-11.md)。

当时的下一项候选是固定响应下的距离条件似然；现已在独立 exp09 试验实现，结果见下文。
由高斯似然内积式[^response]，若所有其他
源参数、质量约定、PSD、频带、窗函数和线性响应均固定，可写
\(s(D)=(D_0/D)s_0\)。令 \(r_0=d-s_0\)、\(\delta=(D_0-D)/D\)，则

\[
\Delta\log\mathcal L(D)
=\delta(r_0|s_0)-\tfrac12\delta^2(s_0|s_0).
\]

这来自代数展开，不是本文声称发明的新物理方法。它避免反复生成波形/响应/做频率积分，也避免
先计算两个较大的绝对 logL 再相减。近 \(D_0\) 时直接计算距离差仍应验证浮点误差。
公平对照至少应包含“已有频域信号、按距离缩放、直接算残差范数”，不能只拿反复完整 FEW
调用作对照。预计算成本、零噪声/固定随机噪声、不同 SNR、近基准距离点、FFT 归一化和
两种实偏振的约定都要测。没有真实探测器响应时，只能称为理想通道实验，不能称为 TDI 验证；
固定源质量但随距离改变红移时，也不能只做 \(1/D\) 缩放。

<!-- 2026-09-11 14:47 CST (linux): Add actual exp09 evidence, numerical
oracles, unbiased timing boundaries and the deployment-layer failure. -->

### 9.1 距离复用：算法收益必须与调用方式、摊销成本一起看

独立 `codex/exp09-distance-conditional-likelihood` 保持 master 的 FEW 源码不变，
用现有环境生成五个距离下的真实 Kerr SSB 波形，再由隔离环境执行 ESA 轨道、二代 TDI、
25 阶插值的 A/E/T 响应。相位/天空方向/探测器参考系质量等全部固定。最大通道缩放相对
误差约 2.83e-13；这证明了此次链路的数值缩放一致性，不等于独立认证响应物理精度。

258 个 100 位 Decimal 直接残差对照、40 个奇偶长度时频 Parseval 对照、99 个扩展精度
TDI 条件似然对照、45 个重新生成完整 FEW→响应→FFT 的对照给出不同层级的证据。
噪声度量是明确设定的合成对角 Gaussian PSD，不冒充真实非等臂 ESA 噪声协方差。
局部似然误差与极远尾部采用不同尺度门槛，不能笼统宣称所有绝对误差小于 1e-8。

公平基线已经拥有相同的频域信号、数据、权重，并提供逐项及分块向量化残差计算。
直接测量各自“准备＋扫描”，在本次短 CPU 数据上，1/4/16/256 个距离的批量优化倍率为
0.265/0.503/1.16/14.74。**只算一次会更慢；足够多的距离一起计算才摊薄准备成本。**
即使已有系数，逐个 Python 调用的开销仍可抵消算法收益；不能把一次批量计算的倍率说成
单次 FEW 波形加速。上述计时不包含双方共同排除的 FEW、TDI、FFT 和数据创建。

另一个发现来自部署层：官方 lisaanalysistools CPU wheel 导入时执行本机不支持的 AVX-512
指令而 SIGILL。隔离环境中用 x86-64 CPU-only 重建后导入和真实响应都通过；记录了二进制
前后哈希和 GDB 指令证据。不能只根据包名的“CPU”标签假定所有 x86 CPU 都能运行，
也不能把发布包故障误判为 FEW 数值或 CUDA 故障。

推导依据参见 [Thrane–Talbot 附录 C.3/C.4](https://arxiv.org/html/1809.02293v7)，
运行证据、保留的初始失败与计时修正见[实验索引](../collaboration/linux/EXPERIMENTS_2026-09-11.md)。

## 参考文献与代码证据

网页核对日期：2026-09-11 CST。源码定位以本文开头的提交为准；相对链接供当前工作区阅读。

[^evidence]: 本项目提交 `b28696b4`：[混合计算原始计时与内存](https://github.com/YaxinCui/FastEMRIWaveforms/blob/b28696b424043a12b49382a20c03c54b11c03cd2/collaboration/linux/mixed32_kerr_probe.json)，[可复现实验脚本](https://github.com/YaxinCui/FastEMRIWaveforms/blob/b28696b424043a12b49382a20c03c54b11c03cd2/collaboration/linux/mixed32_kerr_probe.py)。同分支结果，不能等同本分支已实现。
[^sweep]: 本项目提交 `b28696b4`：[七组精度扫查](https://github.com/YaxinCui/FastEMRIWaveforms/blob/b28696b424043a12b49382a20c03c54b11c03cd2/collaboration/linux/mixed32_accuracy_sweep.json)。
[^profile]: 本项目提交 `b28696b4`：[阶段计时](https://github.com/YaxinCui/FastEMRIWaveforms/blob/b28696b424043a12b49382a20c03c54b11c03cd2/collaboration/linux/pipeline_stage_probe.json)，[带同步的诊断脚本](https://github.com/YaxinCui/FastEMRIWaveforms/blob/b28696b424043a12b49382a20c03c54b11c03cd2/collaboration/linux/pipeline_stage_probe.py)。
[^zero-copy]: 本项目提交 `b28696b4`：[FP64 零拷贝实验](https://github.com/YaxinCui/FastEMRIWaveforms/blob/b28696b424043a12b49382a20c03c54b11c03cd2/collaboration/linux/fp64_zero_copy_probe.json)；本分支 [ampinterp2d.py](../src/few/amplitude/ampinterp2d.py)，`AmpInterp2D.__call__` 的模式 gather/flatten。
[^amplitude]: 本分支 [ampinterp2d.py](../src/few/amplitude/ampinterp2d.py)，`AmpInterpKerrEccEq` 构造、固定自旋要求及 `get_amplitudes` 的相邻面线性插值。
[^trajectory]: 本分支 [dopr853.py](../src/few/trajectory/dopr853.py)，`DOPR853` 步进工作数组；[integrate.py](../src/few/trajectory/integrate.py)，单系统 RHS wrapper 与积分状态。
[^flux]: 本分支 [flux.py](../src/few/trajectory/ode/flux.py)，`interpolate_flux_grids`、`evaluate_rhs`、固定参数与通量样条调用。
[^ode]: 本分支 [ODEBase](../src/few/trajectory/ode/base.py) 与 [inspiral.py](../src/few/trajectory/inspiral.py)，子类化、`evaluate_rhs`、额外参数接口。
[^selector]: 本分支 [modeselector.py](../src/few/utils/modeselector.py)，`sensitivity_fn`、模式 SNR 估计、`argsort` 与 `sort`、正负模式规则。
[^spline]: 本分支 [interpolatedmodesum.py](../src/few/summation/interpolatedmodesum.py)，公共 `t` 的复制；[interpolate.cu](../src/few/cutils/interpolate.cu)，样条三对角矩阵与 RHS。
[^kernel]: 本分支 [interpolate.cu](../src/few/cutils/interpolate.cu)，相位/正负模式求和、区间 stream 与同步生命周期。
[^roman]: 本分支 [romannet.py](../src/few/amplitude/romannet.py)，`RomanAmplitude` 文档与前向路径。
[^lifecycle]: 本分支 [waveform/base.py](../src/few/waveform/base.py) 的调用状态与批处理；[integrate.py](../src/few/trajectory/integrate.py) 的状态复位；[files/manager.py](../src/few/files/manager.py) 的下载/校验生命周期。改进建议属于本报告的源码推论，不等于已经实现。
[^tests]: 本分支 [test_amplitudes.py](../tests/test_amplitudes.py)，Kerr 点参考循环外的最终断言；[run-tests.yml](../.github/workflows/run-tests.yml)，`FEW_WITH_GPU=OFF` 的 CI 配置。先核对参考再修复，不能单纯收紧/放宽阈值。
[^kerr]: [The Fast and the Frame-Dragging: Efficient Waveform Generation for Eccentric Equatorial Inspirals into Rapidly-Spinning Black Holes](https://arxiv.org/html/2506.09470v2)，2025，III.2.2：自旋插值、系数表大小与设计。本文的具体访问裁剪建议来自本地代码和元数据分析。
[^systematics]: Khalvati 等，[Systematic errors in fast relativistic waveforms for Extreme Mass Ratio Inspirals](https://arxiv.org/html/2509.08875v1)，2025，III.1、III.2/Table 1：圆 Kerr 通量的截断与插值。其剪枝谱方法可降低自身耗时，但表中样条仍更快，不能宣称谱方法必然提速。
[^nit]: Lynch 等，[Fast inspirals and the treatment of orbital resonances](https://arxiv.org/html/2405.21072v2)，2024，共振附近 Full/Partial NIT；比较基线和 toy self-force 限制需保留。
[^self-force]: Barack、Pound，[Self-force and radiation reaction in general relativity](https://arxiv.org/abs/1805.10385)，2019，自作用力、双时间尺度和后绝热展开综述。
[^geodesic]: Fujita、Hikida，[Analytical solutions of bound timelike geodesic orbits in Kerr spacetime](https://arxiv.org/abs/0906.1420)，2009，束缚 Kerr 轨道及频率。
[^hdf5]: h5py 官方 [Datasets](https://docs.h5py.org/en/stable/high/dataset.html)，切片、contiguous/chunked 布局、压缩与 `read_direct`。
[^cupy]: CuPy 官方 [Memory Management](https://docs.cupy.dev/en/stable/user_guide/memory.html)，used/reserved、内存池和额外 CUDA 分配。
[^cuda]: NVIDIA 官方 [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)，性能剖析、Amdahl、数据传输。参考原则，不代表本机工具链已经升级至文档最新版。
[^graphs]: NVIDIA 官方 [CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)，重复工作提交与图建立/执行生命周期；具体 API 需匹配本机 Toolkit。
[^fd]: Speri 等，[Fast and Fourier: extreme mass ratio inspiral waveforms in the frequency domain](https://www.frontiersin.org/journals/applied-mathematics-and-statistics/articles/10.3389/fams.2023.1266739/full)，2023，Table 2 与多音驻相方法；本分支 [FDInterpolatedModeSum](../src/few/summation/fdinterp.py)。
[^response]: Katz 等，[Assessing the data-analysis impact of LISA orbit approximations using a GPU-accelerated response model](https://arxiv.org/abs/2204.06633)，2022；[官方 lisa-on-gpu 工程](https://github.com/mikekatz04/lisa-on-gpu)。
[^binning]: Zackay、Dai、Venumadhav，[Relative Binning and Fast Likelihood Evaluation for Gravitational Wave Parameter Estimation](https://arxiv.org/abs/1806.08792)，2018，GW170817 示例。
[^mode-binning]: Leslie、Dai、Pratten，[Mode-by-mode relative binning: Fast likelihood estimation for gravitational waveforms with spin-orbit precession and multiple harmonics](https://arxiv.org/abs/2109.09872)，2021；用于多模式方法参照，不是 EMRI 验证。
[^roq]: Canizares 等，[Accelerated gravitational-wave parameter estimation with reduced order modeling](https://arxiv.org/abs/1404.6284)，2015，非自旋双中子星九维推断约 30× 的实例。
[^delayed]: Banterle 等，[Accelerating Metropolis-Hastings algorithms by Delayed Acceptance](https://www.aimsciences.org/article/doi/10.3934/fods.2019005)，2019，两阶段接受规则与效率条件。
[^autodiff]: JAX 官方 [Advanced automatic differentiation](https://docs.jax.dev/en/latest/advanced_autodiff.html)，JVP/VJP 与自定义导数。
[^ripple]: Edwards 等，[ripple: Differentiable and Hardware-Accelerated Waveforms for Gravitational Wave Data Analysis](https://arxiv.org/abs/2302.05329)，2023；其 CBC/Fisher 性能不直接外推 FEW。
[^sbi]: Cole 等，[Sequential simulation-based inference for extreme mass ratio inspirals](https://arxiv.org/abs/2505.16795v2)，2025 首发、2026 修订；非自旋十一维搜索及一维边缘提议。
[^accuracy]: Lindblom、Owen、Brown，[Model Waveform Accuracy Standards for Gravitational Wave Data Analysis](https://arxiv.org/abs/0809.3844)，2008，探测器加权误差标准。
[^accuracy-use]: Lindblom，[Use and Abuse of the Model Waveform Accuracy Standards](https://arxiv.org/abs/0907.0457)，2009，误差标准的适用条件。
