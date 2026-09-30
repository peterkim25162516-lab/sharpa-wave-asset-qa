# Sharpa WaveSimParity｜面试知识库 QA

> 用途：机器人仿真、具身智能、Benchmark/Evaluation、研究工程与规划控制岗位的项目面试。
>
> 使用方法：先熟练掌握开场介绍和标有“必问”的问题，再练习每题的“继续追问”。不要逐字背诵；必须能用自己的话解释公式、实验设计、失败记录和个人边界。
>
> 事实边界：这是独立、非官方、simulation-only 项目。没有机械手真机、传感器/执行器硬件实验、Sim2Real、安全验证或官方背书；A800 仅是运行 OVPhysX 的仿真计算资源。项目没有公开 push、remote 或 PR。`DIVERGENT` 不是官方 bug，`INCONCLUSIVE` 不是实验失败。

## 0. 面试开场：先这样向面试官介绍项目

### 0.1 推荐 90 秒版

> 我做了一个非官方、纯仿真的 Sharpa Wave 灵巧手资产 QA 与跨模拟器验证项目。它解决的是：同一模型在 URDF、MJCF 和 USD 中都能加载，也不代表关节、坐标系和动力学语义一致。
>
> 第一阶段围绕固定版本的 60 个模型入口检查结构、网格、限位和 FK，得到 497 PASS、0 FAIL。第二阶段使用统一场景和映射，让 MuJoCo 与只运行物理后端的 kit-less OVPhysX 做同题测试；左右手分别运行后累计覆盖 44 个关节和 10 个末端 frame。最终封存 campaign 的每个 case 都在新进程运行，并接受重复性、步长减半和哈希校验；此前失败尝试则作为 excluded evidence 保留。
>
> 无接触 small-step 与 sine/chirp 的末端位置最大差约 2.11 和 2.32 毫米，高于预先冻结的 2 毫米阈值，因此保留 DIVERGENT 结论。最小接触 C0 的 32 个 case 都完成，但释放时序仍对步长敏感，所以停在 VALID / INCONCLUSIVE。项目价值不是判定哪个引擎更真实，而是建立一套可复核、能拒绝过度结论的验证流程；没有机械手真机、Sim2Real 或官方背书。

### 0.2 只有 30 秒时这样说

> 这是一个非官方、纯仿真的 Sharpa Wave 验证项目。我先检查 60 个 URDF、MJCF 和 USD 模型入口，再让 MuJoCo 与 kit-less OVPhysX 按同一协议执行实验。结果稳定发现约 2.1 至 2.3 毫米的末端位置超限，接触释放时序也对步长敏感。我用预注册阈值和可复核证据保留 DIVERGENT / INCONCLUSIVE，没有把它说成官方 bug 或真机结论。

### 0.3 面试官问“你具体做了什么”时

> 这是一个由我确定研究方向、范围和停止边界，并借助 AI coding agent 推进实现的个人项目。我的可核验责任包括固定研究对象、确认阶段验收条件、授权和组织正式运行、检查结果是否满足协议，以及决定哪些结论可以说、哪些必须保留为 DIVERGENT 或 INCONCLUSIVE。代码和文档有较多 AI 辅助，因此我不会声称所有模块都是逐行手写；我只认领自己能够打开源码、说明输入输出、关键约束、失败路径和取舍的部分。Sharpa 模型本身来自公开上游，我没有参与机械手硬件设计，也不把上游资产或已有 PR 说成自己的成果。

### 0.4 开场后最值得主动递出的三个钩子

面试官通常会顺着下面三个点追问。介绍时可根据岗位只突出其中一个：

1. **评测/Benchmark 岗：**“阈值和判断顺序在看结果前冻结，失败 run 也进入纠错记录。”
2. **仿真/控制岗：**“joint 和 orientation 过线，但 distal-frame position 略超 2 mm；诊断指向动态关节响应而非 FK 几何。”
3. **工程/平台岗：**“每个 case 都用 fresh process，结果由离线 finalizer 从 exact inventory 和内容哈希生成。”

### 0.5 一眼记住的结果卡

| 阶段 | 规模 | 正式结果 | 面试一句话 |
|---|---:|---|---|
| v0.1 资产 QA | 60 入口、516 检查 | 497 PASS / 6 KNOWN / 13 WARN / 0 FAIL | 标准手跨格式 FK 几乎重合 |
| Gate 0 | 32/32 | DIVERGENT | small-step 末端位置 2.109/2.111 mm，略超 2 mm |
| R1 readback | 44/44 joints、4 fresh process | READBACK_VALID / 四类 UNMAPPABLE | 读到数值不等于语义可映射 |
| Freeze B | 24/24 | VALID / INCONCLUSIVE | 摩擦输入灵敏度不够一致，不能硬下结论 |
| Trajectory T1 | 32/32、96,000 integration steps | VALID / DIVERGENT | sine/chirp 四格位置均约 2.321–2.324 mm |
| Contact C0 | 32/32 | VALID / INCONCLUSIVE | 唯一失败是释放事件 dt 偏移 5.5 ms > 4 ms |

---

## 1. 面试答题原则

### 1.1 每道项目题都按四层回答

1. **问题：**当时要排除什么风险？
2. **设计：**你冻结了哪些输入、对照和判断规则？
3. **证据：**用哪些数字证明执行有效或结果超限？
4. **边界：**这个结果不能推出什么？

一个合格答案不能只说“我用了 MuJoCo 和 PhysX”，也不能只背最终数字。面试官真正想确认的是：你是否知道输入、输出、变量、阈值和失败路径之间的关系。

### 1.2 数字要同时说口径

例如不要只说“误差是 2.11 mm”，而要说：

> 在固定资产、position-mode 控制、small-step 场景和冻结采样规则下，左右手 distal-link frame origin 的最大位置差为 2.109 和 2.111 mm，略高于 2.000 mm 阈值；joint 和 orientation 指标仍在阈值内。

### 1.3 不确定结论不要说成失败

- `COMPLETED`：程序跑完；
- `VALID`：证据满足协议；
- `WITHIN_TOLERANCE / DIVERGENT / INCONCLUSIVE`：科学判断；
- `pass_ready`：是否满足宣告“通过”的全部条件。

这些维度不能混在一起。`VALID / INCONCLUSIVE` 完全合理，表示数据可信，但不能给出更强结论。

### 1.4 时间有限时先练这 12 题

按顺序先练：Q01 项目问题、Q03 上游与个人贡献、Q07 技术难点、Q10 硬件边界、Q11 资产格式、Q25 v0.1 限制、Q28 joint mapping、Q58 离线 finalizer、Q69 小幅超限、Q78 归因边界、Q107 C0 结论、Q117 AI 辅助边界。其余问题作为追问库，不需要一次背完。

---

## 2. 项目定位与个人边界 QA

**Q01｜这个项目到底解决什么问题？【必问】**

**参考回答：** 它解决“同一机器人在多种描述格式和物理引擎之间是否真的一致”这个问题。第一层检查资产结构和运动学是否自洽，第二层让 MuJoCo 与 kit-less OVPhysX 在同一输入合同下运行，第三层再用诊断实验缩小差异来源。重点是把“能加载”“能跑完”“证据有效”和“数值在容差内”分开。

**继续追问：** 为什么单个模型在两个软件里都能动，仍不足以证明一致？

**Q02｜为什么选择 Sharpa Wave？**

**参考回答：** 上游公开提供 URDF、MJCF 和 USD/USDA 等多种模型入口，同一机械手还有左右手、安装形式和控制配置，天然适合做跨格式资产 QA。标准单手 22 个标量关节，规模足以暴露映射、控制和末端误差，又适合做可控的最小实验。

**边界：** 这是对固定上游 commit 的研究，不代表之后所有版本。

**Q03｜你的成果与 Sharpa 官方资产分别是什么？【必问】**

**参考回答：** 机器人模型和品牌属于上游；我的项目成果是独立验证工具、实验协议、adapter、比较指标、证据链和报告。3 个 dual-hand MJCF 入口各产生 missing-mesh 和 load-failed 两项，共 6 个 KNOWN outcome；这些网格问题已有上游 PR 记录，所以我不冒充首次发现。

**不要说：**“我开发了 Sharpa Wave 机械手”或“我修复了官方模型”，除非将来确有公开贡献证据。

**Q04｜这是官方合作项目吗？**

**参考回答：** 不是。项目明确标为 independent、unofficial、simulation-only，没有 Sharpa、MuJoCo、NVIDIA 或 Isaac Sim 的官方背书。

**Q05｜为什么先做 CPU-first，而不是直接启动大型模拟器？**

**参考回答：** 大量高价值问题其实不需要 GPU：XML 解析、网格路径、树结构、限位、执行器绑定和 FK 都可在 CPU 上快速、确定性地检查。先排除资产层错误，可以避免把简单的文件问题误判成物理引擎差异，也让 CI 成本更低。

**Q06｜这个项目当前算完成了吗？**

**参考回答：** 已冻结的范围已经完成到 Contact Gate C0，每个阶段都有正式标签和可复核证据。如果目标是建立首轮资产 QA、无接触 parity 和最小合成接触验证，当前已闭环；原生指尖、双手交互和浮动基座属于新阶段，不能倒回去改写已封存结论。

**Q07｜项目最大的技术难点是什么？【必问】**

**参考回答：** 最大难点不是让两个仿真器都启动，而是建立真正可比的输入和可信的证据：同一关节语义、同一控制目标、同一时间网格、同一采样时刻、同一指标定义，同时还要证明没有缓存、旧证据、路径或运行时默认值混入结果。

**Q08｜这个项目对业务或团队有什么价值？**

**参考回答：** 它可作为模型接入和升级前的回归门：先快速发现丢网格、限位漂移、映射错误，再用少量冻结场景观察后端升级是否改变动态响应。它的目标是减少资产、控制和求解器问题混杂造成的排查成本；目前尚无外部团队采用、节省工时或 ROI 数据，因此这是潜在工程价值，不是已验证业务收益。

**Q09｜为什么结果不是全部 PASS，仍然值得写进简历？**

**参考回答：** 评测项目的价值不是把阈值调到全部通过，而是构造能证伪的协议并忠实保留结果。这里既得到跨格式 FK 高度一致的正结果，也测到可重复的毫米级动态差异和接触时序敏感性，还明确了哪些参数目前不可映射。这比只有一张“成功运行”截图更有研究和工程价值。

**Q10｜硬件和 Sim2Real 做了吗？【高风险】**

**参考回答：** 没有机械手真机、传感器/执行器硬件实验、硬件 ground truth、Sim2Real 或安全验证。A800 只提供 OVPhysX 仿真计算。当前只能讨论固定软件版本和仿真协议下的可重复差异，不能判断哪个引擎更接近真实机械手。

---

## 3. 资产格式与 v0.1 QA

**Q11｜URDF、MJCF、USD/USDA 分别是什么？【必问】**

**参考回答：** URDF 常用于 ROS 生态，核心是 link-joint 树和几何/惯性描述；MJCF 是 MuJoCo 的模型语言，能表达更直接的仿真、执行器和接触设置；USD 是可组合的场景描述系统，USDA 是其文本表示，PhysX schema 可在 stage 上表达物理属性。它们的抽象层不同，所以不能只按字段名机械比较。

**Q12｜USDA 和 resolved USD stage 有什么区别？**

**参考回答：** 单个 USDA 文本只是一个 layer 的 authored opinions；resolved stage 还包含 sublayer、reference、payload、variant 和 composition 后的最终值。v0.1 只做文本层检查，正式 OVPhysX 阶段才打开 canonical outer stage 并读取解析后的物理 schema。

**Q13｜“文件能解析”为什么不等于“资产正确”？**

**参考回答：** XML 合法只说明语法成立。模型仍可能丢网格、路径大小写不兼容、关节形成环、限位反向、执行器绑错 joint，或者不同格式用不同坐标约定。QA 必须覆盖语法、引用、拓扑、语义和运行时多个层次。

**Q14｜v0.1 扫描了多少内容？**

**参考回答：** 固定上游版本下共 60 个入口：15 URDF、15 MJCF、30 USDA；形成 15 个配置组。最终共有 516 项检查，结果是 497 PASS、6 KNOWN、13 WARN、0 FAIL、0 SKIP。

**Q15｜资产发现为什么要确定性？**

**参考回答：** 同一目录每次必须得到相同排序、分组和标识，否则报告差异可能只是文件枚举顺序变化。发现模块会规范化 side、mounting、control 和 floating-base 等维度，并拒绝模糊或重复入口。

**Q16｜静态 QA 主要查什么？**

**参考回答：** 包括重复名称、link/body 父子树、root 与 cycle、关节数量和类型、axis、limit、effort/velocity 元数据、mesh 是否存在、路径大小写、actuator target、跨格式 joint set，以及 position 与 MIT-mode overlay 的一致性。

**Q17｜为什么要检查 mesh 路径大小写？**

**参考回答：** Windows 文件系统常常大小写不敏感，Linux 通常敏感。一个在本机能打开的路径，换到 Linux 服务器可能直接失败。路径大小写检查是资产可移植性问题，不只是代码风格。

**Q18｜FK 是什么，为什么适合做跨格式检查？【必问】**

**参考回答：** FK 从关节状态沿运动链组合刚体变换，计算末端 frame 的位置和姿态。它不依赖摩擦、质量或求解器，所以能比较纯几何和关节定义。如果 URDF 与 MJCF 在大量合法姿势下 FK 重合，说明基本轴、零位、父子关系和几何偏置高度一致。

**Q19｜FK 检查怎样避免“挑几个好看的姿势”？**

**参考回答：** 使用固定随机种子生成 256 组合法关节姿势，在关节限位内采样，并比较共同存在的 distal frames。固定种子保证复现；多姿势覆盖比只看零位更容易暴露轴方向、offset 或 joint-order 问题。

**Q20｜v0.1 的 FK 结果是多少？**

**参考回答：** 标准左右手 URDF—MJCF 的最大位置差约 `0.00007073 mm`，最大姿态差约 `0.00008585°`。这远小于后续毫米级动态差异，因此基本几何翻译错误不是首要嫌疑。

**Q21｜wrist/flange 的 29 mm 左右偏移为什么是 WARN，不是 FAIL？**

**参考回答：** 偏移在相关变体中稳定且有结构规律，更像 base/frame convention 不同，而不是随机损坏。意图未被上游权威说明前，正确做法是公开为 WARN，并限制结论，而不是擅自对齐后宣布通过或直接判错。

**Q22｜KNOWN、WARN、FAIL 有什么区别？**

**参考回答：** KNOWN 是已有明确来源或上游记录的问题；WARN 是可观察差异，但证据不足以判错；FAIL 是当前规则下的新硬失败。分类同时保护科学边界和贡献归属。

**Q23｜为什么 497 个 PASS 不能淹没 6 个 KNOWN 和 13 个 WARN？**

**参考回答：** 聚合状态不是多数投票。一个关键路径或语义警告可能比数百个简单通过项更重要，因此报告保留各类计数和具体 finding，不把高通过率包装成“资产完全无问题”。

**Q24｜MuJoCo smoke test 能证明什么？**

**参考回答：** 它能证明模型可 headless 加载、能完成有限步、状态是有限数、重复运行是否一致，并可测机器相关实时因子。它不能证明控制器质量、跨模拟器一致性或现实物理准确性。

**Q25｜v0.1 最大的限制是什么？**

**参考回答：** USDA 支持是文本层，不解析完整 composition，也不运行 PhysX；FK 是 fixed-base、抽样式检查；MuJoCo smoke 只验证基本可运行性。后续 Gate 0 正是为补 resolved USD、控制路径和动态比较而设计。

---

## 4. 仿真、控制与数学基础 QA

**Q26｜joint 和 DoF 有什么区别？**

**参考回答：** joint 是两个刚体之间的运动约束，DoF 是独立运动坐标的数量。revolute joint 通常 1 DoF，ball joint 有 3 个旋转 DoF，free joint 有 6 个物理 DoF。这里标准单手恰好是 22 个标量 joint，所以可称 22 DoF；44/44 是左右手 campaign 总覆盖，不是同时仿真一个 44-DoF 双手系统。

**Q27｜为什么先用 fixed base？**

**参考回答：** 固定基座去掉根部 6 DoF、惯性漂移和额外接触，让第一次跨引擎比较更可解释。浮动基座还会引入 quaternion 状态、根部质量惯量、坐标顺序和积分差异，适合后续单独预注册。

**Q28｜canonical joint mapping 只是改名字吗？【必问】**

**参考回答：** 不是。映射至少包含 backend index、scope、sign、offset 和 unit，统一关系可写成 `q_canonical = sign × q_backend + offset`。还要检查预期集合完全相等，没有漏项、额外项、重复 name/index 或模糊绑定。

**Q29｜position mode 是不是直接把 qpos 写到目标值？**

**参考回答：** 不是，而且两个后端的执行路径并不相同。OVPhysX 侧用显式 IdealPD 计算 `τ = Kp(q_target-q) + Kd(v_target-v) + τ_ff`，再按 effort limit 截断；MuJoCo 侧把 canonical position target 写入 `data.ctrl`，由编译后的 MJCF position actuator 执行。项目统一的是目标与可核验合同，不是宣称两边内部控制器实现相同。

**Q30｜怎样证明两个后端真的收到了同一个目标？**

**参考回答：** canonical target 由共用场景模块生成，adapter 只负责转换到后端顺序；MuJoCo 把目标写入编译后 position actuator 的 `data.ctrl`，OVPhysX 则额外做 target readback。Gate 0 的 16/16 个 OVPhysX case 最大 target 误差为 `0 rad`，PD formula error 与 clip-validation error 均为 `0 Nm`，clip count 为 0。这证明目标输送符合已记录合同，不等于证明两个后端内部控制语义完全相同。

**Q31｜Kp、Kd、passive damping 是一回事吗？**

**参考回答：** Kp 对位置误差产生恢复力矩，controller Kd 对目标与实际速度误差起作用；passive damping 通常是模型层的 `-c·v`。它们计算层、目标速度、限幅和作用顺序可能不同，不能简单相加后就宣称跨引擎等价。

**Q32｜joint dry friction 和 contact friction 有什么区别？**

**参考回答：** joint dry friction 是关节内部对相对运动的库仑/静摩擦类阻力；contact friction 是两个表面切向接触产生的阻力。MuJoCo `frictionloss` 与 PhysX legacy joint friction 即使同名同数值，也未必同单位或同作用语义。

**Q33｜armature 是什么？**

**参考回答：** armature 是折算到广义坐标侧的附加执行器或转子惯量，会影响加速度、高频响应和数值条件。它不是 link mass，也不是 damping。

**Q34｜质量一样，运动为什么仍可能不同？**

**参考回答：** 总质量相同不代表质心位置和惯性张量相同。旋转动力学取决于质量分布、主惯量和主轴方向；控制器、关节惯量、摩擦与 solver 也都会改变响应。

**Q35｜为什么 quaternion 不能直接逐元素相减？**

**参考回答：** `q` 与 `-q` 表示同一个旋转，直接相减可能把完全相同的姿态算成很大误差。项目统一 `qwxyz`，先做符号对齐，再用稳定 geodesic angle 比较旋转。

**Q36｜末端 frame position 指的是真实指尖接触点吗？【高风险】**

**参考回答：** 不是。Gate 0/T1 的 `*_DP` 是 distal-phalanx link origin。C0 的合成球还相对 `index_DP` 有固定局部偏移。面试时必须说“末端 link frame origin”，不能偷换成真实指尖表面。

**Q37｜为什么 MuJoCo 记录 qpos 后还要刷新 kinematics？**

**参考回答：** `mj_step` 后 qpos 已更新，但某些派生的 `xpos/xquat` 可能仍对应旧状态。权威实现会在采样 frame pose 前调用 `mj_kinematics`，并用 sampled qpos 独立重放，防止把采样时序错误误判成跨引擎差异。

**Q38｜repeatability 和 dt-halving 分别回答什么？【必问】**

**参考回答：** repeatability 问“相同 dt 和输入，从新进程再跑一次是否一致”；dt-halving 问“把步长减半但总物理时长保持不变，结论是否仍稳定”。前者排查随机和状态污染，后者排查离散化敏感性，两者不能互相替代。

**Q39｜为什么更小 dt 不一定所有指标都单调变好？**

**参考回答：** controller update、contact solver、约束稳定化、iteration、事件量化和浮点采样都会随 dt 变化。更细步长通常降低部分积分误差，但不保证所有离散事件时刻或最大值单调收敛。

**Q40｜N 个 interval 为什么有 N+1 个 sample？**

**参考回答：** sample 0 是初始状态；每推进一个 interval 再得到一个新状态，所以 N 次推进产生 N+1 个状态样本。C0 基础 dt 是 3500 interval/3501 sample，减半 dt 是 7000/7001。

**Q41｜怎样比较 frame orientation？**

**参考回答：** 先把两边 quaternion 统一顺序和符号，再计算相对旋转的 geodesic angle。零角附近使用数值稳定形式，避免 `acos(dot)` 在 dot 接近 1 时把同一姿态报告成伪小角。

**Q42｜跨模拟器结果不同，最常见原因有哪些？**

**参考回答：** 资产 composition、关节 sign/offset、控制器实现、target 更新时刻、passive damping、dry friction、armature、质量惯量、solver、积分器、迭代次数、接触定义和采样时序都可能贡献。项目通过分层 QA 和受控干预逐步排除，而不是看到曲线不同就猜一个原因。

---
## 5. 系统架构与可复现证据 QA

**Q43｜整个系统的数据流是什么？【必问】**

**参考回答：** 主链路是 `manifest → deterministic case matrix → mapping/adapter → fresh worker → raw evidence → offline finalizer → private bundle + public report`。协议层定义什么输入合法，执行层适配两个后端，证据层证明来源和完整性，分析发布层计算指标并控制结论边界。

**Q44｜为什么不直接写两个脚本，各导出一个 CSV 再比较？**

**参考回答：** 两个脚本很容易在关节顺序、单位、目标生成、采样时刻和默认参数上悄悄分叉。最后即使有两份 CSV，也不能证明两边做的是同一道题。共同协议让 backend 只负责执行，把比较语义放在共享层。

**取舍：** 这增加了 schema、hash 和证据管理成本，但换来可核验性。

**Q45｜canonical manifest 是什么？【必问】**

**参考回答：** 它像正式实验前封好的试卷，固定资产与源码身份、hand、scenario、初始状态、目标轨迹、dt、repeat、阈值、完整矩阵和允许声称的结论。finalizer 以 manifest 生成的 expected set 为准，不以“目录里实际有什么”反推实验范围。

**Q46｜为什么 manifest 要在看结果前冻结？**

**参考回答：** 防止 researcher degrees of freedom。阈值、窗口或排除规则如果在看过结果后调整，就可能把“不通过”人为改成“通过”。需要修改协议时应新建版本并重跑，而不是静默覆盖。

**Q47｜为什么同时需要 file hash 和 semantic hash？**

**参考回答：** file hash 判断原始字节是否相同；semantic hash 对 canonical JSON 计算，判断解析后的实验含义是否相同。换行或键顺序可能改变文件字节但不改变语义，所以两种哈希回答不同问题。

**Q48｜case matrix 如何防止漏跑或挑结果？**

**参考回答：** case 由 `backend × hand × scenario/condition × dt × repeat` 按固定顺序展开，case ID 由字段派生。finalizer 比较 expected set 与 actual set，缺 case、重复 case、未知 case 或 case ID 与字段不一致都会拒绝。

**Q49｜adapter 层的作用是什么？**

**参考回答：** adapter 隔离 MuJoCo 与 OVPhysX 的专用 API，只向上层提供统一的 load、mapping、set target、step、sample 和 close。runner 和 compare 只看 canonical joint/frame/target，不依赖后端对象。

**Q50｜统一接口会不会掩盖后端真实差异？**

**参考回答：** 有这个风险，所以只统一能够定义共同语义的量。后端特有、无法可靠映射的能力保留为 native evidence 或 `unavailable`，不能为了接口整齐伪造一个值。R1 的 `UNMAPPABLE` 就是主动保留这种边界。

**Q51｜如果 worker 写错目标或 mapping，系统怎么发现？**

**参考回答：** finalizer 不完全信任 worker，会从 manifest 重新生成 canonical target，核对 target digest、case ID、model path、dt、步数、sample 数、关节顺序、mapping、来源 hash 和终点 readback。测试还会注入伪造 readback、错误 waveform digest 和错序 qpos，确认系统 fail closed。

**Q52｜为什么使用 strict JSON/schema？**

**参考回答：** 为了阻止“未知字段被悄悄忽略”或 NaN 在 JSON 中流动。闭合 contract/schema 会拒绝 unknown/missing field、非法 enum、NaN/Infinity 和错误类型；采用 duplicate-key-aware loader 的 evidence、runner、readback 等入口还会显式拒绝重复 key。不是所有通用 JSON Schema 都能在对象解析后发现重复 key。

**Q53｜为什么每个 case 都要 fresh process？【必问】**

**参考回答：** 模拟器可能保留 singleton、插件 registry、CUDA context、import cache、上一个 stage、controller 或 contact sensor 状态。新进程让每个 case 有独立初始化、命令、日志和退出码，减少顺序依赖。

**边界：** fresh-process exact repeat 是强工程信号，不是现实正确性的证明。

**Q54｜provenance 链为什么不止一个 Git commit？**

**参考回答：** commit 只描述本项目源码。完整链还要包含上游资产 commit/tree/content hash、source tree/archive、manifest/schema、依赖与 runtime、worker、case、进程身份、payload 和 bundle root。否则无法回答“到底用哪份资产和环境跑的”。

**Q55｜immutable source snapshot 与 sealed bundle 有什么区别？**

**参考回答：** immutable snapshot 固定“运行前的源码输入”，sealed bundle 固定“运行后的证据输出”。前者防执行时代码漂移，后者让事后修改可被检测，解决的是两个时间点的问题。

**Q56｜bundle root 怎么计算？**

**参考回答：** 对每个 payload 记录规范化相对路径、字节数和 SHA-256，按路径排序并用长度 framing 聚合为一个根哈希。这样 root 代表逻辑文件树，不受 ZIP 时间戳或压缩实现影响。

**Q57｜只比较保存下来的 root 字符串够吗？**

**参考回答：** 不够。验证时必须重新枚举目录，重新计算每个文件和根哈希，同时拒绝额外、缺失、重复、symlink、reparse point、特殊文件、绝对路径和 `..`。否则攻击者可以同时改文件与清单。

**Q58｜为什么 finalizer 必须离线且不能启动模拟器？【必问】**

**参考回答：** finalizer 的职责是验证已经采集的证据。如果缺 case 时还能调用 GPU 顺手补跑，就无法区分原始实验与事后修补。离线 finalizer 只做 copy-first、inventory/schema/provenance 校验、指标计算和公私投影。

**Q59｜什么叫 fail-closed？**

**参考回答：** 遇到未知、缺失或矛盾信息时停止或降级为 `INCONCLUSIVE`，而不是猜一个默认值继续。例如 mapping 不完整、来源 hash 不符、出现非有限数、输出目录已存在、public projection 含禁止字段时，都不能生成“通过”。

**Q60｜为什么不能把几次旧 run 拼成一套完整实验？**

**参考回答：** 正式 bundle 内所有 case 必须共享同一 source、manifest、mapping、time grid 和 runtime 语义。旧曲线即使肉眼相同，只要协议版本不同就不能拼接。正确做法是新 revision、新 run ID、整矩阵重跑，旧证据标为 excluded。

**Q61｜corrigendum 能修什么，不能修什么？**

**参考回答：** 它可以公开解释分析或文档错误、说明旧 run 为什么被排除；不能暗中替换 raw trajectory、改变阈值、改输入或把失败 case 删掉。是否可以 analysis-only 修正，取决于是否改变原始实验语义。

**Q62｜为什么分 private bundle 和 public projection？**

**参考回答：** 私有层保存完整轨迹、日志、进程与 runtime 身份、路径和 backend-native 量；公开层只保留脱敏 aggregate、阈值、状态、覆盖数和可公开 hash。public JSON 用字段白名单生成，并扫描账号、主机、绝对路径、PID、session、GPU 标识和 secret。

**Q63｜937 passed 证明什么？又不证明什么？**

**参考回答：** 它说明 schema、mapping、runner、compare、bundle、launcher、finalizer、隐私和大量 mutation/failure path 有系统测试。它不是 937 个物理 rollout，不证明物理真实性，也不能替代封存的 GPU evidence。仓库虽配置 CPU CI workflow，但当前没有 remote，不能声称公共 CI 已跑绿。

**Q64｜为什么使用远端 GPU，却坚持无 Kit、renderer 和 camera？**

**参考回答：** 研究问题只需要 OVPhysX 物理后端，不需要渲染与传感器链。禁用 Kit/renderer/camera 能减少软件栈和硬件能力混杂，也符合所用 A800 没有 RT Core 的事实。准确名称必须是 kit-less OVPhysX，而不是完整 Isaac Sim。

---

## 6. Formal Gate 0 与差异归因 QA

**Q65｜Gate 0 的矩阵是什么？【必问】**

**参考回答：** 两个 backend、左右两只标准手、`zero_hold`/`small_step`/`gravity_settling` 三种场景、两个 fresh repeat；small-step 额外跑 base 与 halved dt，合计 32 个 formal case。全部 fixed-base、position mode、无地面和 intentional contact。

**Q66｜为什么正式版本必须加载 outer USDA，而不是 inner USD？**

**参考回答：** 外层 stage 包含上游 position-mode tuning 和 composition 后的最终属性。早期版本直接打开 converter inner layer，漏掉外层配置；所以权威实验固定 canonical outer stage，并把早期 run 只保留为诊断历史。

**Q67｜Gate 0 怎样证明执行有效？**

**参考回答：** 32/32 case、100% requested-step completion、44/44 joint、10/10 frame、0 个 non-finite run；两次 fresh repeat 的 joint/position/orientation 最大差均为 0；dt-halving 三类最大差均低于诊断阈值；OVPhysX 侧的 target 和显式 PD effort 路径也通过 readback、公式与限幅检查。

**Q68｜Gate 0 的阈值和结果是什么？【必问】**

**参考回答：** 冻结阈值是 joint `0.02 rad`、frame position `0.002 m`、orientation `0.05 rad`。zero-hold 和 gravity-settling 全部在阈值内；small-step 的 joint 与 orientation 也在阈值内，但左/右 frame position 为 `2.10924/2.11119 mm`，各超约 `0.109/0.111 mm`，所以是 `DIVERGENT / pass_ready=false`。

**Q69｜只超 0.11 mm，为什么不四舍五入成通过？【压力题】**

**参考回答：** 阈值在看结果前冻结，不能按结果修改评分标准。但报告必须同时说明超限量很小、其他指标通过，准确称为“小而稳定的协议内超限”，不能夸大成两个引擎差异巨大。

**Q70｜DIVERGENT 是否说明某个引擎有 bug？【必问】**

**参考回答：** 不说明。它只表示在固定资产、控制、采样和阈值下至少一项指标超限。原因可能来自模型语义、控制实现、被动参数、solver 或协议；没有额外真值和根因证据，不能归咎任何一方。

**Q71｜哪个模拟器更准确？**

**参考回答：** 当前无法回答，因为没有机械手真机、硬件实验或其他现实 ground truth。本项目评估的是一致性和可重复性，不是给两个后端排序。

**Q72｜干摩擦诊断是怎样设计的？**

**参考回答：** 固定正式 OVPhysX trace，只把 MuJoCo 编译后的 `dof_frictionloss` scale 设为 `1.0/0.5/0.0`，保持相同初态和目标。`1.0x` 必须先精确复现 formal MuJoCo trace，证明干预框架没有漂移，再观察随 treatment 变化的差异。

**Q73｜去掉 MuJoCo dry friction 后发生了什么？【高频】**

**参考回答：** 在选定 left small-step 轨迹上，joint max、frame-position max、orientation max 和 joint RMSE 分别下降约 `51.36%`、`59.67%`、`75.86%`、`89.48%`；formal peak 附近 thumb 指标下降更明显。这支持 dry friction 是 material contributor。

**Q74｜为什么仍不能说干摩擦是唯一根因？**

**参考回答：** treatment 只针对一条选定轨迹和 MuJoCo 一侧；归零后仍有残差，最差 joint gap 还转移到早期 pinky transient，thumb_DP 全局位置差仍约 `0.851 mm`。另外 PhysX friction 字段与 MuJoCo `frictionloss` 没有被证明等价。

**Q75｜total-viscous 调参说明了什么？**

**参考回答：** 把 controller Kv 与 passive damping 组合对齐后，pinky 注册窗口改善约 `82.36%`，thumb 窗口只改善 `4.22%`，而部分全局 frame 指标变差。所以一个共同粘性调参不能解释两类残差，结论是 mixed/inconclusive。

**Q76｜common-FK 为什么重要？【必问】**

**参考回答：** 它把几条记录到的 22-joint 轨迹送进同一个 fixed-base FK 模型，不再做 dynamics。选定 `left_thumb_DP` 注册窗口中未解释残差只约占 recorded gap 的 `0.0053%`，说明该窗口的 frame gap 几乎完全随 joint-state gap 传播，不是两套几何尺子又额外造成一大块差异。它属于已披露的 `post_hoc_mechanistic_analysis_not_formal_gate0`，并有先行的 single-step feasibility probe，不是 formal Gate 0 或可泛化的因果结论。

**Q77｜common-FK 能定位动力学根因吗？**

**参考回答：** 不能。它只证明给定 q 后几何传播自洽，无法解释 q 为什么不同。后者仍可能涉及摩擦、阻尼、armature、控制更新和 solver。

**Q78｜三项诊断合起来最稳妥的根因表述是什么？**

**参考回答：** 对已注册的 left small-step 诊断轨迹，证据更像 dynamics/control-state evolution 的差异，而不是标准手 FK 几何定义错误；MuJoCo dry friction 是该轨迹的重要贡献因素，单一 viscous retuning 不能完整解释。common-FK 又只是选定 `left_thumb_DP` 窗口上的 disclosed post-hoc mechanistic analysis。它们不能泛化到所有场景，底层唯一因果仍未被识别，也没有官方 bug 证据。

---

## 7. R1、Freeze B 与持续轨迹 T1 QA

**Q79｜R1 为什么只做 readback，不跑轨迹？**

**参考回答：** 它先回答“OVPhysX 的配置最终以什么形式进入 runtime，以及这些证据能否安全映射到既有 MuJoCo 语义”，避免把同名字段直接当作等价物。R1 在 authored、resolved USD 和 runtime-effective 三层读取 OVPhysX 参数；adapter 初始化/reset 可能写入零状态或零目标，但正式 trace 的 steps、commands、targets 和 samples 都是 0，因此它不是轨迹实验。

**Q80｜R1 的规模和结果是什么？**

**参考回答：** OVPhysX 左右手共 44/44 canonical joints、4 个 fresh process；mapping、solver evidence record 和 canonical numeric vector 全部精确重复，状态为 `READBACK_VALID`。把这些 evidence 与既有 MuJoCo 参数语义对照后，friction、armature、control/drive、solver/runtime 四类都保守判为 `UNMAPPABLE`。

**Q81｜UNMAPPABLE 是否等于“两边参数不同”？【必问】**

**参考回答：** 不是。它表示当前证据不足以建立可靠的一一语义映射。一个后端可能暴露 authored field，另一个暴露组合后的 runtime effect；即使都有数字，也不能放在同一列宣称等价。缺失值必须保留 `unavailable/null`，不能填 0。

**Q82｜为什么 readback exact 仍然不能宣告 parity？**

**参考回答：** exact repeat 只说明同一 runtime 暴露的数据稳定；它不证明两个引擎的字段含义相同，也不证明这些字段对 dynamics 的作用相同。R1 是 evidence-valid 的测量，不是参数等价证明。

**Q83｜Freeze B 把问题缩小成了什么？**

**参考回答：** 它不再问两个 friction 字段是否等价，而问“改变 OVPhysX legacy joint-friction input 后，跨引擎轨迹差异是否呈现足够大、足够一致的 sensitivity pattern”。这是更窄、可证伪的问题。

**Q84｜Freeze B 的矩阵是什么？**

**参考回答：** 共 24/24 case：8 个 MuJoCo unchanged controls 加 16 个 OVPhysX control/treatment；44/44 joint、10/10 frame、24 个 fresh process。repeatability exact，dt-halving、finite/sanity 和 16/16 target readback 全部通过。

**Q85｜Freeze B 为什么是 INCONCLUSIVE？【必问】**

**参考回答：** 预注册规则要求 8/8 sensitivity 都 `>=0.25` 才是 SUPPORTED，8/8 都 `<=0.10` 才是 NOT_SUPPORTED。实测 `5/8 <=0.10`、`3/8 >0.10`、`0/8 >=0.25`，两边条件都没满足，所以唯一合法结论是 `VALID / INCONCLUSIVE`。

**Q86｜Freeze B 与 MuJoCo 去摩擦结果矛盾吗？**

**参考回答：** 不矛盾。前者改变 OVPhysX legacy friction input，后者改变 MuJoCo 编译后的 `dof_frictionloss`，R1 已说明两者不能视为一一等价。不同 backend、字段和语义层的干预得到不同敏感性并不矛盾。

**Q87｜为什么还要做 T1 的 sine 和 chirp？**

**参考回答：** small-step 只覆盖一次短暂阶跃。sine 提供持续周期响应，chirp 让频率逐渐变化，可以检查差异是否只属于一个瞬态，还是在更长、更丰富的动态激励中仍存在。

**Q88｜T1 的有效性结果是什么？**

**参考回答：** 32/32 case、44/44 joint、10/10 frame，共 96,000 integration steps；0 non-finite，fresh repeat 三类最大差均为 0，dt-halving 全部通过；OVPhysX controller observation 完整，effort clip 为 0。日志中的 928 条 warning 均已分类，未分类 warning 为 0，fatal 为 0；不能把它缩写成“没有 warning”。

**Q89｜T1 的跨引擎结果是什么？【必问】**

**参考回答：** 左/右 × sine/chirp 四格的 joint max 约 `0.01977–0.01989 rad`，orientation max 约 `0.04816–0.04830 rad`，都在各自阈值内；frame position 为 `2.32091–2.32438 mm`，四格都高于 `2.000 mm`。因此是 `VALID / DIVERGENT`。

**Q90｜T1 比 Gate 0 多说明了什么？**

**参考回答：** 它说明毫米级 distal-frame position 超限不是 single-step 的偶然现象，在持续 sine/chirp 轨迹中也稳定出现。它加强了可重复观察，但仍不判断哪个后端更真实，也没有给出唯一根因。

---

## 8. Contact Gate C0 QA

**Q91｜为什么 C0 不直接从原生指尖软接触开始？【必问】**

**参考回答：** 原生 collision mesh、material、cooking、elastomer、多 contact patch 和过滤规则会同时引入许多变量。C0 先用解析明确的无摩擦 sphere-box normal contact，目标是验证接触观测、对照和时序管线是否稳定，不是验证真实指尖。

**Q92｜C0 的 fixture 是什么？**

**参考回答：** 在左右手 `index_DP` frame 上挂一个 massless sphere，局部中心 `[0.025,0,0] m`、半径 `0.005 m`；下方是 static box，顶面 `z=0.130 m`。gravity、friction、restitution 为 0，CCD 禁用，原生手部碰撞关闭，只允许目标 sphere-box pair。

**Q93｜massless overlay 怎样避免改变手指动力学？**

**参考回答：** 不能只看总质量，而要检查 overlay 前后 mass、COM、principal axes、mass-space inertia 和重建的 link-axis inertia tensor 都未改变。这样合成几何主要改变碰撞观测，不额外改变刚体惯量。

**Q94｜analytic signed gap 是什么？**

**参考回答：** `g = z_probe_center - z_box_top - sphere_radius`。`g>0` 表示分离，`g=0` 相切，`g<0` 表示解析几何重叠。它比两个 solver 各自的 penetration/contact-patch 字段更可移植。

**Q95｜C0 的动作怎样设计？**

**参考回答：** 总时长 7 秒：0–0.5 s 保持，0.5–3.0 s 平滑压入，3.0–4.0 s 保持，4.0–6.5 s 平滑释放，6.5–7.0 s 恢复。index chain 峰值是 MCP flexion 0.60、PIP 0.90、DIP 0.60 rad，其余关节为 0。

**Q96｜为什么使用 quintic smoothstep？**

**参考回答：** `h(u)=10u³-15u⁴+6u⁵` 在端点速度和加速度平滑衔接，比突然阶跃更少引入无关冲击和高频激励，更适合观察接触建立与释放。

**Q97｜sham 是什么，为什么必须有？【必问】**

**参考回答：** sham 保留同一几何、初态、command 和 gap 计算，只关闭目标 collision pair。它回答“没有接触阻挡时本来会怎么动”，可用来扣除 Gate 0/T1 已存在的无接触 baseline。

**Q98｜为什么 sham 的 gap 可以是负数？**

**参考回答：** sham 关闭了碰撞，合成球可以在解析几何上穿过盒面，所以 hold gap 为负是预期的。它反而证明 excitation 足够：如果接触开启，约束应产生可观测阻挡。

**Q99｜difference-in-differences 在这里怎么用？**

**参考回答：** 先在每个 backend 内算 contact 相对 sham 的额外效果：`g_contact-g_sham`、`q_contact-q_sham` 和 `R_contact·inverse(R_sham)`，再比较两个 backend 的 effect。它减去已有无接触 baseline，但只是配对控制结构，不是一般因果推断证明。

**Q100｜为什么不直接比较两个后端报告的 contact force？【高频】**

**参考回答：** 两个 solver 对 contact point、patch、impulse、force 和采样时刻的定义可能不同，硬比数值会制造虚假精度。C0 把 backend-native force 只用于生成 selected-pair active Boolean；正式可移植量是 analytic gap、事件和 contact-minus-sham effect。

**Q101｜pair_active 怎样定义？**

**参考回答：** 只观察明确选中的 `synthetic_index_probe__static_box`，OVPhysX pilot 后冻结为 `selected_pair_force_norm > 1e-4 N`。native magnitude 留在私有描述性证据，不进入跨引擎 force pass/fail。

**Q102｜为什么接触事件需要 debounce？**

**参考回答：** 离散求解在临界点可能出现单 sample 抖动。C0 要求状态连续 4 ms 才确认：base dt 需要 2 个 interval，halved dt 需要 4 个；确认后把事件回溯到连续 run 的首个 interval。

**Q103｜onset/release 的时间为何用 interval midpoint？**

**参考回答：** 离散采样只知道事件发生在 `[t_(k-1),t_k]` 内，不能假装知道连续时刻。保存 interval 并比较 midpoint，是对时间分辨率更诚实的表示。

**Q104｜C0 的矩阵为什么是 32 个 case？**

**参考回答：** `2 backend × 2 hand × 2 condition(contact/sham) × 2 dt × 2 repeat = 32`。所有 32 个 case 都在 fresh process 中完成，44/44 joint 和 10/10 frame 覆盖完整。

**Q105｜C0 哪些项目通过了？**

**参考回答：** 完整性、finite、runtime proof、mapping、admission/excitation、两次精确重复、pair bitset/event interval、初始与恢复分离、contact hold duty 1.0、sham/recovery duty 0，以及 joint/gap/orientation/hold-duty 四类 dt-halving 指标都通过。

**Q106｜C0 唯一失败项是什么？【必问】**

**参考回答：** OVPhysX release event midpoint：base 为 `4.605 s`，halved 为 `4.5995 s`，相差 `5.5 ms`，超过冻结的 `4.0 ms`。左右手和两个 fresh repeat 都复现；MuJoCo onset/release dt delta 为 1.5 ms，OVPhysX onset 为 0.5 ms。

**Q107｜为什么 C0 是 VALID / INCONCLUSIVE，而不是 DIVERGENT？【必问】**

**参考回答：** 冻结决策顺序是 admission → repeatability → dt-halving → cross-simulator。证据完整，所以 `VALID`；但 release-event dt gate 失败，必须在第三步停止，不能再应用正式跨后端阈值。因此 science 是 `INCONCLUSIVE`，`pass_ready=false`。

**Q108｜同一 dt 两次完全一样，为什么仍会 dt-sensitive？**

**参考回答：** 确定性和数值收敛是两个问题。同一离散网格可以每次得到完全相同事件；换更细网格后，solver、控制更新和状态切换落在不同 interval，事件时间仍可能系统移动。

**Q109｜C0 现在能说明两个后端接触表现不同吗？**

**参考回答：** 不能做正式 cross-simulator contact label。只能说管线有效，admission 与 repeatability 已通过，但 OVPhysX release event 对 dt 的敏感性超限，实验未越过 dt-halving 前置门，因此没有进入正式 cross-simulator classification。

**Q110｜C0 能外推到真实指尖、软材料和抓取吗？【高风险】**

**参考回答：** 不能。C0 是零摩擦、零回弹、零重力、合成球—盒单接触对；它只验证最小方法。原生 collision mesh、elastomer、多点接触、复杂物体和双手抓取都需要新协议。

---

## 9. 工程失败、复盘与行为面 QA

**Q111｜项目中最典型的一次失败是什么？【行为面必问】**

**参考回答：** 项目审计先后发现：早期 Gate 0 直接打开 converter 的 inner USD，漏掉 outer position-mode tuning；MuJoCo 在 `mj_step` 后记录的 Cartesian 派生量也没有随新 qpos 刷新。两次问题都意味着比较合同不成立。我据此接受旧 run 失效、保留其证据，并在 source/protocol 修正后要求完整重跑，把旧结果明确标为 excluded；除非我能现场解释对应源码，否则不把具体代码定位说成自己独立完成。

**Q112｜Contact C0 中修过哪些容易被忽略的问题？**

**参考回答：** 包括 halved dt 仍错误使用 base 的 3500 steps、浮点累加让末尾时间略超过 7 s，以及旧 quaternion 角度公式在相同姿态上产生极小伪差。最终改成由 duration/dt 派生步数、整数 canonical time grid 和稳定 quaternion geodesic，并重新验证完整矩阵。

**Q113｜为什么保留失败 run，而不是清理掉？**

**参考回答：** 失败证据说明问题何时被发现、为什么旧结论失效、阈值是否保持不变，也防止最终结果看起来像第一次就完美成功。删除失败历史会削弱审计性；正确做法是保留并明确 excluded reason。

**Q114｜项目里最关键的工程取舍是什么？**

**参考回答：** 先限制 fixed-base、单手、position mode；接触阶段又选择无摩擦合成球—盒。它牺牲场景真实性，换来变量隔离和可解释性。验证平台应先让最小问题稳定，再逐层增加原生指尖、双手和浮动基座。

**Q115｜如果重新做一次，你会提前做什么？**

**参考回答：** 更早统一 integer time grid、quaternion distance、outer-stage identity 和 runtime readback contract；在第一次 formal run 前增加针对步数、N+1 samples、结束时间、unknown fields、null/zero 混淆和 stale derived state 的 mutation tests，减少整矩阵重跑。

**Q116｜你如何面对一个“看起来合理但证据不足”的解释？**

**参考回答：** 把解释转成受控、可证伪的干预，并提前定义支持/不支持条件。dry-friction 有 material-contributor 证据；total-viscous 是 mixed；Freeze B 没落入任一注册区域就保持 INCONCLUSIVE。不能用直觉补齐因果链。

**Q117｜项目使用了 AI 辅助吗？该怎么诚实回答？【个人边界】**

**参考回答：** 项目确实使用了 AI coding agent 辅助开发、审计和文档整理，应该如实说明。我的可核验责任是确定问题、范围、验收与停止边界，组织正式运行，检查证据是否满足协议，并对最终表述负责；不要声称所有代码逐行手写。面试前必须选出 2–3 个我实际核查、能够现场打开源码或白板解释的模块；其余部分应说“借助 AI agent 完成并通过测试与证据链验收”，不能靠背文档冒充逐行实现。

**Q118｜一个人做项目，怎样避免自己既当运动员又当裁判？**

**参考回答：** 用预注册 manifest 和阈值减少事后自由度；finalizer 不依赖 simulator；用 mutation tests 主动攻击证据链；旧 run 不覆盖；publication-safe public projection 与私有原始证据分层，但 public projection 当前并未对外发布。仍然缺少外部同行复核，所以不能把个人实验说成行业标准。

**Q119｜怎样把它变成团队可用的平台？**

**参考回答：** 把 CPU asset QA 接入 PR CI，把 GPU campaign 做成受控的周期或 release gate；按 adapter 接入新后端；用版本化 manifest 管理场景；把 private evidence 放到受控对象存储，public summary 进入 dashboard；对阈值变更强制 review 和 migration note。

**Q120｜项目的最大不足是什么？**

**参考回答：** 正式跨模拟器动态验证目前只覆盖 fixed-base 标准左右单手和最小合成接触；v0.1 虽然静态或烟测了 wrist/flange/dual/floating-base，但没有给它们做动态 parity。项目也没有原生软指尖、双手交互、完整 Kit/RTX、外部使用者或公共 CI 运行证据。它提供的是特定资产、软件版本和冻结协议下可复核的方法实现与结果，不证明对其他机器人普遍有效，更不证明现实物理真实性。

**Q121｜如果下一阶段继续，你会先做什么？**

**参考回答：** 先在新的预注册阶段解决 C0 release event 的 dt 稳定性，再决定是否进入正式 cross-simulator contact label；之后才逐步加入原生指尖、双手和浮动基座。机械手真机、传感器/执行器硬件实验和 Sim2Real 不在本项目路线内。

**Q122｜这个项目最重要的学习是什么？**

**参考回答：** 学会把“程序跑完”“结果可重复”“数值对 dt 稳定”“证据可信”和“科学结论通过”拆成独立层。验证系统的成熟度不体现在全是 PASS，而在于它能准确拒绝不够强的结论。

---

## 10. 压力面快速 QA

**Q123｜2 mm 阈值有什么行业依据？是不是拍脑袋？**

**参考回答：** 它是本项目为首轮筛查预先冻结的工程容差，不是行业通用标准，也未由硬件任务误差标定。它的价值是提供稳定判定规则；若未来面向具体任务，应根据下游容差和风险重新预注册，而不是追溯修改旧结果。

**Q124｜阈值差一点就过，说明项目没有实际意义吧？**

**参考回答：** 相反，小幅超限最能检验协议纪律。正确做法既不四舍五入成 PASS，也不夸大成灾难；报告绝对超限约 0.11/0.32 mm、其他维度通过，并把它作为回归与归因起点。

**Q125｜两次 repeat 都是精确 0，是否说明测试太弱？**

**参考回答：** exact repeat 说明当前固定栈和输入下是确定性的，不代表指标足够敏感。项目还有 dt-halving、mutation tests、跨后端指标和 contact event gate；C0 正好证明 exact repeat 仍可同时存在 dt sensitivity。

**Q126｜为什么不直接把 PhysX 参数调到和 MuJoCo 曲线一致？**

**参考回答：** 那会把验证变成事后拟合，而且同名参数未被证明语义等价。更合理的是先读 runtime、做单变量 sensitivity、在新协议里定义 calibration objective，并把 calibration 与 holdout evaluation 分开。

**Q127｜如果没有 ground truth，做 parity 有什么用？**

**参考回答：** parity 不能判真实性，但能发现资产漂移、控制语义变化和后端升级回归；也能告诉团队同一策略或任务换后端后可能出现什么差异。它是兼容性与一致性测试，不是真值认证。

**Q128｜为什么不直接比较 joint state，非要看 frame？**

**参考回答：** joint 指标能定位状态差，frame 指标更接近任务空间影响；串联运动链会放大或抵消各关节误差。两者一起看才能知道小 joint gap 对末端空间造成多大后果。common-FK 再用于验证两者传播是否自洽。

**Q129｜为什么不比较 contact force，是否逃避关键指标？**

**参考回答：** 不是逃避，而是先解决可比语义。两个 solver 的 native force/impulse/patch 定义不同，直接硬比会产生虚假精度。C0 先建立 selected-pair Boolean、analytic gap、event 和 contact-minus-sham effect；force parity 需要单独的语义合同和校准研究。

**Q130｜A800 能运行 OVPhysX，为什么不能说做了 Isaac Sim？**

**参考回答：** Isaac Sim 是包含 Kit、RTX renderer、传感器等在内的更大应用栈。本项目只把 A800 当作 GPU 计算资源，运行 kit-less OVPhysX physics backend，没有 Kit/renderer/camera/Replicator。算力卡能运行物理后端，不等于完整 Isaac Sim 应用流程已被验证，也不等于做过机械手硬件实验。

**Q131｜937 passed 是否等于项目质量很高？**

**参考回答：** 它是很好的工程证据，但只能说明被覆盖的代码合同和失败路径通过。测试仍可能遗漏 bug；正式物理结论必须回到封存 evidence、阈值与适用范围，不能用测试数量代替科学效度。

**Q132｜项目有 GitHub remote、PR 或公开用户吗？**

**参考回答：** 当前没有。它是本地独立 Git 仓库，未 fork、未加 remote、未 push、未提交 PR，也没有外部采用证据。可以谈完成的工程原型和本地提交历史，不能声称开源影响力或线上用户。

**Q133｜为什么没有完整验证 dual-hand、wrist/flange 和 floating-base？**

**参考回答：** 首轮 parity 为了可解释性只选标准 fixed-base 左右单手。wrist/flange 已在资产 QA 中保留 frame offset 警告；dual-hand 和 floating-base 会引入更多自由度、惯性和接触，应作为独立 campaign，不应混入现有阈值。

**Q134｜如果面试官坚持问“所以到底是哪边错了？”**

**参考回答：** 我会明确说现有证据不能回答。项目已经排除大块 FK 几何差异并确认 dry friction 是重要因素，但 runtime 语义与 solver 仍未唯一识别。强行选边会超出证据，也违背这个项目的设计目标。

**Q135｜这个项目最像哪类实际工作？**

**参考回答：** 它同时接近资产质量门、仿真 backend 回归测试、benchmark harness 和研究型诊断。核心能力是把含糊的“两个系统差不多吗”转成可执行合同、完整矩阵、可审计证据和边界清楚的结论。

### 补充高频追问

**Q136｜为什么静态 FK 几乎一致，动态末端位置仍能相差 2 mm？**

**参考回答：** FK 只回答“给定同一组关节角，末端在哪里”；动态仿真还决定“关节角怎样随时间变化”。控制器、摩擦、阻尼、armature、质量惯量、solver 和 dt 都会让关节轨迹不同。因此微米以下的静态 FK 差与毫米级动态 frame 差并不矛盾。

**Q137｜两条轨迹怎样对齐？是否做了插值或人为平移？**

**参考回答：** 同一 dt 的跨后端比较要求 sample 数、step index 和物理时间一一相同，不做拟合对齐、时间平移或插值；不一致就拒绝比较。dt-halving 时，base 的第 `k` 个 sample 与 halved 的第 `2k` 个 sample 配对，保证对应同一物理时刻。

**Q138｜为什么验收使用最大误差，而不是只看 RMSE？**

**参考回答：** 最大误差捕捉最坏时刻，适合作为保守验收门；RMSE 描述整条轨迹的平均量级，更适合诊断。两者回答的问题不同，不能用较好看的平均值掩盖局部超限。

**Q139｜项目的创新点是什么？是否提出了新算法？【高风险】**

**参考回答：** 项目没有提出新物理求解器，不应包装成算法创新。主要工程与研究贡献是把跨格式检查、共同实验合同、后端适配、失败保全、数值稳定性门和可审计证据链组合成完整验证流程，并在固定 Wave 资产上产出可复核结果。

**Q140｜这套方法能直接推广到其他机器人吗？**

**参考回答：** 架构上可以增加 mapping、asset parser 和 backend adapter，但正式验证目前只覆盖固定版本 Sharpa Wave 的标准左右手与 MuJoCo/kit-less OVPhysX。可扩展性是设计目标，不是已经验证的泛化能力。

**Q141｜两次 exact repeat 能否证明跨机器也完全确定？**

**参考回答：** 不能。它只证明冻结软件栈、输入以及各自运行环境中的 fresh-process 重复一致；项目没有证明不同 CPU、GPU、驱动或操作系统之间逐字节确定。

**Q142｜个人项目如果被问“你怎样处理团队冲突”，怎么回答？**

**参考回答：** 这个项目没有真实团队冲突，我不会虚构。可以改谈方案取舍、自我质疑以及如何复核 AI 辅助产出；如果面试官必须听团队协作案例，应换用另一段真实团队经历回答。

---

## 11. 公式与概念速记

| 概念 | 面试速记 |
|---|---|
| Canonical mapping | `q_canonical = sign × q_backend + offset`，还需 index/scope/unit/coverage |
| OVPhysX 显式 PD | `τ = Kp(q_target-q) + Kd(v_target-v) + τ_ff`，之后做 effort clipping；MuJoCo 使用编译后的 MJCF position actuator |
| FK | `T_wc = T_wp · T_pj · T_motion(q) · T_jc`；回答给定 q 时 frame 在哪里 |
| Quaternion distance | 先 sign-align，再算相对旋转 geodesic；`q` 与 `-q` 同一姿态 |
| Analytic gap | `g = z_sphere_center - z_box_top - radius` |
| Contact effect | `g_contact-g_sham`、`q_contact-q_sham`、`R_contact·R_sham⁻¹` |
| Repeatability | 同 backend、同 dt、fresh process 重跑是否一致 |
| dt-halving | dt 减半、总物理时长不变，在共同物理时刻比较 |
| N / N+1 | N 个 integration interval 对应包含 sample 0 的 N+1 个状态样本 |
| Debounce | 状态连续 4 ms 才确认接触 onset/release |
| SHA-256 | 检查当前字节是否与登记字节相同，不证明物理正确 |
| DIVERGENT | 前置门通过，但至少一个主指标超过冻结阈值 |
| INCONCLUSIVE | 协议条件不足以支持更强标签；可能是前置门失败，也可能是有效指标落在预注册中间区域 |
| UNMAPPABLE | 有字段或数值，但不能安全声称跨后端语义等价 |

---

## 12. 数字速记卡

### 12.1 资产与覆盖

- 上游固定 commit：`6eea427eb24189519f32b9f21674cd534d3f973c`；
- 60 入口：15 URDF + 15 MJCF + 30 USDA；
- 516 checks：497 PASS + 6 KNOWN + 13 WARN；0 FAIL、0 SKIP；
- 标准手 FK：`<0.000071 mm`、`<0.000086°`；
- formal parity coverage：左右单手合计 44/44 joints、10/10 distal frames。

### 12.2 Gate 0

- 阈值：joint `0.02 rad`、position `2.000 mm`、orientation `0.05 rad`；
- left/right small-step position：`2.109/2.111 mm`；
- 超限量：约 `0.109/0.111 mm`；
- 结果：`DIVERGENT / pass_ready=false`。

### 12.3 三项归因

- 选定 left small-step 诊断轨迹的 dry-friction 归零：joint max `-51.36%`、position max `-59.67%`、orientation max `-75.86%`、joint RMSE `-89.48%`；
- total-viscous：pinky `+82.36%`，thumb `+4.22%`，整体 mixed；
- selected `left_thumb_DP` 窗口的 post-hoc common-FK unexplained residual：约 `0.0053%`，两候选均 `CLOSES`。

### 12.4 R1 / Freeze B / T1

- R1：44/44 joints、4 fresh process、0 trajectory；4 families `UNMAPPABLE`；
- Freeze B：24/24；`5/8 <=0.10`、`3/8 >0.10`、`0/8 >=0.25`；`VALID / INCONCLUSIVE`；
- T1：32/32、96,000 steps；四格 position `2.321–2.324 mm`；`VALID / DIVERGENT`。

### 12.5 Contact C0

- 32/32 fresh-process cases；
- base/halved dt：`0.002/0.001 s`；3500/7000 intervals；
- contact debounce：`4 ms`；
- OVPhysX release midpoint：`4.605/4.5995 s`；差 `5.5 ms > 4.0 ms`；
- 结果：`VALID / INCONCLUSIVE / pass_ready=false`。

### 12.6 测试

- 最近一次记录的全量自动化测试：`937 passed in 189.62s`；
- 这是软件测试数，不是 formal simulation case 数。

---

## 13. 模拟面试问题树

下面不是让你一次背完，而是用来检查某个回答能否继续下钻。

### 主线 A：项目价值

1. 这个项目解决什么问题？
2. 为什么“能加载”不等于“一致”？
3. 你比现有 smoke test 多做了什么？
4. 没有机械手硬件 ground truth，结果还有何价值？
5. 哪个产物可以直接成为团队回归门？

### 主线 B：个人贡献

1. 哪些资产来自上游？
2. 你设计了哪些协议和模块？
3. 哪个技术决策由你做出，为什么？
4. 项目是否使用 AI 辅助？你如何验证生成内容？
5. 哪一段代码或公式你可以现场白板解释？

### 主线 C：2 mm 超限

1. 指标的 frame 是哪里？
2. 阈值何时确定？
3. repeat 和 dt-halving 是否通过？
4. 关节与姿态为什么未超限？
5. dry friction、viscous 和 common-FK 分别排除了什么？
6. 现在能否判断哪个后端更准？

### 主线 D：C0 接触

1. 为什么 synthetic sphere-box？
2. 为什么有 sham？
3. portable contact quantity 是什么？
4. onset/release 和 debounce 怎么定义？
5. 为什么 32/32 仍是 INCONCLUSIVE？
6. 为什么没有继续算正式 cross-simulator label？

### 主线 E：可复现系统

1. manifest、mapping、adapter 各自负责什么？
2. 为什么 fresh process？
3. finalizer 为什么不能补跑？
4. hash 能证明什么、不能证明什么？
5. 如何阻止旧 run 拼接、路径泄漏和多余文件？
6. 如果 threshold 要变，如何版本化？

---

## 14. 面试前自检清单

在把该项目写进简历前，至少能不看文档回答以下内容：

- [ ] 30 秒和 90 秒项目介绍；
- [ ] 明确说出上游资产与本人验证工作的边界；
- [ ] 解释 44/44 不是同时双手 44 DoF，10/10 不是实际接触面；
- [ ] 画出 `manifest → adapter → worker → evidence → finalizer → report`；
- [ ] 写出 PD、mapping、FK、analytic gap 和 contact-minus-sham 的基本公式；
- [ ] 解释 repeatability 与 dt-halving 的区别；
- [ ] 说清 Gate 0 的 2 mm 阈值、2.109/2.111 mm 结果和绝对超限量；
- [ ] 说清 dry-friction、total-viscous、common-FK 各自支持什么、不支持什么；
- [ ] 解释 R1 为什么 `READBACK_VALID` 仍然 `UNMAPPABLE`；
- [ ] 解释 Freeze B 的 5/8、3/8、0/8 与 INCONCLUSIVE；
- [ ] 解释 T1 为什么比 single-step 更有说服力；
- [ ] 解释 C0 的 sham、debounce、N+1 samples 与唯一失败项；
- [ ] 能讲至少两次真实失败及如何保全旧证据；
- [ ] 主动声明没有机械手真机或硬件实验、Sim2Real、官方背书、remote、push 或 PR；
- [ ] 能诚实说明 AI 辅助开发边界，不声称所有代码逐行手写。

如果以上任一项只能背名词、不能解释输入输出和失败路径，应在面试前回到技术完整版和对应源码补齐，而不是提高简历措辞强度。

---

## 15. 延伸阅读与核对入口

- [完整技术档案与基础知识库（飞书）](https://sjtu.feishu.cn/wiki/ASRQwjR7EiSvRJk7eqrctveOnug)
- [讲人话版项目说明（飞书）](https://sjtu.feishu.cn/wiki/SL7qw1ewCiI9ZskkOx3cCGdPnPh)
- [Sharpa Robotics 上游资产仓库](https://github.com/sharpa-robotics/sharpa-urdf-usd-xml)
- [ROS URDF 官方教程](https://docs.ros.org/en/kilted/Tutorials/Intermediate/URDF/URDF-Main.html)
- [MuJoCo XML Reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html)
- [MuJoCo Computation](https://mujoco.readthedocs.io/en/latest/computation/)
- [OpenUSD Introduction](https://openusd.org/release/intro.html)
- [OpenUSD Physics](https://openusd.org/release/api/usd_physics_page_front.html)
- [PhysX Articulations](https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/docs/Articulations.html)
- [Isaac Lab kit-less OVPhysX](https://isaac-sim.github.io/IsaacLab/v3.0.0-beta2/source/overview/core-concepts/physical-backends/ovphysx/index.html)

本文是面试准备材料，不是对外论文或官方评测。技术数字如与口头记忆冲突，以封存实验 evidence、正式 public report 和完整技术档案为准。
