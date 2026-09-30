# Sharpa Wave Asset QA × WaveSimParity：完整技术档案与基础知识库

> 覆盖范围：v0.1 CPU-first 资产 QA、WaveSimParity Gate 0、三个归因诊断、R1 effective-parameter readback、Freeze B、Trajectory T1、Contact Gate C0。  
> 状态日期：2026-08-31（Asia/Shanghai）。  
> 项目性质：独立、非官方、simulation-only。没有硬件实验、Sim2Real、安全验证或 Sharpa 官方背书。

---

## 0. 先说结论

这个项目不是“把两个模拟器跑起来看看曲线”，而是把 Sharpa Wave 公开机器人资产的
静态检查、跨格式运动学检查、跨模拟器实验、失败保全、参数诊断、证据封包和公开脱敏
串成一套可复核的工程流程。

最终完成了四层工作：

1. **资产层**：检查 URDF、MJCF、ASCII USDA 的结构、关节、限位、网格、执行器和
   运动学一致性。
2. **无接触动态层**：在 MuJoCo 与 kit-less OVPhysX 中运行相同 fixed-base、position-mode
   场景，比较关节和末端帧轨迹。
3. **参数与归因层**：用摩擦干预、总粘性干预、共同 FK 重放、OVPhysX effective readback
   和 Freeze B 灵敏度实验，逐步缩小差异来源，但不越过证据边界。
4. **接触层**：用一个明确、可解析的合成无摩擦球—盒接触对建立 Contact Gate C0，
   同时设置 sham 对照，验证接触观测、重复性和 timestep sensitivity。

截至当前：

| 阶段 | 完成度 | 最终标签 | 最重要的结果 |
| --- | ---: | --- | --- |
| v0.1 资产 QA | 完成 | `497 PASS / 6 KNOWN / 13 WARN / 0 FAIL / 0 SKIP` | 60 个入口；标准左右手 URDF↔MJCF FK 最大位置误差低于 `0.000071 mm` |
| WaveSimParity Gate 0 | `32/32` | `DIVERGENT`, `pass_ready=false` | 左右 `small_step` 的末端帧位置最大差为 `2.109/2.111 mm`，超过 `2.000 mm` 阈值 |
| R1 effective readback | `44/44` joints | `READBACK_VALID` | 四类参数均因证据层不完整或语义不可直接对齐而保守判为 `UNMAPPABLE` |
| Freeze B | `24/24` | `VALID / INCONCLUSIVE` | 八个灵敏度值中 `5/8 <=0.10`、`0/8 >=0.25`，不满足任一预注册结论 |
| Trajectory T1 | `32/32` | `VALID / DIVERGENT` | sine/chirp 四个单元的末端帧位置最大差约 `2.321–2.324 mm`，均超过 `2.000 mm` |
| Contact Gate C0 | `32/32` | `VALID / INCONCLUSIVE`, `pass_ready=false` | 唯一失败项是 OVPhysX dt-halving 释放事件中点偏移 `5.5 ms > 4.0 ms` |

当前分支共有 **937 项测试通过**。所有正式阶段都保留源代码身份、资产身份、manifest、
schema、fresh-process、完整运行日志、严格 JSON、文件清单与 SHA-256 根哈希。公开结果只保留
脱敏后的 summary/report，原始机器信息继续留在本地私有证据中。

---

## 1. 项目目标与边界

### 1.1 为什么要做

同一台机器人通常同时维护 URDF、MJCF 和 USD。它们面向不同软件栈，表达能力、默认值、
单位、坐标约定和控制接口都不同。文件“能打开”不等于它们表达了同一台机器人；XML
“能解析”也不等于关节树、限位、执行器、惯量、碰撞或 frame convention 正确。

这个项目把容易被人工漏掉的问题变成明确检查：

- 某个 mesh 是否缺失、路径大小写是否可移植；
- 关节是否重复、漏掉、父子关系错误或形成环；
- 同名关节在不同格式中的轴、上下限和单位是否一致；
- actuator 是否覆盖预期关节、是否绑错目标；
- 随机关节姿态下，URDF 与 MJCF 的 distal-frame FK 是否一致；
- 同一目标序列在两个 simulator 中是否完成、有限、可重复、对 dt 稳健；
- 一个差异究竟属于资产、控制、摩擦/阻尼、运动学采样、数值误差还是尚不可归因。

### 1.2 明确不做什么

以下内容不在当前成果范围内：

- 真实硬件实验；
- Sim2Real、真实接触精度或安全结论；
- 完整 Isaac Sim 应用、Kit、renderer、camera 或 RTX 渲染链；
- 原生指尖 elastomer、真实材料或 native fingertip collision surface 的验证；
- 双手交互、floating base、wrist/flange 动态对比；
- 判断哪一个 simulator 更接近现实；
- 将 `DIVERGENT` 自动解释为 Sharpa、MuJoCo、NVIDIA、PhysX 或 Isaac Sim 的官方 bug。

### 1.3 固定研究对象

- 上游公开仓库：`sharpa-robotics/sharpa-urdf-usd-xml`。
- 固定上游 commit：`6eea427eb24189519f32b9f21674cd534d3f973c`。
- v0.1 资产内容树 SHA-256：
  `9c2d71aec9f8fe77aeb03660f735eb55bc53b93ba595f6cbfe5cc4e8fbb67de4`。
- WaveSimParity 使用的 canonical LF 资产 SHA-256：
  `b198e39b1030f279bf531c01d7e2321339dccf4482e662e6e8c671c9499d54ad`。
- 标准 fixed-base 左右手各 22 个 canonical scalar DoF、5 个 distal `*_DP` frame；合计
  `44/44` joints 和 `10/10` distal frames。
- 控制模式：position mode；跨模拟器 formal path 使用显式 IdealPD effort contract。

---

## 2. 项目演进时间线

### 2.1 v0.1：CPU-first 资产 QA（2026-08-26）

- `8d2d228`：建立资产发现、解析、静态审计、FK、MuJoCo smoke test、CLI 和测试。
- `916df2a`：固化可复现 v0.1 报告。
- 当时验收测试为 `57 passed`。

### 2.2 Gate 0：跨模拟器最小可行性（2026-08-26—27）

- `f03f48a`：准备 Gate 0 协议。
- `2a84245`：实现 manifest、adapter、runner、mapping、bundle、compare、report。
- 随后修复 idle-GPU 判断、远端 runtime home、venv entrypoint、受限 symlink 和 quaternion
  repeatability 等工程问题。
- `4918bc0`：记录第一版 Gate 0。
- `82ad6e1`：改为 canonical outer position-mode USD entrypoint。
- `c72fb10`、`4162fbb`：明确 IdealPD 控制与 MuJoCo 编译后控制 provenance。
- `aa2fd21`：发布控制路径修正后的结果。
- `c571d9b`：在 MuJoCo 采样 frame 前调用当前状态运动学刷新，消除 `qpos` 与
  `xpos/xquat` 跨步错位。
- `c134ec6`：发布 authoritative current-state Gate 0 结果。

### 2.3 差异归因诊断（2026-08-27）

- `e238b29`：MuJoCo dry-friction scale scan。
- `82951f9`：total-viscous controller-Kv 对齐实验。
- `c840a2b` 起：共同 MuJoCo FK 重放与五关节 Shapley attribution。
- `1e0f76b`：固定最终 common-FK 证据。

### 2.4 R1 与 Freeze B（2026-08-28—30）

- `557576f`：冻结 OVPhysX effective-parameter readback。
- `b86bca7`：发布 readback。
- `259fdbb` 起：预注册 legacy joint-friction Freeze B sensitivity study，并用多份
  corrigendum 公开记录证据绑定、admission、asset preflight、runtime order 和 time-grid
  修正。
- `d890f8e`：最终 Freeze B campaign G 轨迹源码。
- `8e2fd84`：最终 analysis I 时间网格兼容修正；不重跑 simulator、不改轨迹。
- `ff7bf04`：发布脱敏 Freeze B 结果。

### 2.5 Trajectory T1（2026-08-30）

- `ddec6aa`：冻结 sine/chirp manifest v2。
- `7db7e64`：规范空 GPU process envelope。
- `8af132c`：把 session consistency 按 backend 约束。
- `f92591a`：发布完整 T1 a3 结果。

### 2.6 Contact Gate C0（2026-08-31）

- `7c9cfdb`：实现 C0 合成接触协议、数据合同、两个 backend、runner、metrics、compare、
  bundle、report 和测试。
- `f328b2c`：证明 camera-free runtime，而不是错误地把被动 Python camera class import
  当成 camera 执行。
- `2001dcb`：加入 live PhysX CCD state readback。
- `2ba40cd`：把 mass/inertia 证明绑定到实际 `PxArticulationLink`。
- `12e84a1`：按物理时长正确派生 halved-dt 步数。
- `5c1ab46`：用 canonical integer time grid 固定终点时间。
- `17eccdb`：用稳定的 `atan2` quaternion geodesic 消除零角附近的 `acos` 数值伪差。
- `a5ed797`：发布 C0 脱敏结果和状态文档。

---

## 3. v0.1：资产 QA 具体做了什么

### 3.1 自动发现与分组

扫描固定资产根目录，识别：

- `15` 个 URDF 入口；
- `15` 个 MJCF 入口；
- `30` 个 ASCII USDA 入口；
- 共 `60` 个入口，覆盖 left/right、standard、wrist、flange、dual、floating-base 以及
  position/MIT-mode overlay。

发现结果不是简单文件列表，而是标准化为 canonical name、side、mounting、control mode、
floating-base 等字段，再按同一物理变体分组比较。

### 3.2 解析与静态检查

对 URDF/MJCF：

- XML 合法性；
- link/body/joint/actuator/geom/mesh 名称与重复项；
- 父子树、root、cycle；
- scalar joint 计数、类型、轴、上下限、effort、velocity；
- mesh 相对路径、存在性、大小写；
- actuator target 和 joint coverage；
- inertial 与 frame 信息的有限性。

对 USDA v0.1：

- 只解析 ASCII overlay 的直接文本属性；
- 检查 joint block、limit 和 position/MIT overlay 一致性；
- PhysX 角度限位从 degree 规范化到 radian；
- 不把 text-level pass 冒充 resolved USD stage 或 PhysX runtime validation。

### 3.3 跨格式一致性

15 个 configuration group 的 joint set 完整。对显式 authored limit pair，最大规范化差异
为 `0.001676°`。九个 dual/float-base group 的部分 USDA joint limit 通过更深层 USD 组合
继承、未直接出现在文本 overlay，因此保守保留 `WARN`，没有猜默认值。

### 3.4 FK 检查

使用固定 seed `20260826`，对每个适用变体采样 `256` 组合法关节角。先做一次 zero-pose
hand-base alignment，然后比较五个 distal phalanx frame 的位置和旋转，不逐样本拟合。

标准左右手结果：

- 最大位置误差低于 `0.000071 mm`；
- 最大旋转误差低于 `0.000086°`。

wrist/flange 变体出现稳定的约 `29.0 mm` / `29.5 mm` 基座对齐后位置偏移。因为偏移平滑、
一致且可能来自预期 frame convention，这些被记为 `WARN`，不是未经确认的 bug。

### 3.5 MuJoCo smoke 与 known finding

每个 MJCF 做 headless load、50 个 finite step、重复 rollout 和 CPU real-time factor 记录。
`12/15` 模型通过加载和有限重复 rollout。三个 dual-hand MJCF 重现了上游已经通过 PR #1
处理的 mesh-path 问题，因此标为 `KNOWN`，没有冒充新发现。

### 3.6 v0.1 最终统计

`516` 个检查：

- `497 PASS`
- `6 KNOWN`
- `13 WARN`
- `0 FAIL`
- `0 SKIP`

`KNOWN` 表示已知且有公开归因的问题；`WARN` 表示观察到差异但证据不足以判错；默认命令
仅对未知失败返回非零，`--strict` 可把 warning 也升级为本地/CI gate。

---

## 4. WaveSimParity 共用实验骨架

### 4.1 为什么是 MuJoCo ↔ kit-less OVPhysX

远端 A100/A800 类型 GPU 不带 RT Core，不适合完整 Isaac Sim 渲染流程。于是正式口径固定为：

- 本地：MuJoCo；
- 远端：Isaac Lab `v3.0.0-beta2.patch1` 提供的实验性 kit-less OVPhysX；
- Python `3.12`，纯 headless；
- 不启动 Kit、renderer、camera 或 RTX pipeline。

这仍是真实 OVPhysX physics step，但不是“完整 Isaac Sim 对比”。未来若做完整 Isaac Sim，
应单独使用 RTX/L40S/4090 等具备 RT Core 的节点并新建协议。

### 4.2 canonical manifest

manifest 是实验的单一事实来源，包含：

- 上游资产 commit/hash；
- 左右手与 canonical joint/frame 名单；
- backend、scenario、initial state、target profile；
- base/halved dt；
- repeat count、minimum completion fraction；
- 数值阈值与 claim boundary。

manifest 同时有文件 SHA-256 和 canonical semantic SHA-256。前者验证字节不变，后者验证
规范化 JSON 语义不变。Schema 采用闭合字段和 fail-closed 校验：未知字段、重复 key、NaN、
Infinity、错误顺序、错误相对路径或身份漂移都会拒绝。

### 4.3 canonical mapping

每只手 22 个 canonical scalar joint 和 5 个 `*_DP` frame。mapping record 不只保存名字，
还保存 backend index、坐标换算和 scope。验收必须是：

- `44/44` joint coverage；
- `10/10` distal-frame coverage；
- 不允许一对多、多对一、漏项或偷偷按位置猜名字。

### 4.4 fresh process

每个 formal case 都在新 Python process 中启动。这样可减少以下污染：

- simulator singleton 与全局缓存；
- CUDA context 和插件残留；
- 前一场景的 state、target、contact sensor 或 USD stage；
- 只在某个进程中偶然出现的初始化顺序。

每个 case 单独保存 command、stdout/stderr、exit code 和 canonical run JSON。需要区分协议演进：
Gate 0 已做到 per-case 独立 Python process，但早于 `process_identity.py`，不能追溯声称它具有
后续 campaign 的 kernel-derived identity。T1、Freeze B、C0 等现代 formal campaign 才进一步
要求 fresh-process hash 与 OS-level identity 唯一并交叉绑定：Windows 使用 PID + process
creation FILETIME，POSIX 使用 boot ID + PID + process start ticks；这些身份只留在私有证据。

### 4.5 base dt、halved dt 与 repeatability

- **repeatability**：同 backend、hand、scenario、dt 的两个 fresh repeat 是否一致。
- **dt-halving**：把 dt 减半、总物理时长保持不变，比较 common physical time 上的轨迹。
- **cross-simulator**：只有执行、finite、mapping、repeatability 和 dt gate 通过后，才比较
  MuJoCo 与 OVPhysX。

这三层不能混为一谈。一个不稳定或 dt-sensitive 的实验不能被直接称为“跨模拟器
divergence”，否则可能把数值积分问题错误归因给 simulator。

### 4.6 公私证据分层

私有 evidence 允许包含：

- 原始 run payload 与完整 sample；
- stdout/stderr；
- process/session identity；
- GPU/driver/runtime probe；
- 绝对路径、helper build provenance；
- backend-native quantity。

公开 projection 只包含去身份化的 summary/report。finalizer 会扫描账号、主机、路径、
session、PID、GPU UUID、native force 等字段，并要求公开目录严格只有允许的文件。

---

## 5. Formal Gate 0

### 5.1 实验矩阵

Gate 0 使用：

- 2 backends：MuJoCo、kit-less OVPhysX；
- 2 hands：left、right；
- 3 scenarios：`zero_hold`、`small_step`、`gravity_settling`；
- 2 repeats；
- `small_step` 额外做 base/halved dt。

合计 `32` 个 formal case。本阶段没有 ground plane 或 intentional contact target。

### 5.2 关键控制与采样修正

第一版 formal bundle 错误地打开 nested converter USD，而不是包含上游 position-mode tuning
的 outer USDA。后续固定 canonical outer stage，并使用显式 `IdealPDActuator` effort path；
resolved Kp/Kd 进入 controller，backend PhysX drive stiffness/damping 被验证为零。

另一次审计发现 MuJoCo `mj_step` 后 `qpos` 已前进，而 `xpos/xquat` 仍可能是上一状态的派生
Cartesian field。权威版本在记录 frame pose 前调用 `mj_kinematics`，并用 sampled qpos 独立
重放校验 frame。

### 5.3 完整性与稳定性结果

- `32/32` cases，requested step completion `100%`；
- `44/44` joints，`10/10` frames；
- non-finite run `0`；
- repeatability：joint/position/orientation 最大差均为 `0`；
- dt-halving：`0.00135195 rad`、`0.000161719 m`、`0.00284117 rad`，均通过
  `0.01 rad / 0.001 m / 0.02 rad` 阈值；
- 16/16 OVPhysX position-target readback 最大误差 `0 rad`；
- PD formula/clipping validation 最大误差 `0 Nm`，clip count `0`；
- 最大 computed/applied command `0.660047 Nm`。

### 5.4 跨模拟器结果

冻结阈值为 joint `0.02 rad`、frame position `0.002 m`、frame orientation `0.05 rad`。

| Hand | Scenario | Joint max | Frame position max | Orientation max | 结论 |
| --- | --- | ---: | ---: | ---: | --- |
| left | zero_hold | `0` | `0.000000047507 m` | `0.00000120065 rad` | within tolerance |
| left | gravity_settling | `0.00277094 rad` | `0.000327972 m` | `0.00383823 rad` | within tolerance |
| left | small_step | `0.0186829 rad` | `0.00210924 m` | `0.0457305 rad` | `DIVERGENT` |
| right | zero_hold | `0` | `0.0000000622974 m` | `0.000000847544 rad` | within tolerance |
| right | gravity_settling | `0.00272553 rad` | `0.000319917 m` | `0.00369807 rad` | within tolerance |
| right | small_step | `0.0187022 rad` | `0.00211119 m` | `0.0458414 rad` | `DIVERGENT` |

只有左右 `small_step` 的 frame-position 超过阈值，分别超出约 `0.109 mm` 和 `0.111 mm`。
最大点在 `thumb_DP` distal-link origin 附近，不是物理指尖表面。

### 5.5 Gate 0 结论

执行有效、比较可判定，但 acceptance 未通过：`DIVERGENT / pass_ready=false`。含义仅是
“在固定资产、版本、控制器、场景、dt 和阈值下观察到可重复差异”，不回答哪一边正确。

Gate 0 权威源码：`c571d9bb37a619fe9e867447d3be448acc615e3c`；私有 bundle root：
`d2e028d884f5484f7a7d5ecc0506201ff751c0847ed87b7a63d3a3b8cd056505`，358 个 payload，
84,635,733 bytes。

---

## 6. Gate 0 之后的三步归因

### 6.1 MuJoCo dry-friction 干预

问题：Gate 0 joint gap 与 `frictionloss / Kp` 有强相关，但相关不等于因果。于是固定 OVPhysX
formal trace，只改变 MuJoCo 编译后的 `dof_frictionloss` scale：`1.0`、`0.5`、`0.0`。

每条轨迹 250 step / 251 finite sample、无接触、相同 target 和 initial state。`1.0x` 与旧
formal MuJoCo trace 在 joint/frame-position/orientation 上完全一致，证明干预框架没有漂移。

从 `1.0x` 到 `0x`：

- trajectory-wide joint max 降低 `51.36%`；
- distal-frame position max 降低 `59.67%`；
- orientation max 降低 `75.86%`；
- joint RMSE 降低 `89.48%`；
- formal peak 的 `thumb_CMC_FE` 与 `thumb_DP` gap 分别降低 `99.49%`、`95.61%`。

这支持“MuJoCo dry friction 是该单条 left small_step 差异的重要贡献者”，但不是完整解释，
更不是 PhysX friction 等价或 bug 结论。零 dry-friction 后最差 joint gap 转移为早期
`left_pinky_CMC` transient `0.00908694 rad`，剩余全局 `thumb_DP` 位置最大差
`0.850667 mm`。

### 6.2 total-viscous controller-Kv 干预

问题：能否令 MuJoCo `controller Kv + passive damping = OV explicit-PD Kd`，用一个共同
viscous retuning 同时解释 pinky joint transient 和 thumb frame residual？

结果：

- `left_pinky_CMC` 注册窗口改善 `82.3625%`；
- `left_thumb_DP` 注册窗口只改善 `4.2222%`；
- frame-position RMS 和 global orientation max 反而变差。

因此 scientific status 为 `mixed_or_inconclusive`：一个 common viscous retuning 不足以
解释两类残差，不能称为整体轨迹改善。

### 6.3 common-FK attribution

把冻结 OVPhysX、zero-friction MuJoCo、viscous-treatment MuJoCo 的 22-joint trace 都送入
同一个 fixed-base MuJoCo FK model，只重放运动学，不做 dynamics step、不拟合 alignment。

在 `0.11–0.17 s` 的 `left_thumb_DP` 窗口：

- zero-friction control 未解释残差约为 recorded gap 的 `0.005274%`；
- viscous treatment 约为 `0.005273%`；
- 两者都满足预注册 `CLOSES` 规则。

五个 thumb joint 的 exact Shapley allocation 显示：control 中 `left_thumb_CMC_AA` 是唯一
满足注册 dominance 条件的正向 closer；treatment 中没有 player 同时满足两项 dominance
门槛。

正确结论是：**该 selected frame gap 在共同 FK 模型下几乎完全跟随记录到的 joint-state
gap**。这说明 frame 差异在运动学上自洽，但仍不解释 joint state 为什么不同。

---

## 7. R1 effective-parameter readback

### 7.1 为什么要做 readback

仅比较配置文件中的同名字段很危险。相同名字可能单位、作用位置或组合规则不同；真正进入
runtime 的 effective value 还可能受到 layer composition、默认值、转换和 solver 的影响。

R1 因此不跑实验轨迹，而是在四个 fresh process 中对左右手 44 个 canonical joint 做：

- authored layer；
- resolved USD layer；
- runtime-effective layer；
- control/drive 与 solver/runtime 可获得性；
- 映射、向量和 process repeatability。

### 7.2 结果

- status：`READBACK_VALID`；
- joint coverage：`44/44`；
- four fresh processes；
- mapping、solver evidence record、canonical numeric vector 完全重复；
- 用户/实验 trace step、command、target、trajectory sample 均为零。

参数结论：

| Family | 观察 | 语义结论 |
| --- | --- | --- |
| friction | 88/88 record 有 authored/resolved/runtime-effective layer | `UNMAPPABLE` |
| armature | 88/88 record 有三层证据 | `UNMAPPABLE` |
| control/drive | Kp/Kd、backend drive、effort limit、target interpretation 可见；部分 passive/gear 信息不可见 | `UNMAPPABLE` |
| solver/runtime | pipeline 和 configured dt 部分可见；integrator、iterations、solver type、stabilization 等不可见 | `UNMAPPABLE` |

`UNMAPPABLE` 不是“参数不同”，而是“现有证据不足以定义可靠的跨引擎一一语义映射”。缺失
字段被记录为 unavailable，没有用猜测或默认值补齐。

私有 bundle：56 payload、2,331,969 bytes，root
`858eee53e0ece9956a06b7cbd72a3d0a8348156e3c236ebf31ffd72fceb4fd4a`。

---

## 8. Freeze B：OVPhysX legacy joint-friction input sensitivity

### 8.1 研究问题

R1 只能读到 runtime value，不能从字段名字证明跨引擎物理等价。Freeze B 改问一个更窄、
可证伪的问题：对 frozen OVPhysX native friction input 做预注册干预时，轨迹差异是否呈现
足够一致、足够大的 sensitivity pattern？

### 8.2 运行与有效性

- 24/24 cases：8 MuJoCo unchanged controls + 16 OVPhysX；
- `44/44` joint、`10/10` frame；
- 24/24 fresh-process；
- repeatability exact；
- dt-halving、finite/sanity 和 16/16 position-target readback 均通过。

### 8.3 冻结判定

每个 hand/timestep cell 产生 `S_Dq`、`S_Dp`，共八个 sensitivity value：

| Hand | Timestep | `S_Dq` | `S_Dp` |
| --- | --- | ---: | ---: |
| left | base | `0.0930583` | `0.102107` |
| left | halved | `0.0970278` | `0.106797` |
| right | base | `0.0900038` | `0.0991497` |
| right | halved | `0.0938482` | `0.103678` |

- `SUPPORTED`：8/8 都 `>=0.25`；
- `NOT_SUPPORTED`：8/8 都 `<=0.10`；
- 其他有效结果：`INCONCLUSIVE`。

实测 `5/8 <=0.10`、`3/8 >0.10`、`0/8 >=0.25`，所以只能是
`VALID / INCONCLUSIVE`。runtime readback 还观察到 zero treatment 对 R1 的 8 个 case 都没有
改变，但这是描述性信息，不负责决定 label。

### 8.4 证据修正边界

正式轨迹只在 campaign G 收集一次。后续 H/I 是 analysis-only 修正：

- H 修正 canonical joint order 后，因要求跨 trace timestamp bit-exact 而产生一个
  `INVALID/null` 私有 bundle；
- I 仅允许同 step、同长度、绝对误差 `<=1e-12 s` 的 timestamp；不取整、不改 raw trace、
  protocol、window、formula、threshold、target、case order 或 dt，也没有重跑 simulator。

最终 bundle：407 payload、93,091,294 bytes，root
`3aed0876784c4575be20e5f5fbb6879f840829ae0b7885bda49ca133698bf04b`。

---

## 9. Trajectory T1：sine 与 chirp

### 9.1 范围与矩阵

- fixed-base left/right，各 22 joint；
- position mode，沿用冻结 explicit-PD contract；
- `offset_sine` 与 `offset_linear_chirp`；
- base/halved dt；
- 每个 backend/hand/scenario/dt 两个 fresh repeat；
- 合计 32 case；
- contact-free intent，无 ground plane/contact target。

### 9.2 有效性

- `32/32`、step completion `1.0`；
- `44/44`、`10/10`；
- non-finite `0`；
- repeat maxima：joint `0`、position `0`、orientation `0`；
- dt-halving maxima：`0.0003000945 rad`、`0.00004576161 m`、`0.0008436642 rad`，均通过；
- OVPhysX controller-state observation 48,016 个，effort clip `0`；
- runtime log fatal match `0`、unclassified warning `0`。

### 9.3 结果

| Hand | Scenario | Joint max | Frame position max | Orientation max | Label |
| --- | --- | ---: | ---: | ---: | --- |
| left | offset sine | `0.0198874 rad` | `0.00232438 m` | `0.0483022 rad` | `DIVERGENT` |
| left | offset chirp | `0.0197838 rad` | `0.00232210 m` | `0.0481766 rad` | `DIVERGENT` |
| right | offset sine | `0.0198750 rad` | `0.00232344 m` | `0.0482774 rad` | `DIVERGENT` |
| right | offset chirp | `0.0197732 rad` | `0.00232091 m` | `0.0481631 rad` | `DIVERGENT` |

joint 和 orientation 没超过各自阈值；四个 frame-position cell 全部比 `2.000 mm` 高约
`0.321–0.324 mm`，因此 T1 为 `VALID / DIVERGENT`。这增强了“差异不只出现在单个 step
命令”的可重复观察，但仍不确定动态原因。

T1 a1/a2 都被完整排除：a1 是空 GPU-process envelope 表示与 finalizer 不兼容；a2 是旧
global-session 规则错误地要求本地 MuJoCo 与远端 OVPhysX 共享一个 session。a3 在新源码下
完整重跑 16+16 fresh process，只有 a3 进入最终 bundle。

最终 private bundle：404 payload、755,944,170 bytes，root
`cf00924cf075cd4813750280aabfe58de6a119e21c4b3b1cb9fe9f84357eb464`。

---

## 10. Contact Gate C0

### 10.1 为什么先做 synthetic contact

原生指尖 collision mesh、材料、cooking 和 elastomer 会同时引入很多不可控变量。C0 先用
解析几何隔离一对 normal contact，目标是回答“两个 backend 的接触观测和接触效应管线是否
能在明确条件下稳定运行”，而不是验证真实指尖。

### 10.2 fixture

- 在 `{side}_index_DP` frame 上挂 massless sphere；
- local center `[0.025, 0, 0] m`，radius `0.005 m`；
- static box center `[0,0,0.120] m`，half extents `[0.150,0.150,0.010] m`；
- top surface `z=0.130 m`；
- analytic signed gap：`g = z_probe_center - 0.130 - 0.005`；
- `g>0` separation、`g=0` tangency、`g<0` analytic overlap；
- friction/restitution 全部 0，gravity 0，CCD disabled；
- native hand collision 全部 disabled，只允许 synthetic sphere + static box；
- 唯一选中 pair：`synthetic_index_probe__static_box`。

每个 run 还必须证明：native collision inventory 完整、质量/质心/惯量未被 overlay 改变、
direct filter 精确解析一个 sensor body 和一个 target、运行中没有 camera prim/Kit/renderer/
Replicator/synthetic-data runtime。

### 10.3 command

7 秒 press–hold–release。index chain peak target：

- MCP flexion `0.60 rad`；
- MCP ab/adduction `0.00 rad`；
- PIP `0.90 rad`；
- DIP `0.60 rad`；
- 其余关节为 0。

目标 envelope 使用 quintic smoothstep `h(u)=10u^3-15u^4+6u^5`：0–0.5 s 保持 0，
0.5–3.0 s 平滑压入，3.0–4.0 s 保持，4.0–6.5 s 平滑释放，6.5–7.0 s 回到 0。

base `dt=0.002 s`：3500 interval / 3501 sample；halved `dt=0.001 s`：7000 interval /
7001 sample。sample 0 明确 inactive；target[k] 作用于下一 interval，event/duty 使用 sample
1..N。

### 10.4 contact 与 sham

- `contact`：启用唯一 sphere-box pair；
- `sham`：保留相同几何、初始状态和 target，但禁用 collision pair。

portable `pair_active` 由 backend-direct selected-pair observation 判定。OVPhysX excluded pilot
给出 contact force 约 `0.00554 N`、sham `0 N`，因此 formal 前冻结 activity threshold：

`pair_active[k] := selected_pair_force_norm[k] > 1e-4 N`

原生 force magnitude 只用于生成 Boolean，作为私有描述性证据；绝不把两个 backend 的
native force 数值直接比较。

接触效应采用 difference-in-differences：

- blocked travel：`g_contact - g_sham`；
- joint effect：`q_contact - q_sham`；
- orientation effect：`R_contact * inverse(R_sham)`。

这样可以减去已有的 contact-free baseline difference，避免把基础动态差异冒充接触效应。

### 10.5 event 与 debounce

debounce window 固定 `4 ms`：base dt 连续 2 interval，halved dt 连续 4 interval。

- onset：第一段至少 4 ms 连续 active 的首个 interval；
- release：onset 后第一段至少 4 ms 连续 inactive 的首个 interval；
- event 保存 `[t_(k-1), t_k]`，比较使用 interval midpoint；
- steady hold window `[3.25,3.75) s`；
- recovery `[6.75,7.0) s`。

### 10.6 矩阵与有效性

`2 backends × 2 hands × 2 conditions × 2 dt × 2 repeats = 32 cases`。

最终：

- 32/32、32 fresh process；
- 44/44 joints、10/10 frames；
- finite/runtime log/admission/excitation 全部通过；
- repeatability 5/5：三个连续量 max 都是 0，pair bitset 与 event interval exact；
- initial gap 最小 `0.06439997 m`；
- recovery gap 最小 `0.06438679 m`；
- sham hold mean gap 最大 `-0.00981185 m`；
- contact hold duty `1.0`，sham/recovery duty `0`。

### 10.7 唯一失败项与最终判定

dt-halving：

- joint/effect `0.0012479424 rad <= 0.01`；
- gap/blocked `0.0000111756 m <= 0.001`；
- orientation/effect `0.0005679039 rad <= 0.02`；
- hold duty delta `0 <= 0.01`；
- event midpoint max `0.0055 s > 0.004 s`，失败。

具体来源：OVPhysX release midpoint base `4.605 s`，halved `4.5995 s`；左右手与两个 fresh
repeat 均复现。MuJoCo onset/release dt delta 为 `1.5 ms`，OVPhysX onset 为 `0.5 ms`。

冻结判定顺序要求 repeatability 和 dt-halving 先通过，之后才允许应用 cross-simulator
threshold。因此 C0 正确结论是：

- evidence `VALID`；
- science `INCONCLUSIVE`；
- reason `admission_repeatability_or_dt_failed`；
- `pass_ready=false`；
- 不进入 `DIVERGENT` 或 `WITHIN_TOLERANCE` cross-simulator 判定。

### 10.8 C0 失败尝试与修正

- early pilot：修正被动 camera module import 的错误拦截、PhysX schema 注册时机和 inertia
  tensor double rotation；p4 仅用于证明一条 OVPhysX case 可完整运行，全部排除。
- a1：halved dt 错误沿用 base 的 3500 step；新 revision 按总物理时长派生 7000 step。
- a2：浮点累计让最后 sample 稍大于 7.0 s；改用 integer canonical time grid，同时保留
  runtime time integrity check。
- a3：本地 16 case 完成，但旧 `2*acos(|dot|)` 在 byte-identical quaternion 上产生约
  `2.98e-8 rad` 伪差；远端 owned worker 在 case boundary 安全停止。改用 sign-aligned、
  normalized `4*atan2(||q1-q2||,||q1+q2||)` 后全矩阵 a4 重跑。
- a4：最终 active evidence。没有混用任何先前 attempt 的轨迹。

正式 source `17eccdb672d89cb1dca78efe41181097e26c4e12`。local/remote/private bundle root
分别为：

- `6eb41643c064c33c5f2acd7909ace4c931d1abdeea254ceecd92df699da25729`；
- `4f5f9a4555548b7ebff3bc295e0e93ed00cdb386b6ccef404ff2815c9b8237cd`；
- `7f69ff9416c755401a7b2915684c457930feb13cb0803effc2955c90468dd69d`。

---

## 11. 代码架构

整个仓库可以同时看成两条主线、四个平面。两条主线是：

- v0.1 资产 QA：发现 → 解析 → 静态审计 → FK → MuJoCo smoke → 报告；
- WaveSimParity：冻结 manifest → canonical mapping/target → backend adapter → case 执行 →
  证据记录 → 比较/诊断 → private bundle → public projection。

四个平面是：

1. **数据/协议平面**：contracts、scenarios、configs、schemas，定义什么输入合法；
2. **执行平面**：adapters、runner、worker、`run_*`，把同一 case 投到不同 backend；
3. **证据/审计平面**：process identity、bundle、diagnostics、finalizer，证明跑的确实是声明的
   代码和输入；
4. **分析/发布平面**：compare、metrics、sensitivity、report，把有效性判断与科学标签分开。

### 11.1 顶层目录

```text
sharpa-wave-asset-qa/
├─ configs/                 实验配置、known issues、canonical manifests、corrigenda
├─ docs/                    计划、状态、诊断、纠错与边界文档
├─ results/                 只跟踪脱敏公开结果；完整 raw evidence 默认 ignored
├─ schemas/                 闭合 JSON Schema
├─ scripts/                 本地 launcher、远端 launcher/probe、finalizer、诊断入口
├─ src/wave_asset_qa/       Python package
├─ tests/                   单元、mutation、launcher、finalizer 与隐私测试
├─ locks/                   固定远端 kit-less OVPhysX 依赖解析
├─ external/                只读使用的上游公开资产，不由 validator 修改
├─ .github/                 CPU CI workflow；不承担远端 GPU formal run
├─ pyproject.toml           包元数据、依赖、CLI、pytest/coverage 配置
├─ LICENSE                  Apache-2.0
└─ NOTICE                   上游资产归属与商标边界
```

### 11.2 v0.1 核心模块

| 模块 | 责任 |
| --- | --- |
| `discovery.py` | 确定性发现 URDF/MJCF/USDA，并规范化 side/mounting/control/floating-base |
| `models.py` | Joint、frame、actuator、asset reference 等共同数据结构 |
| `parsers.py` | 依赖轻量的 URDF、MJCF、ASCII USDA 解析与单位规范化 |
| `kinematics.py` | NumPy-only rotation、quaternion、transform、FK 基础运算 |
| `fk_compare.py` | fixed-seed URDF↔MJCF distal-frame FK 比较 |
| `simulation.py` | 可选 MuJoCo headless smoke、finite、repeatability 和 RTF |
| `audit.py` | 端到端检查编排、规则执行、known finding 和聚合统计 |
| `report.py` | v0.1 JSON/Markdown 输出 |
| `cli.py` | `waveqa audit` 命令行入口 |

### 11.3 通用 WaveSimParity 模块

| 模块 | 责任 |
| --- | --- |
| `adapters/base.py` | simulator-neutral adapter interface、trace sample、run result |
| `adapters/mujoco.py` | MJCF load、canonical mapping、position controller、frame sample |
| `adapters/ovphysx.py` | resolved USD、articulation/controller、kit-less stepping 与 runtime evidence |
| `parity/contracts.py` | manifest/result 的 strict dataclass contract 与状态 enum |
| `parity/scenarios.py` | 读 manifest、生成 canonical target、展开 deterministic case matrix |
| `parity/mapping.py` | joint/frame mapping 的名称、索引、符号、offset、unit 与 coverage |
| `parity/runner.py` | 单 case 执行、fresh-process orchestration、strict JSON I/O |
| `parity/compare.py` | execution assessment、repeat、dt 与 cross-sim metric |
| `parity/bundle.py` | safe relative path、SHA-256、inventory root、atomic manifest |
| `parity/report.py` | Gate 0 稳定 JSON/Markdown report |
| `parity/process_identity.py` | 后期 formal campaign 的 kernel-derived fresh-process 身份；不追溯到 Gate 0 |
| `parity/effective_readback.py` | authored/resolved/runtime 参数证据与 availability |
| `parity/sensitivity.py` | Freeze B sensitivity metric 与 scientific rule |
| `parity/waveform_metrics.py` | sine/chirp waveform 比较与 dt aggregation |
| `parity/kinematic_attribution.py` | common-FK replay 与 attribution |

### 11.4 Contact C0 模块

| 模块 | 责任 |
| --- | --- |
| `contact/contracts.py` | fixture、condition、schedule、threshold、claim boundary 的闭合合同 |
| `contact/scenarios.py` | quintic command、integer time grid、gap、debounce、32-case expansion |
| `contact/records.py` | public run/sample schema 与 canonical JSON |
| `contact/adapters.py` | 两个 backend 的 synthetic overlay、direct pair observation、private evidence |
| `contact/worker.py` | 单 case fresh worker、process binding、public/private 记录落盘 |
| `contact/runner.py` | case 选择、target/sample/fixture/mapping 的 fail-closed 验证 |
| `contact/metrics.py` | onset/release、hold/recovery、repeat、dt、condition effect |
| `contact/compare.py` | evidence status 与 science status 分层决策 |
| `contact/bundle.py` | strict JSON、regular-file tree、inventory root、private preimage |
| `contact/report.py` | 公共字段白名单、路径/secret scrub、summary/report |
| `contact/native/physx_ccd_readback.cpp` | 只读 PhysX 5.9 live CCD、mass、COM、inertia helper；不修改或释放引擎对象 |

### 11.5 Scripts 分层

- `fetch_sharpa_assets.*`：按固定上游 commit 获取/校验公共资产；
- `run_*_local.py`：本地 MuJoCo 矩阵，每 case 新进程；
- `run_*_remote.sh`：远端严格 GPU preflight、schema probe、worker、monitor、postflight；
- `probe_*.py`：只读环境、schema、runtime、effective parameter probe；
- `finalize_*.py`：本地离线验证，绝不运行 simulator；
- `run_mujoco_*_diagnostic.py`：摩擦和 controller-Kv 干预；
- `run_thumb_dp_common_fk_analysis.py`：已有 trace 的离线共同 FK 重放；
- `install_gate0_snapshot.py`：校验 source archive 后安装 immutable tree snapshot。

### 11.6 为什么 finalizer 必须离线

finalizer 的工作是验证和解释已经采集的证据，而不是“缺什么就顺手再跑一次”。它不会
import simulator，也不会调用 physics step。这样能保证：

- 原始 evidence 与 analysis 分离；
- finalization 不能偷偷补 case；
- 复核者无需 GPU 即可验证 hash、schema、完整性和 metric；
- analysis bug 可以明确区分为需要完整 rerun，还是只作用于既有 evidence 的披露修正。

---

## 12. 一次正式实验从开始到结束

### 12.1 Freeze / preregister

1. 在专用本地分支实现协议。
2. 写 canonical manifest 和闭合 schema。
3. 把场景、target、dt、repeat、阈值、判定顺序和 claim boundary 写进计划。
4. 用 synthetic/mutation tests 覆盖错误输入和 fail-closed 路径。
5. 在看正式结果前固定 manifest file/semantic SHA-256。

### 12.2 Freeze source identity

正式执行按各阶段冻结 contract 检查工作树；现代 C0 要求 tracked 与 untracked 都 clean，并拒绝
`src/`、`scripts/` 下的 bytecode artifact。记录：

- source commit；
- source Git tree；
- `git archive` SHA-256；
- asset commit/tree/content hash；
- manifest/schema hash。

source tree 中不能残留 ignored Python bytecode；否则同名 snapshot 可能携带不可见运行产物，
launcher 会拒绝执行。

### 12.3 本地矩阵

本地 launcher：

- 证明当前目录就是 Git top-level，HEAD 等于该实验的 formal revision；
- 验证 tracked worktree/index clean，运行脚本已被 Git 跟踪，导入包来自当前 checkout；
- 验证 source/asset/manifest；
- output path 必须不存在，禁止覆盖；
- 每个 case 用 `python -P` 启动 fresh Python，并清理 `PYTHONHOME`、`PYTHONPATH`、
  `PYTHONSTARTUP`、`PYTHONUSERBASE` 等外部污染后重建受控导入路径；
- 保存 command、stdout/stderr、exit code、process evidence 和 run JSON；
- 首个失败即保全已产生 evidence，禁止选择性补跑。

### 12.4 远端 immutable snapshot

上传的不是 mutable working directory，而是 source archive。远端 installer：

- 验证 archive SHA-256；
- 验证 commit 与 tree；
- 拒绝 path traversal、symlink/special entry 和额外文件；
- 安装到以 source tree 命名的只读身份目录；
- 重新枚举 snapshot 并生成 snapshot SHA-256。

snapshot 根哈希只排除它自己生成的 `.snapshot-sha256`，以避免自引用循环；source
revision、tree、archive 等 provenance marker 仍在哈希覆盖范围内。当前仓库 HEAD 是后续文档
状态，Gate 0、T1、C0 等正式实验各自绑定历史 source revision/tree/archive/snapshot，绝不能
拿当前 HEAD 替代当时的实验身份。

### 12.5 GPU 选择纪律

只在某张卡同时满足以下三项时使用：

- `memory.used <= 32 MiB`；
- `utilization.gpu == 0`；
- 没有 compute process。

启动前立即二次检查。没有严格空闲卡就不创建 run、不排队、不等待占卡；不杀任何外来
进程，不抢卡，不改驱动，不用 sudo。runner 只暴露选中的单卡，并持续保留 monitor 证据。

### 12.6 Remote case execution

远端 launcher 先验证：

- run ID 不存在；
- snapshot/asset/runtime identity 正确；
- GPU 仍空闲；
- no Kit/renderer/camera contract；
- 环境和 cache/home 都被限制在批准的工作根下。

然后按固定顺序执行 schema probe 和 case worker。每个 case 都有 preflight、owned process
group、worker binding、monitor、postflight 和 exit code。只允许终止当前 run 自己创建的
process group；绝不操作外来进程。

远端 source、asset、依赖、虚拟环境、cache、日志和 results 都必须留在获批边界内；只 expose
一张 physical GPU，worker 统一把它视为 `cuda:0`。launcher 保存运行前后 GPU 状态和中途
monitor，只能管理自己创建的 process group。Gate 0 远端先做左右手 resolved-schema probe 再
跑 16 个 case；R1 是 4 个只读 fresh readback、不跑轨迹；Freeze B/T1/C0 远端各 16 个 case。

### 12.7 Fetch 与 local finalization

远端到达 terminal success 后，完整取回证据到 ignored local results。finalizer 不直接在外部
输入树上分析，而是先验证 local/remote 输入互不嵌套、输出不存在，再 copy-first 到自己拥有
的 staging。随后重新：

1. 枚举 local/remote 文件，拒绝 symlink、reparse point、special file；
2. 核对 evidence manifest、path、size、file hash、root；
3. 解析所有 run/private preimage；
4. 核对 source、asset、dependency、mapping、target、step/sample、process；
5. 扫描 non-finite、fatal logs、runtime contract；
6. 先做 repeat 和 dt，再做允许的 scientific comparison；
7. 创建 sealed private bundle；
8. 生成严格脱敏 public summary/report；
9. 再次证明 public 副本与 bundle 内 public 成员逐字节一致。

sealed bundle 构建时会再次枚举 path、size 和 file hash，确认 sealing 期间没有变化，最后在
同一文件系统内原子晋升。immutable source snapshot 防止“执行时代码漂移”，sealed evidence
bundle 防止“执行后证据漂移”；这两层不可变性由不同 hash 证明，不能混为一谈。

### 12.8 Documentation 与 local commit

状态文档必须同时写：

- 完成了什么；
- 没通过什么；
- 哪些 attempt 被排除；
- 修复有没有改变 protocol/threshold/raw trajectory；
- 能声称什么、不能声称什么；
- source、manifest、bundle 和 public output 身份。

只提交源代码、协议、测试、脱敏 summary/report 和状态文档。raw evidence、进程信息、机器
路径、GPU UUID、native force 等继续留在 ignored private bundle。没有单独授权时不添加
remote、不 fork、不 push、不开 PR。

---

## 13. Fail-closed 设计清单

### 13.1 输入与 JSON

- 严格 UTF-8 JSON；
- duplicate key 拒绝；
- `NaN`、`Infinity`、`-Infinity` 拒绝；
- closed field set，未知字段拒绝；
- enum、数量、顺序、单位、finite/range 都验证；
- manifest、case ID、target digest 必须 canonical。

### 13.2 文件系统

- 只接受批准根目录内的 resolved path；
- 拒绝 `..`、absolute member、path traversal；
- 拒绝 symlink/reparse/special file；
- output 已存在就停止，不覆盖、不合并；
- write 使用 exclusive create/atomic replace；
- sealing 前后重新枚举，防止目录在计算 hash 时变化。

### 13.3 运行与进程

- 每 case 新 process；
- Gate 0 证明 per-case 独立 Python；现代 campaign 才要求 command/run/private-process 相互
  hash 绑定和 kernel-derived identity unique；
- C0 还要求 local 16 与 remote 16 的 identity union 恰好 32，且两侧没有交集；
- worker exit、stdout/stderr、requested/completed step 都检查；
- terminal error 保留，不使用同一 run ID 重试；
- 旧 evidence 永不拼进新 revision 的 formal bundle。

### 13.4 数值与科学

- sample 的 joint、velocity、target、pose、gap、Boolean 必须完整 finite；
- target 重新按 manifest 计算，不信任 worker 自报；
- physical duration 与 requested steps 必须一致；
- repeat、dt、cross-sim 按冻结顺序；
- capability unavailable 记 unavailable/invalid，不填猜测值；
- cross-engine 原生 force/count/solver quantity 不进入 portable pass/fail。

### 13.5 隐私与发布

- public JSON key/token 白名单；
- 扫描账号、主机、absolute path、session、process、GPU UUID、secret/token；
- claim boundary 中 hardware、Sim2Real、full Isaac Sim、native fingertip 等必须显式 false；
- public output 文件集合必须精确；
- private bundle hash 与 public projection hash 分开。

closed projection 加 regex/denylist 是强门槛，但不是能发现世界上所有秘密的万能扫描器；发布
前仍应独立检查 Git history、token、路径与人名。sealed bundle 是 content-addressed、可检出
篡改，不等于成品目录一定被操作系统设为只读；真正移除写位的是 immutable source snapshot。

---

## 14. 基础知识库：机器人资产格式

### 14.1 URDF

URDF 是 XML 机器人结构描述，核心对象是 link、joint、parent/child、origin、axis、limit、
visual、collision 和 inertial。它像一份结构说明书，但不是统一的完整 dynamics/solver
配置。不同软件导入同一 URDF 后仍可能给 contact、damping、friction、actuator 或默认值
不同语义。

### 14.2 MJCF

MJCF 是 MuJoCo 原生 XML，除 body/joint/geom 外还直接表达 actuator、contact、constraint、
damping、frictionloss、armature、world、gravity 与 solver/integrator 配置。它不是“换后缀的
URDF”，而是更接近可直接编译运行的 engine-specific model。

### 14.3 USD 与 USDA

USD 是可组合 scene graph。一个 outer layer 可以 reference/payload 多个 inner layer，同一
属性可被上层 override；运行时真正生效的是 fully composed resolved stage。

- `.usda`：human-readable ASCII；
- `.usdc`：binary crate；
- `.usd`：容器扩展名，不能只靠后缀判断内部编码。

v0.1 只做 ASCII overlay text-level parsing；Gate 0/C0 才实际打开 outer stage 并检查 resolved
PhysX schema。两类能力不能混写。

### 14.4 PhysX schema

PhysX schema 给 USD prim 附加 rigid body、articulation、joint、drive、collision、scene、CCD
等结构化物理语义。需要区分：

- prim：scene graph 对象；
- schema：对象上的物理定义；
- PhysX：实际执行动力学的 engine。

“文件可打开”不等于 schema 完整；“schema 存在”不等于实际 stepping；重要属性还需
runtime readback。

---

## 15. 基础知识库：MuJoCo、OVPhysX 与 Isaac Sim

### 15.1 MuJoCo

- 独立 physics engine；
- 本项目直接加载 MJCF；
- CPU-first、headless；
- 易于读取 compiled joint/actuator/mass/inertia；
- 是被比较的 backend，不是 ground truth。

### 15.2 kit-less OVPhysX

- 使用 GPU PhysX/OVPhysX physics path；
- 通过冻结 Isaac Lab/Python interface 创建 simulation；
- 加载 resolved USD；
- 不启动完整 Kit application；
- 不执行 renderer、camera 或 RTX workload；
- 输出被 adapter 规范到同一 canonical record。

### 15.3 层次关系

```text
PhysX / OVPhysX       dynamics solver layer
Isaac Lab             robotics Python framework layer
Omniverse Kit         application/extension runtime
Isaac Sim             Kit + USD + PhysX + sensor + rendering product
renderer/camera/RTX   visual and sensor path
```

所以正确名称是“MuJoCo ↔ kit-less OVPhysX”，不是“完整 Isaac Sim 渲染对比”。

---

## 16. 基础知识库：joint、DoF、fixed/floating base 与 mapping

### 16.1 Joint 与 DoF

joint 是连接关系，DoF 是独立运动坐标。revolute/prismatic 通常 1 DoF，ball joint 3 个旋转
DoF，free joint 有 6 个物理 DoF。当前标准单手恰好有 22 个 scalar joint，因此可称 22
DoF；44/44 是左右手 campaign 总覆盖，不是同时运行一个 44-DoF 双手系统。

### 16.2 Fixed base

fixed base 把 hand root 固定在 world，去掉根部六自由度、惯性漂移和额外接触变量，使首次
cross-sim 对比更容易解释。Gate 0、T1、C0 都是 fixed-base。

### 16.3 Floating base

floating base 增加根部 3 translation + 3 rotation DoF。position state 常以 quaternion 保存
7 个变量，velocity 仍是 6 个物理自由度。它会引入新的 coordinate order、mass/inertia、
contact 和 integration 差异，当前未做。

### 16.4 Mapping 不只是名字

canonical mapping 还必须保存 backend index、scope、sign、offset、unit：

```text
q_canonical = sign × q_backend + offset
```

coverage 要求预期集合完全相等，无漏项、额外项、重复 name/index 或模糊绑定。

---

## 17. 基础知识库：FK、frame 与 quaternion

### 17.1 FK

Forward Kinematics 从关节状态沿树组合刚体变换：

```text
T_world_child = T_world_parent × T_parent_joint × T_motion(q) × T_joint_child
```

它回答“给定 q，frame 在哪里”，不回答“为什么 q 会这样变化”。因此 common-FK 能证明
frame gap 与 joint gap 的几何一致性，却不能直接解释 dynamics cause。

### 17.2 Frame

frame 是带 position/orientation 的局部坐标系。项目的 `*_DP` 是 distal-phalanx link
origin，不是实际 fingertip surface。C0 的 sphere 还相对 `index_DP` 有固定 local offset。

### 17.3 Quaternion

项目统一 `qwxyz`，pose 为 `(x,y,z,qw,qx,qy,qz)`。`q` 与 `-q` 表示同一旋转，因此不能
直接逐元素比较。零角附近使用 sign-aligned stable geodesic：

```text
theta = 4 atan2(||q1-q2'||, ||q1+q2'||)
```

这避免 `acos` 在 dot≈1 时把相同旋转报告为极小非零角。

---

## 18. 基础知识库：position control 与动力学参数

### 18.1 Position mode 不是瞬移

典型显式 PD：

```text
tau_computed = Kp(q_target-q) + Kd(v_target-v) + tau_feedforward
tau_applied = clip(tau_computed, -tau_limit, +tau_limit)
```

同一 q target 在 Kp/Kd、限幅、passive damping、dry friction、armature、solver 或 dt 不同
时会产生不同 state trajectory。

### 18.2 Contact friction 与 joint dry friction

- contact friction：两个 surface 间的 tangential resistance；
- joint dry friction：关节内部与运动方向相反的阻力。

MuJoCo `frictionloss` 与 PhysX legacy joint friction 不一定同单位或同作用层，数值相同不
代表物理等价。

### 18.3 Damping

passive damping 近似 `tau=-c*v`；controller Kd 也产生 velocity-dependent torque，但二者
计算层、target、limiting 和作用顺序不同，不能随意相加后宣称跨引擎等价。

### 18.4 Armature

armature 是折算到 generalized coordinate 的附加 actuator/rotor inertia，会影响加速度、
高频响应和数值条件。它不是 link mass，也不是 damping。

### 18.5 Mass、COM、inertia

总质量相同仍可能因 COM 或 inertia tensor 不同而有不同旋转动力学。C0 massless overlay
必须同时证明 mass、COM、principal axes、mass-space inertia 与 link-axis reconstructed
tensor 未变。

### 18.6 Restitution 与 CCD

- restitution 控制反弹；C0 为 0；
- CCD 沿 interval 检查高速穿透；C0 禁用并同时验证 schema opinion 与 live PhysX flag。

---

## 19. 基础知识库：dt、repeatability 与事件

### 19.1 dt

dt 是一次 integration advance 的物理时间。更小 dt 常减少部分离散误差，但 contact
solver、controller update、constraint stabilization、iteration 和浮点取样可能使指标并非
单调改善。

### 19.2 Repeatability 与 dt-halving 是两个问题

- repeatability：同一个 dt 再跑一次是否一致；
- dt-halving：dt 减半、总时长相同后是否仍接近。

C0 同一 dt 的两个 repeat exact，但 OVPhysX release 对 dt 的偏移仍为 5.5 ms。可重复性好
与 timestep sensitivity 高可以同时存在。

### 19.3 N interval、N+1 sample

sample 0 是 initial state。target[k] 作用于下一 interval，state[k] 是前一 interval 后的
observation。base 3500 interval 对应 3501 sample；halved 7000 对应 7001 sample。

### 19.4 Debounce

raw Boolean 可能短暂抖动。C0 需要连续 4 ms 才确认 event：base 2 interval、halved 4
interval。确认后 event backdate 到连续 run 的第一个 interval，并用 interval midpoint 比较。

---

## 20. 基础知识库：contact/sham 与 difference-in-differences

sham 保留同一 geometry、command、initial state 和 analytic gap，但禁用 collision pair。
它是负对照，回答“没有接触阻挡时本来会怎样”。

对同一 backend/hand/dt/repeat：

```text
b(t)   = g_contact - g_sham
e_q(t) = q_contact - q_sham
R_e(t) = R_contact × inverse(R_sham)
```

再比较两个 backend 的 effect。这个 paired difference 能减去已有 contact-free baseline
offset。它是一种实验控制结构，不等同于人口统计学因果推断。

raw contact point/patch/manifold count 和 native force magnitude 依赖 backend 表达，不能直接
跨引擎比较。portable quantity 是 selected-pair Boolean、analytic gap、event 和
contact-minus-sham effect。

---

## 21. 基础知识库：状态标签

### 21.1 Execution

回答“程序是否完整执行”。`COMPLETED` 不等于科学通过。

### 21.2 Evidence

回答“身份、schema、完整性、finite、mapping、capability、bundle 和 privacy 是否可信”。
`INVALID` 会强制 science `INCONCLUSIVE`。

### 21.3 Science

- `WITHIN_TOLERANCE`：满足所有前置门槛，主要跨引擎指标均在阈值内；
- `DIVERGENT`：满足前置门槛，至少一项主要指标超过冻结阈值；
- `INCONCLUSIVE`：证据或前置门槛不足，不能正式判断前两者。

`VALID / INCONCLUSIVE` 完全合理：数据可信，但尚不能给出更强科学分类。

### 21.4 pass_ready

`pass_ready=true` 需要 `VALID + WITHIN_TOLERANCE + complete mapping/coverage`。程序没崩、证据
valid、甚至结果可判定，都不自动等于 pass-ready。

---

## 22. Hash、provenance 与可复现证据

### 22.1 SHA-256 能与不能证明什么

SHA-256 可证明当前字节与登记字节相同，不能证明代码无 bug、实验物理正确或来源可信。
因此它必须和 source/asset provenance、schema、runtime evidence 和 scientific rule 一起用。

### 22.2 File hash 与 semantic hash

- file hash 对原始 bytes；换行/空格改变也会变；
- semantic hash 对 canonical JSON；关注解析后的含义。

两者分别回答“是不是同一文件”和“是不是同一实验语义”。

### 22.3 Inventory root

每条记录包括 relative path、size、file SHA-256。按 path 排序并用长度 framing 聚合为一个
root SHA-256；创建时间等非内容 metadata 不进入根哈希。验证时仍要重新枚举目录、拒绝额外
payload，不能只比较 stored root 字符串。

### 22.4 Provenance

可靠来源链至少包含上游 commit/tree、asset content hash、source revision/tree/archive、
manifest/schema hash、dependency identity、runtime identity、fresh process、case payload 和
bundle root。“最新版”不是可复现身份。

---

## 23. 公开结果与私有证据边界

可以公开：

- 去身份化 aggregate metrics；
- source/asset/manifest/public artifact hash；
- result label、threshold、case/mapping count；
- correction history 和 claim boundary。

不直接公开：

- 服务器地址、登录账号、SSH 信息；
- 本地/远端绝对路径；
- GPU UUID、process/session identity；
- raw runtime logs；
- backend-native selected-pair force；
- 未脱敏完整 evidence bundle。

这份知识库也遵守同一边界：它写出技术事实和公开 hash，但不把访问方法或机器身份复制到
云端。

---

## 24. 可复现操作手册

本节的目的不是鼓励直接重跑已经冻结的正式实验，而是说明每一层如何被独立核验。正式
结果应优先验证现有证据；只有提出新问题、修复协议错误或明确建立新 revision 时才重新
采集。

### 24.1 最低软件要求

CPU 侧需要 Git、Python 和可创建隔离环境的工具。v0.1 支持 Python 3.10/3.12；正式远端
OVPhysX 栈冻结为 Python 3.12.14、Isaac Lab `v3.0.0-beta2.patch1`、
`isaaclab-ovphysx 3.0.2`、`ovphysx 0.4.13`。这些版本是本次证据身份的一部分，不能用
“最新版本应该也一样”替代。

推荐的本地基础检查流程：

```text
python -m venv .venv
python -m pip install -e ".[dev,simulation]"
pytest
```

Windows 下的 v0.1 一键入口是 `scripts/run_v0_1.ps1`。它固定 seed、发现资产、运行 audit、
生成 JSON/Markdown。若只做代码审查，可先运行测试，不需要访问 GPU。

### 24.2 首先验证身份，而不是首先运行

在正式执行前依次核对：

1. 当前 Git commit 与 tree 是否等于目标 revision；
2. 工作树是否满足该阶段规定的 clean/allowlist；
3. 上游资产 commit、tree 与 canonical content hash 是否一致；
4. manifest file hash、semantic hash 和 schema hash 是否一致；
5. launcher、worker、adapter、finalizer 是否都来自同一冻结源码；
6. Python 可执行文件、关键包和 backend build 是否等于冻结身份；
7. `src/`、`scripts/` 是否没有未登记 bytecode 或其他多余 payload。

这里的顺序很重要：身份错误时继续跑，只会得到一份昂贵但不能纳入结论的日志。

### 24.3 本地矩阵

各阶段的本地入口分别为：

- Gate 0：`scripts/run_gate0_local.py`；
- Freeze B：`scripts/run_ovphysx_freeze_b_local.py`；
- Trajectory T1：`scripts/run_waveform_t1_local.py`；
- Contact C0：`scripts/run_contact_c0_local.py`。

launcher 要为每个 case 启动 fresh Python process，写出 case manifest、stdout/stderr、runtime
identity、轨迹和 terminal marker。已存在目标目录时应失败，而不是覆盖；一个 case 失败时也
不能假装整棵 evidence tree 已完成。

### 24.4 远端 OVPhysX 矩阵

远端只使用 immutable source snapshot、固定依赖环境和 strict-idle GPU。对应入口为：

- Gate 0：`scripts/run_gate0_remote.sh`；
- R1 readback：`scripts/run_ovphysx_effective_readback_remote.sh`；
- Freeze B：`scripts/run_ovphysx_freeze_b_remote.sh`；
- T1：`scripts/run_waveform_t1_remote.sh`；
- C0：`scripts/run_contact_c0_remote.sh`。

正式 launcher 的职责不只是启动进程。它还执行 GPU 双重 preflight、schema/capability probe、
source/asset identity 验证、per-case fresh process、process-envelope 记录、terminal-state 原子
提交和 postflight。任何 terminal error 都应保全，不能自动换 run id 掩盖失败。

### 24.5 严格空闲 GPU 的定义

本项目使用三个条件的交集：显存占用不高于冻结上限、GPU utilization 为 0、没有 compute
process。选择后、启动前立即复查。若没有卡满足条件，就什么都不改并等待；不抢卡、不排队
占位、不杀别人的进程。

这是实验隔离规则，不是“GPU 越空结果越科学”的物理假设。其作用是降低外部进程影响、
避免误伤共享资源，并让运行条件可审计。

### 24.6 取回与 finalization

远端到达 `collected` 后，完整取回 evidence tree。finalizer 在本地离线完成：

1. exact inventory 枚举，拒绝缺失和额外文件；
2. 逐文件 hash、size 与 bundle-root 重算；
3. schema、enum、case id、hand/backend/scenario 组合检查；
4. source、asset、manifest、runtime 与 fresh-process provenance 检查；
5. 44/44 joints、10/10 frames 和预期 case 数检查；
6. 数值有限性、shape、sample count、time grid 与单位检查；
7. repeatability、dt-halving、cross-sim 或 treatment/sham 指标计算；
8. 按预注册决策树生成 execution/evidence/science/pass-ready；
9. 生成 private bundle 和脱敏 public projection；
10. 对 public projection 做 forbidden-token/privacy scan。

finalizer 不能启动模拟器、补 case、改轨迹、拟合阈值或写回 raw evidence。这样才能把“采集”
与“解释”分开。

### 24.7 为什么旧证据不能拼起来补齐新实验

一个正式 bundle 的 case 必须共享同一个 source/manifest/runtime 语义。即使两个旧 run 的曲线
看起来一样，只要 launcher、mapping、时间网格、采样语义或 source revision 不同，就不能用
它们拼成一个新矩阵。修正应采用：新 commit、新 immutable snapshot、新 run id、整矩阵从零
重跑；旧证据保留在 excluded/corrigendum 中。

### 24.8 验证而不重跑模拟器

对已有正式产物，推荐先做四层只读验证：

- `git show`/`git cat-file` 核对 commit 与 tree；
- 对公开 summary/report 重算 SHA-256；
- 用 finalizer 或专用 verifier 对 private inventory/root 重算；
- 运行全量 `pytest`，确认当前代码的解析器、比较器、mutations、launchers 和 privacy gates。

这通常已经能回答“文档里的数字是否来自登记证据”。只有要检验新环境、改变研究问题或
复现 backend 采集本身时，才需要 GPU。

---

## 25. 如何解读全部结果

### 25.1 v0.1 回答的是“资产是否自洽”

v0.1 说明标准手模型的结构、joint/limit、mesh 引用和 CPU FK 大体一致，并公开保留 dual
mesh path、overlay limit 和 wrist/flange frame offset 等已知项。它不能回答 PhysX runtime、
控制器或接触行为。

### 25.2 Gate 0 回答的是“最小无接触任务是否跨引擎一致”

zero-hold 和 gravity-settling 在冻结阈值内；small-step 的 joint 与 orientation 在阈值内，但
DP position 约超 0.11 mm，因此整体为 DIVERGENT。这是一个小而稳定、可复现的协议内差异，
不是“两个引擎完全不同”。

### 25.3 诊断链把“差异在哪里”推进到“哪些因素可能贡献”

dry-friction intervention 显著减小一条候选轨迹的残差；total-viscous 只对部分 window 有效；
common-FK 说明 DP 差几乎可由 joint-state 差经同一 FK 解释。合起来最稳妥的表述是：主要问题
更像 dynamics/control-state evolution，而不是两套 FK 几何定义本身；但现有证据没有唯一识别
底层因果，也没有证明某 backend 实现错误。

### 25.4 R1 说明“读到数值”不等于“语义可映射”

44 个 joint 的 readback 稳定且精确重复，但四类参数都被标作 UNMAPPABLE。原因不是没有数字，
而是一个 backend 的 authored/resolved/runtime 字段无法安全对应到另一个 backend 的单一动力学
语义。把 unavailable 或语义不明字段填 0，会制造虚假的参数一致性。

### 25.5 Freeze B 没有支持也没有否定单一摩擦解释

冻结支持门槛要求 8/8 sensitivity ratio 都至少 0.25；否定门槛要求 8/8 都不高于 0.10。
实测 5/8 不高于 0.10、3/8 略高于 0.10、0/8 达到 0.25，所以唯一合法结论是
INCONCLUSIVE。这个结果仍然有价值：它排除了“证据明显强支持”的说法，并把下一步问题
缩小到更精细的 native runtime dynamics。

### 25.6 T1 说明差异不是只出现在单步命令

sine 与 chirp 四个 hand×waveform cell 都完整、精确重复、通过 dt-halving；每格 DP position
约 2.321–2.324 mm，略超 2 mm，而 joint/orientation 没超。它把 Gate 0 的 small-step 现象扩展
到更长、更连续的控制轨迹，但仍然只是阈值意义上的 DIVERGENT。

### 25.7 C0 当前停在前置稳定性门，不应偷跑跨引擎结论

C0 的 32 个 case、mapping、finite、runtime proof、repeatability、admission 和大多数 dt 指标
都通过；唯一失败是 OVPhysX release event midpoint 的 base/half 差 5.5 ms，高于 4 ms。
预注册决策顺序要求在此停止，所以正式结论是 `VALID / INCONCLUSIVE`，`pass_ready=false`。
描述性 onset/release 数字可报告，但不能在正式 label 中越过 dt gate 宣称 contact parity 或
contact divergence。

---

## 26. 能说什么，不能说什么

### 26.1 可核验的成果表述

- 建立了覆盖 URDF、MJCF、USDA 的 CPU-first 资产 QA；
- 建立了同一 canonical manifest 下 MuJoCo 与 kit-less OVPhysX 的跨模拟器证据管线；
- 对标准 fixed-base 左右手完成 44/44 joint 与 10/10 distal-frame mapping；
- 实现 fresh-process、repeatability、dt-halving、hash/provenance、exact inventory 和脱敏发布；
- 完成无接触 Gate 0、参数诊断、R1 readback、Freeze B、T1 和 synthetic Contact C0；
- 当前全量测试为 937 passed；
- 对失败尝试采用 corrigendum/excluded evidence，而不是覆盖历史。

### 26.2 不能从现有证据推出的结论

- 不能说项目得到 Sharpa、NVIDIA、MuJoCo 或 Isaac Sim 官方认可；
- 不能说做过真机、硬件、Sim2Real 或安全验证；
- 不能说哪个模拟器更接近真实世界；
- 不能把 kit-less OVPhysX 写成完整 Isaac Sim；
- 不能把 v0.1 的 USDA 文本检查写成 resolved PhysX 验证；
- 不能把 DIVERGENT 自动写成官方 bug；
- 不能把 C0 的 synthetic sphere-box 推广到原生指尖材料、软接触或抓取；
- 不能把 DP frame origin 写成实际接触面；
- 不能把 `null/not observed` 写成 0；
- 不能声称仓库已公开发布、已 push、已有 PR 或已有外部 CI 运行。

### 26.3 一句最准确的总述

这是一个非官方、simulation-only、证据优先的机器人资产与跨模拟器验证项目：它已经可靠地
测出并界定若干小而稳定的差异，也明确记录哪些问题仍然不能下结论。

---

## 27. 常见误解与 FAQ

### Q1：497 PASS，为什么总体不是 PASS？

因为还有 6 个已知问题与 13 个警告。总状态采用保守聚合，不能让大量 PASS 淹没需要公开的
边界。

### Q2：32/32 完成，为什么 C0 不是通过？

完成率只回答 case 是否跑完。C0 的 evidence 是 VALID，但一个预注册 dt-halving event gate
失败，所以 science 必须 INCONCLUSIVE，pass-ready 仍为 false。

### Q3：超过 2 mm 只有约 0.11 或 0.32 mm，值得报告吗？

值得。阈值在看结果前冻结，不能因为超得不多就事后放宽。与此同时必须连同绝对超限量、
其他指标通过和阈值语境一起写，避免夸大。

### Q4：DIVERGENT 是否意味着 PhysX 或 MuJoCo 有 bug？

不意味着。它只表示在指定资产、控制、采样和阈值下，两条输出曲线的某项指标超限。原因可
来自模型语义、参数映射、求解器、控制实现或协议本身。

### Q5：为什么要两次 fresh-process repeat？

同进程重复可能共享全局状态、缓存和初始化结果。fresh process 更能暴露 import order、随机
初始化、全局 registry 和 teardown 污染；两次 exact 一致是强工程信号，但仍不是现实正确性。

### Q6：为什么 dt-halving 这么重要？

如果结论随积分步长明显变化，就难以判断看到的是 backend 差异还是离散化误差。dt-halving
是数值稳定性门，不是对真实连续系统的证明。

### Q7：为什么 contact force 不直接跨引擎比较？

不同 solver 对 contact point、patch、impulse、force 和采样时刻的定义不同。C0 先比较 pair
是否 active、analytic gap、事件和 contact-minus-sham effect；native force 只作同 backend
描述。

### Q8：为什么需要 sham？

没有 sham，看到的轨迹差包含原有无接触 baseline。sham 保留同样命令和几何，只关闭目标
pair，使 difference-in-differences 更接近“接触额外造成的变化”。

### Q9：为什么 R1 不直接给出一套等价参数？

readback 只能告诉我们 backend 暴露了什么；它不能自动建立跨引擎语义等价。遇到不可安全
映射的字段，诚实的结论是 UNMAPPABLE，而不是猜一个数。

### Q10：为什么保留失败 run？

失败 run 是协议演进证据。保留它能说明错误在何处被发现、为什么旧证据被排除、阈值是否
保持不变，也防止成功结果看起来像从第一次尝试就完美得到。

### Q11：为什么不直接在远端生成最终报告？

远端负责受控采集，本地 finalizer 负责离线验证。分离后，finalizer 不能偷偷补跑或依赖 GPU
现场状态，public/private 投影也更容易审计。

### Q12：A800 能跑 OVPhysX，为什么不叫完整 Isaac Sim？

本项目只调用 kit-less OVPhysX，无 Kit、renderer、camera、Replicator 或 RTX 渲染管线。
“能做物理计算”和“完成完整 Isaac Sim 流程”是两件事。

### Q13：项目现在算完成了吗？

当前约定范围已完成到 synthetic Contact Gate C0，并形成可核验结果。native fingertip contact、
elastomer、wrist/flange dynamics、dual-hand interaction、floating-base、完整 RTX/Kit Isaac Sim
和 policy playback 都尚未开展；它们是可选未来研究，不是当前成果。硬件始终排除在路线图外。

---

## 28. 术语表

| 术语 | 这里的含义 |
|---|---|
| Asset QA | 对模型文件、引用、结构、参数和可加载性的系统检查 |
| Backend | 实际执行运动学/动力学的模拟器实现 |
| Canonical | 跨后端共同采用、经规范化且冻结的定义 |
| Manifest | 声明 case、场景、控制、采样、阈值和预期身份的机器可读配置 |
| Schema | 对 JSON 字段、类型、枚举和约束的机器可验证定义 |
| Provenance | 从上游资产、源码、依赖到 runtime 和产物的来源链 |
| Evidence tree | 一次采集的 case 文件、日志、身份和 terminal marker 目录树 |
| Bundle | 经 finalizer 验证、带 inventory/root 的证据集合 |
| Public projection | 从 private bundle 派生的脱敏 summary/report |
| Fresh process | 每个 case 在新操作系统进程中执行 |
| Joint | 连接两个 link 的运动约束/自由度定义 |
| DoF | 可独立变化的广义坐标数量 |
| Fixed base | 基座固定在世界坐标系，不增加自由运动基座 DoF |
| Floating base | 基座具有平移和旋转自由度，本项目当前未纳入 parity |
| FK | 由 joint state 计算 link/frame pose 的前向运动学 |
| DP | Distal phalanx；本文指标使用其 link origin，不代表表面接触点 |
| Position mode | 目标位置经 PD/effort 等动力学路径驱动，不是瞬间设定 qpos |
| Kp/Kd | 位置误差和速度误差的比例/微分增益 |
| Dry friction | 与运动方向相关的关节库仑/静摩擦类项 |
| Damping | 通常与速度成比例的耗散项 |
| Armature | 加在关节侧的等效惯量项 |
| CCD | Continuous Collision Detection，防止快速物体穿透的连续碰撞检测 |
| dt | 模拟积分时间步长 |
| Repeatability | 同一输入在独立重复中是否给出一致输出 |
| dt-halving | 时间步长减半后关键结果是否稳定 |
| Contact pair | 研究中明确选择并过滤的两个碰撞几何体 |
| Sham | 保留其他设置但关闭目标 treatment 的对照条件 |
| Difference-in-differences | 先算每个 backend 的 contact-sham effect，再比较 effect |
| Debounce | 要求 Boolean 状态持续一段时间才认定事件，过滤瞬时抖动 |
| VALID | 证据身份、完整性和结构满足要求 |
| INVALID | 证据层失败，不能继续正式科学判断 |
| WITHIN_TOLERANCE | 所有前置门槛和主要指标都在冻结阈值内 |
| DIVERGENT | 前置门槛通过，但至少一项主要指标超过冻结阈值 |
| INCONCLUSIVE | 按冻结决策树不能给出前两类结论 |
| UNMAPPABLE | 字段存在，但不能安全声称与另一 backend 参数语义等价 |
| pass_ready | 是否满足发布为“通过”的全部工程和科学条件 |

---

## 29. 代码与文档索引

### 29.1 最值得先读的源代码

- `src/wave_asset_qa/discovery.py`、`parsers.py`、`audit.py`：v0.1 发现、解析和审计；
- `src/wave_asset_qa/kinematics.py`：固定种子 FK 与 frame 比较；
- `src/wave_asset_qa/parity/contracts.py`：共用 run/result contract；
- `src/wave_asset_qa/parity/scenarios.py`：Gate 0 scenario 生成；
- `src/wave_asset_qa/parity/mapping.py`：joint/frame canonical mapping；
- `src/wave_asset_qa/parity/runner.py`：case orchestration；
- `src/wave_asset_qa/parity/compare.py`：repeat/dt/cross-sim 比较；
- `src/wave_asset_qa/parity/bundle.py`：inventory、hash 与 bundle；
- `src/wave_asset_qa/adapters/mujoco.py`：MuJoCo adapter；
- `src/wave_asset_qa/adapters/ovphysx.py`：kit-less OVPhysX adapter；
- `src/wave_asset_qa/contact/`：C0 contracts、fixture、worker、metrics、compare、report、bundle；
- `src/wave_asset_qa/contact/native_physx.py`：C0 native 只读 PhysX runtime proof 接口。

文件名以当前源码树为准；若未来重构，应通过 Git tree/commit 找回本文对应版本，不应只凭目录
印象定位。

### 29.2 关键状态与纠错文档

- `docs/GATE0_PLAN.md`、`docs/GATE0_STATUS.md`；
- Gate 0、Freeze B、T1、C0 各自的 plan/status/corrigendum；
- `docs/CONTACT_C0_STATUS.md`；
- `results/v0.1/report.md`；
- tracked 的各阶段 public `summary.json` 与 `report.md`。

这些状态文档不只是摘要，也记录为什么某棵证据被排除。正式引用数字时，应以 public artifact
和登记 hash 为准，而不是只引用 README 中可能被后续压缩过的介绍。

### 29.3 测试覆盖的工程风险

当前测试不仅覆盖正常路径，也覆盖 mutation/failure：schema 错误、case 缺失、多余文件、hash
变化、NaN/Inf、mapping 不完整、时间网格偏移、身份错配、重复进程、source drift、asset drift、
非法 terminal state、public/private 泄漏和阈值边界。当前完整测试结果是 `937 passed`。

本机实际命令为 `python -m pytest -q -p no:cacheprovider`，退出码 0，记录为
`937 passed in 189.62s`。按 pytest 参数化 node ID 粗分：v0.1 57；Gate0/parity/process
identity 227；R1 68；Freeze B 137；三项归因诊断 64；T1 168；C0 216，总计 937。这里的
937 是测试用例，不是 937 次物理仿真；GPU 相关测试多数验证 launcher/contract/fake host 的
fail-before-external-state，正式数值结论仍来自封存 evidence。

CI 配置声明 Python 3.10/3.12 CPU 测试，但仓库目前没有 remote，也未 push；因此只能说
“CI workflow 已配置”，不能说“公共 CI 已经跑绿”。

---

## 30. 关键身份与公开产物附录

### 30.1 当前源码与资产身份

- 当前分支：`feat/wavesim-contact-gate-c0`；
- C0 最终文档提交：`a5ed79728ab2f8589dc4535706a560ff6312bef9`；
- 对应 tree：`5abe029e6d24ba1aeb6fb343a577f00cc832134b`；
- C0 正式运行源码：`17eccdb672d89cb1dca78efe41181097e26c4e12`；
- 对应 tree：`bb0dfccffeb0b2a1e50d60862507fc4e9411ee3b`；
- 上游资产 commit：`6eea427eb24189519f32b9f21674cd534d3f973c`；
- 跨平台 Git tree：`bb00a9d5527b8a76de576ce876ebece67d8ffde1`；
- canonical LF tree SHA-256：`b198e39b1030f279bf531c01d7e2321339dccf4482e662e6e8c671c9499d54ad`；
- v0.1 Windows materialized tree SHA-256：`9c2d71aec9f8fe77aeb03660f735eb55bc53b93ba595f6cbfe5cc4e8fbb67de4`。

最后两种 content hash 因换行物化规则不同而不同，不代表上游资产版本冲突。

### 30.2 正式 bundle roots

| 阶段 | Bundle root / evidence root | 结果 |
|---|---|---|
| Gate 0 | `d2e028d884f5484f7a7d5ecc0506201ff751c0847ed87b7a63d3a3b8cd056505` | DIVERGENT |
| dry-friction diagnostic | `0804626e…` | material contributor（局部） |
| total-viscous diagnostic | `a27826a0…` | mixed_or_inconclusive |
| common-FK | `d2fbad78…` | 两条候选均 CLOSES |
| R1 readback | `858eee53…` | READBACK_VALID，四类 UNMAPPABLE |
| Freeze B | `3aed0876…` | VALID / INCONCLUSIVE |
| T1 | `cf00924c…` | VALID / DIVERGENT |
| C0 local evidence | `6eb41643c064c33c5f2acd7909ace4c931d1abdeea254ceecd92df699da25729` | collected/verified |
| C0 remote evidence | `4f5f9a4555548b7ebff3bc295e0e93ed00cdb386b6ccef404ff2815c9b8237cd` | collected/verified |
| C0 private bundle | `7f69ff9416c755401a7b2915684c457930feb13cb0803effc2955c90468dd69d` | VALID / INCONCLUSIVE |

带省略号的条目是便于阅读的显示前缀；需要逐字核验时应从对应 tracked public artifact 读取完整值。

### 30.3 十二个脱敏公开文件

| 阶段/文件 | SHA-256 | 字节数 |
|---|---|---:|
| v0.1 `report.json` | `75bb3a952f6896cef520330a9f5b9ad6806782915c4e33b56d90f1b01a991f85` | 222,727 |
| v0.1 `report.md` | `c728ce656d63b73c6f3c755cb9381c593573460aac41c7b336ff243adef3e753` | 10,954 |
| Gate 0 summary | `1cf21a7edd5e2ceba72dc294ac54b2689f5b73a665fef067e426e32fc04499be` | 15,559 |
| Gate 0 report | `12b6f6cfb1405862fafd959173f6e7f6f3a81711a1e3ae168ace91aeb92cc756` | 10,444 |
| R1 summary | `d359ffd9622907658edc3c412b1870aab8ae717d1a5805f77bc7b5f530b2d055` | 51,798 |
| R1 report | `5cbc1f33b974e519bc91b864d859f8904d806baf5ab266014496871225330898` | 2,294 |
| Freeze B summary | `6a7cc328b614cc15cba1b3819c90527a371313dad1f70b64bc9e4247ab76d611` | 13,868 |
| Freeze B report | `914f8e42854de817af4646680e15271000c7d4eb2e787fec3d0b3ca7dad1e507` | 5,325 |
| T1 summary | `49b3834e6b30a82a82a398e8498add13395b999d9ad64ff4efa60fc7a5d8ba44` | 414,469 |
| T1 report | `d6591dd1c76f50dbb0a56c0ff79e7d9802c922e706b8996e3ba36a7f1ba70b27` | 392,712 |
| C0 summary | `168070d53be26c0a092b3069713128c3e98625839e41d80bb16ae4c1a63b6801` | 22,802 |
| C0 report | `1078d311745d195eadb6fd588a4362f63a63ac83fb595b9c9cac0be3e7c13728` | 2,356 |

这些文件是“可公开的证据投影”，不是完整 raw evidence。本文没有把 ignored private bundle 上传到
云端，也没有把本地文件位置转写为远端访问说明。

---

## 31. 当前状态与后续边界

当前约定的本地项目工作已经推进到 synthetic Contact Gate C0：源码、测试、正式本地/远端
evidence、finalization、脱敏 summary/report 和状态文档均存在。Git 仓库无 remote，未 fork、
未 push、未提 PR、未创建 release。

如未来继续，合理的 simulation-only 研究方向包括：native fingertip/elastomer contact、
wrist/flange dynamics、dual-hand interaction、floating-base、完整 RTX/Kit Isaac Sim、policy
playback。每一项都应另立 gate、预注册 fixture/metrics/thresholds、冻结新源码和新证据，不能
用本次 C0 的结论直接外推。

硬件实验不在当前或后续约定范围内；本文不包含也不建议把真机、Sim2Real 或现实安全性写成
已完成或待完成事项。

---

## 32. 文档维护规则

1. 新结论必须指向 tracked public artifact 或可验证 private bundle；
2. 数值更新时同时更新 source/manifest/artifact hash；
3. 失败尝试写 corrigendum，不删除历史；
4. 新阈值必须在看正式结果前冻结；
5. private evidence 不直接复制进协作文档；
6. “未测”“不可映射”“不在范围”不得被润色成“为 0”“相同”或“已验证”；
7. 项目称谓始终保留“非官方、simulation-only”；
8. 任何公开发布、fork、push 或 PR 都需要另行明确授权。

本文档的目标不是把所有状态写成成功，而是让后来者能区分：做了什么、怎样做、哪些数据可信、
哪些问题仍然开放，以及哪些话绝对不能从现有证据推出。

---

## 33. 权威延伸阅读

以下链接用于补概念，不替代本项目冻结的 manifest、源码与 evidence：

- [Sharpa 上游公开资产仓库](https://github.com/sharpa-robotics/sharpa-urdf-usd-xml)：本文唯一固定的资产来源；
- [ROS 2 URDF 文档](https://docs.ros.org/en/rolling/Tutorials/Intermediate/URDF/URDF-Main.html)：
  link、joint、geometry 与 robot description 入门；
- [MuJoCo MJCF XML Reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html)：
  MJCF 元素、joint、actuator、contact、damping、frictionloss、armature 和 solver 属性；
- [MuJoCo Computation](https://mujoco.readthedocs.io/en/latest/computation/)：
  forward dynamics、constraint、FK、quaternion 与 timestep/integration 背景；
- [OpenUSD Introduction](https://openusd.org/release/intro.html) 与
  [USD Terms and Concepts](https://openusd.org/release/glossary.html)：layer、prim、stage、
  reference/payload、composition 与 resolved opinion；
- [OpenUSD Physics Schema](https://openusd.org/release/api/usd_physics_page_front.html)：
  rigid body、joint、articulation 与 drive 的跨实现 USD 表达；
- [PhysX Articulations](https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/docs/Articulations.html)：
  reduced-coordinate articulation、fixed/floating base、joint friction 与 link state；
- [PhysX Rigid Body Dynamics](https://nvidia-omniverse.github.io/PhysX/physx/5.4.0/docs/RigidBodyDynamics.html)：
  mass、COM、inertia、velocity、force 和 rigid-body dynamics；
- [Isaac Lab 3.0 beta2 OvPhysX Backend](https://isaac-sim.github.io/IsaacLab/v3.0.0-beta2/source/overview/core-concepts/physical-backends/ovphysx/index.html)：
  官方明确说明 OvPhysX 是高度实验性的 kit-less PhysX 路径，并提示不要与 Kit visualizer 混用；
- [Isaac Lab kit-less installation](https://isaac-sim.github.io/IsaacLab/v3.0.0-beta2/source/setup/installation/kitless_installation.html)：
  kit-less 与完整 Isaac Sim 功能边界、Python 环境和可选 OV runtime。

阅读顺序建议：先 URDF/MJCF/USD，再 FK/动力学与 PhysX articulation，最后看 kit-less OvPhysX。
项目结果的最终依据仍是本文登记的历史 commit、manifest、public artifact hash 和 private bundle。
