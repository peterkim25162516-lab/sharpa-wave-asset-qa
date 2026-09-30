# C0 释放时刻差异：离线诊断（2026-09-29）

后续进展：已完成 [已知载荷与接口交叉实验](CONTACT_LOAD_CALIBRATION.md)，在原生 OVPhysX CPU 接口复现 step_sync 接触读数沿用旧 dt 的行为。下文保留先前离线诊断的证据与当时边界；GPU C0 尚未按修正路径重跑。

## 结论

本次新分析把 C0 的 5.5 ms 差异定位到一个更具体的机制：OVPhysX 的接触信号幅度随 dt 近似成比例变化，而正式事件使用固定的 1e-4 检测门槛。这个组合显著改变了“检测到释放”的时刻。现有证据不支持把 5.5 ms 直接解释为球真实分离运动发生了同等大小的变化。

这是事后诊断，不是新验收。原 C0 保持 VALID / INCONCLUSIVE、pass_ready=false。本次未修改正式轨迹、阈值、模拟器、依赖或协议，未运行 GPU 仿真；远端仅只读检查固定版本源码。

## 方法与证据完整性

- 输入：已封存 a4 private bundle 的 32 个案例，使用正式 ContactRun 加载器验证。
- 在分析前后重算全部 689 个 payload 的清单、大小及 SHA-256，并验证固定 bundle root。
- 16 组 fresh-process 重复的完整 samples 均完全相同。
- 复用正式 debounce_pair_activity 重算 onset/release；原始 pair_active 必须与原始接触信号 > 1e-4 完全一致。
- 对原始信号离线扫描检测门槛和去抖窗口；分析副本不覆盖原样本。
- 比较相同物理时刻的关节、目标和解析球—盒间距；额外计算返回阶段上穿正间距阈值的线性插值时刻。
- 所有新扫描和插值均为 post-hoc exploratory 分析，不是预注册结果，也不是时间步连续收敛证明。

## 1. 正式失败可以精确复现

左手第一次 OVPhysX 运行：

| 步长 | 正式释放区间（s） | 区间中点（s） |
| --- | --- | --- |
| 2 ms | [4.604, 4.606] | 4.6050 |
| 1 ms | [4.599, 4.600] | 4.5995 |

半步长提前 5.5 ms。左右手及两次 fresh repeat 均出现同一差值。两区间不重叠，最近边界仍相差 4 ms，因此单靠区间中点舍入不能解释全部差异。

## 2. 不是去抖窗口造成的

将去抖窗口分别设为 1、2、4、8、16 ms，OVPhysX 的释放中点差始终为 -5.5 ms；MuJoCo 始终为 -1.5 ms。当前信号没有导致结果变化的短暂断触脉冲。

## 3. 固定检测门槛对结果影响显著

以下为左右手及重复均一致的 OVPhysX 结果。门槛单位暂称为“原始接口读数单位”：正式记录字段将其标为 N，但本次发现要求进一步核查单位语义。

| 同时用于两个 dt 的门槛 | 半步长减原步长的释放中点差 |
| --- | ---: |
| 0（严格正数才算 active） | -0.5 ms |
| 1e-6 | -0.5 ms |
| 1e-5 | -1.5 ms |
| 1e-4（正式值） | -5.5 ms |
| 1e-3 | -73.5 ms |
| 1e-2 | 无有效接触事件，不能记成“0 ms／通过” |

另一个只用于机制检验的对照：保留 base 门槛 1e-4，将 halved 门槛按 dt 比例设为 5e-5，释放差缩小为 -0.5 ms。这只说明尺度假设能够解释多数检测偏移，不授权修改验收门槛，也不证明接口数值应直接除以 dt。

## 4. 稳态信号几乎随 dt 减半

固定采用原协议 hold 窗口 [3.25, 3.75) s；在共同时间点比较：

| 后端／手 | base 信号中位数 | halved 信号中位数 | 逐点 halved/base 比值中位数 |
| --- | ---: | ---: | ---: |
| MuJoCo 左 | 2.02429139 | 2.02425984 | 0.99998441 |
| MuJoCo 右 | 2.02429139 | 2.02425984 | 0.99998441 |
| OVPhysX 左 | 0.0041115335 | 0.0020476425 | 0.49806996 |
| OVPhysX 右 | 0.0041114762 | 0.0020477982 | 0.49808605 |

表格只用于后端内部的 dt 尺度诊断，不把两种后端的原始接触力数值作为正式跨引擎可比量。每行两次独立重复一致；共同时间点的目标角度差均为零。OVPhysX 的 hold 最大关节差约 0.001243 rad，因此不能声称两种 dt 的动力学完全相同。

稳态读数近似正比 dt，与“冲量型尺度／冲量到力换算路径异常”的假设一致。它比泛泛归因于求解器不稳定更具体，但尚不等于确认接口返回的就是未换算冲量。

## 5. 几何离开的时刻并没有移动 5.5 ms

读取已记录的解析 signed_gap：球底到盒顶的距离。返回段 [4, 6.5] s 内，OVPhysX base/halved 间距最大差约 11.18 μm（0.01118 mm）；sham 空跑最大差约 2.36 μm。

| 几何分离距离 | 左手 crossing 时间差（halved-base） | 右手时间差 |
| --- | ---: | ---: |
| 1 μm | -0.129 ms | -0.036 ms |
| 10 μm | +0.112 ms | +0.116 ms |
| 100 μm | +0.165 ms | +0.169 ms |

这些是相邻离散样本的线性插值估计，不是亚毫秒真值测量。在零距离附近，OVPhysX 的间距有纳米量级正负抖动，直接找 gap=0 的第一次 crossing 会在约 4.0 s 得到伪“释放”。因此保留零 crossing 作为诊断数据，但不拿它证明实际分离时刻。

## 6. 固定版本调用链审计

只读检查 Server 6 的 IsaacLab checkout：commit ffff603eafc6b74264a5261cc0183d6a65390d78，tracked 工作区无变更。

调用链为：

1. OvPhysxManager.step 读取 physics dt，调用 PhysX.step_sync(dt, sim_time)。
2. 本项目每步之后调用 sensor.update(dt, force_recompute=True)。
3. ContactSensor 读取 ContactBinding.read_force_matrix。
4. Warp kernel 将矩阵直接复制到 force_matrix_w；这段没有显式 dt 缩放。
5. 本项目取唯一选定接触对向量的范数，与固定 1e-4 比较。

安装包 ovphysx 0.4.13 的 Python API 文档和 SDK 自带 contact_binding.md 都明确说明：底层应从最近一步自动取得 dt，完成 impulse / dt 换算。因此不能仅凭 Python 层没有除法就断言“漏除 dt”，更不能盲目补除一次。

目前尚未从原生二进制内部证明究竟是 step_sync 的 dt 状态、原生 contact binding、GPU 接触报告或其他具体路径造成该尺度现象。下一项有判别力的实验应在固定载荷／已知静力平衡的最小场景中，比较 dt=2/1/0.5 ms 的原生 binding 与封装 sensor 读数，并对照 step_sync 与 step+wait；同时保存姿态、载荷及原生 impulse（若可用）。结果出来前不提交“官方 bug”结论。

审计文件 SHA-256：

- ovphysx/api.py: cd0dd9890998571a97df9fe5f04610e4d654082e14d7ebab93a238a3b8777960
- isaaclab_ovphysx/sensors/contact_sensor/contact_sensor.py: 7fbc14a9f5d37aff3c267a51c14e1e8ad50aee8b03eecd2485c27d8f7f2b26bf
- isaaclab_ovphysx/sensors/contact_sensor/kernels.py: 8d5d9625fe674df26d9111b4d2f6f287aba9e9eeccb0f6b9d98fa3517a214894

## 7. 与无接触轨迹归因的关系

这次发现解释的是 C0 的“检测释放时刻”异常，不解释 Gate 0/T1 的全部位置差异。已有无接触诊断仍成立：

- 左手 small-step 的 MuJoCo 干摩擦从 1x/0.5x/0x 改变时，位置最大差为 2.109/1.047/0.851 mm；这是局部干摩擦因果贡献证据。
- 关闭干摩擦后的阻尼调整，让指定小指窗口改善约 82%，但拇指末端窗口仅改善约 4%；不能用一个阻尼原因解释全部残差。
- 同一 FK 模型回放基本解释指定拇指片段的位置差，支持从关节动态响应继续查起。
- R1 四类参数仍 UNMAPPABLE，Freeze B 仍 INCONCLUSIVE；没有证明两个后端的摩擦、惯量或求解器语义完全对齐，也没有完成 T1 全场景因果归因。

详见 MUJOCO_FRICTION_DIAGNOSTIC.md、MUJOCO_TOTAL_VISCOUS_DIAGNOSTIC.md、OVPHYSX_EFFECTIVE_READBACK.md、OVPHYSX_FRICTION_FREEZE_B.md。

## 复现与产物

在仓库根目录，用新的、尚不存在的输出文件执行：

```powershell
.\.venv\Scripts\python.exe scripts\diagnose_contact_release.py --bundle results\contact-c0-bundle-17eccdb-20260831-a4 --output results\contact-release-diagnosis-NEW\diagnosis.json
.\.venv\Scripts\python.exe -m pytest tests\test_contact_release_diagnosis.py tests\test_contact_metrics.py -q
```

本次验证 9 passed。新增测试覆盖：门槛扫描不能改写原始样本、缺失事件保持 None、去抖排除短暂断触并回溯至候选区间、几何 crossing 仅在返回段插值。

- Input bundle root: 7f69ff9416c755401a7b2915684c457930feb13cb0803effc2955c90468dd69d
- Script SHA-256: 11ba0109b72e8833a817176fd8f489951b944ede1c1c6231f9e3aecb89b8fc4b
- Final diagnostic: results/contact-release-diagnosis-20260929/diagnosis-v2.json
- Final diagnostic SHA-256: 8e848c05739f946b6d9264028b5bafcdc85624c988d9066ec0028a1fc667abfa

初版 diagnosis.json 保留为中间分析，v2 增加稳态尺度统计、按 dt 缩放门槛的探索性对照及 fresh-repeat 完整样本核对。最早一次脚本在严格记录验证器拒绝替换 pair_active 后终止、未生成输出；随后显式使用仅供分析的副本，正式记录不变。

科学表述应更新为：C0 的冻结信号定义没有通过 dt-halving 门槛；新增离线证据指向接触报告尺度与检测门槛的相互作用，尚不能认定接触动力学本身存在 5.5 ms 的实际分离偏移。
