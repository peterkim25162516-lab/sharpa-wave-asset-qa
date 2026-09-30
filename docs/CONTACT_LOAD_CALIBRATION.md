# 接触读数的时间步换算：已知载荷与接口交叉实验

日期：2026-09-29。性质：新增诊断，不属于正式 C0 验收或其重跑。

最新进度补充：[实际 ContactSensor a2 对照](CONTACT_SENSOR_PASSTHROUGH_DIAGNOSIS.md) 已完成，原生读数和传感器输出在全部样本中精确相同，异常尺度被原样传递。下文关于 a1 失败、尚待对照的表述为历史状态。

最新进度：[GPU 24-case 矩阵](CONTACT_GPU_MATRIX_DIAGNOSIS.md) 也已完成并独立核验，异步符合 mg、纯同步呈 mg×dt；实际 ContactSensor 对照因新增 fixture 缺少 /World 父节点而在采集前失败，仍待完成。下文 CPU 数据及当时的范围说明保留。

## 结论

在固定 ovphysx 0.4.13 的原生 CPU 接口中，已复现 **step_sync 路径的接触力读数依赖旧时间步** 的行为：

- 新实例只调用 step_sync：read_force_matrix 返回的数值等于静态支持力乘以当前 dt，表现为冲量尺度。
- 调用 step + wait_op：读数正确等于静态支持力。
- 先异步步进、再使用不同 dt 的 step_sync：读数等于支持力乘以「当前 dt / 上一次异步 dt」。

因此问题已从“可能的单位问题”定位到原生同步步进与接触读数换算之间的状态关联。此结论是黑盒因果对照的结果；没有读取或定位原生 C++ 内部变量／具体代码行。

该行为不需要 Sharpa 手、控制器、IsaacLab ContactSensor 或我们的适配器即可复现。它与 C0 使用 step_sync、原始信号随 dt 减半、固定检测门槛导致 5.5 ms 偏移的证据链一致。

**更新：后续原生 GPU 交叉实验也已复现该行为。** 通过正常 Slurm 调度恢复访问后，两次 fresh-process 共 12 个受控阶段全部符合旧 dt 假设，完整样本逐字节一致；详见 [GPU 诊断](CONTACT_GPU_CROSSOVER_DIAGNOSIS.md)。本页下方 24-case 矩阵和 14 阶段数据仍专指 CPU。GPU 全矩阵、C0 ContactSensor 封装对照和独立修正 campaign 尚未完成，不能记成正式 C0 已通过。

## 为什么这个实验能区分原因

使用 1 kg 或 2 kg 的球，静止放在水平盒子上，重力 9.81 m/s²。无其他外载荷时，稳定的垂直支持力应为 mg，即 9.81 N 或 19.62 N。

原生 tensor binding 读取实际质量、位置和速度；只在最后半秒静态窗口分析。判据在开始前由脚本及 protocol.json 固定：质量误差 <1e-5 kg、线速度 <1e-3 m/s、垂直位置范围 <1e-4 m，球心高度在 0.045–0.055 m。读数符合 mg 或 mg×dt 的相对容差均为 2%。

每个案例从新进程开始，不依赖此前运行状态。CPU 模式显式启用且 CUDA_VISIBLE_DEVICES 为空；无 Kit、渲染器、相机。fixture 单独生成，不修改上游资产。

## 实验 A：已知载荷矩阵

2 个质量 × 3 个 dt × 2 种步进接口 × 2 次 fresh repeat，共 24/24 案例完成。

1 kg 球的原始矩阵向量范数中位数：

| dt | step_sync 读数 | step + wait_op 读数 | 应有支持力 |
| --- | ---: | ---: | ---: |
| 2 ms | 0.0196200013 | 9.81000042 | 9.81 N |
| 1 ms | 0.00981000066 | 9.81000042 | 9.81 N |
| 0.5 ms | 0.00490500033 | 9.81000042 | 9.81 N |

2 kg 的全部读数相应翻倍。同步接口的 12 个案例均符合 mg×dt；异步加等待的 12 个案例均符合 mg。相对预测误差最大为 6.71e-8，远小于预先规定的 2%。

两次独立重复的完整稳态 samples 完全一致。read_net_forces 与 read_force_matrix 一致，水平方向近零；不是只读错一个过滤矩阵的形状。质量读回、静态位置和速度门槛全部通过。

## 实验 B：同一场景内交替接口，检验旧 dt 假设

在同一个已知载荷场景内按固定顺序切换接口，质量、重力、球体和盒子不变。下表中的预测在运行前写入脚本。

| 顺序 | 调用方式 | 当前 dt | 读数 / mg 的预测 | 实测 |
| --- | --- | ---: | ---: | ---: |
| 1 | sync，新实例 | 2 ms | 0.002 | 0.00200000013 |
| 2 | async + wait | 2 ms | 1 | 1.00000004 |
| 3 | sync | 1 ms | 0.5 | 0.50000002 |
| 4 | sync | 0.5 ms | 0.25 | 0.25000001 |
| 5 | async + wait | 0.5 ms | 1 | 1.00000004 |
| 6 | sync | 2 ms | 4 | 4.00000017 |
| 7 | async + wait | 2 ms | 1 | 1.00000004 |

两次新进程独立重复：14/14 阶段均符合预测，完整 samples 完全一致，报告的最大线速度为 0。球没有运动、载荷没有改变，读数却随调用历史变化，排除了“支持力真的因为 dt 变了而改变这么多”作为该最小实验的解释。

这也说明 **不能统一给所有读数补一个除以 dt**：新实例纯 sync 时看似能修正，但混合调用时还会受到上一次 async dt 的影响。更合理的候选规避方案是使用经过校准的 step + wait 路径；尚未替换正式适配器或重跑 C0。

## 对之前结论的修正

之前“释放时刻对步长敏感”的表述应明确为“按冻结接触信号定义检测到的释放时刻对步长敏感”。目前不能把 5.5 ms 直接描述成球真实分离时刻的差异。

本次没有推翻 Gate 0/T1 的无接触位置差异，也没有证明 GPU C0 的全部差异都由这个问题造成。正式 C0 仍为 VALID / INCONCLUSIVE，pass_ready=false；旧 evidence 和阈值均不变。

## 证据核验与日志边界

原始 evidence 已取回本地 ignored results/contact-load-calibration-20260929-a1/。独立校验器验证：

- 载荷矩阵 74 个 payload，接口交叉实验 21 个 payload；文件集合、大小、SHA-256 与远端清单完全相同。
- 24 个载荷案例、14,024 个稳态样本、14 个交叉阶段，从原样本重新计算结果，不仅信任 worker summary。
- 质量／位置／速度准入、两种力接口一致性、预测比值和 fresh-repeat 完全一致性通过。

日志并非 warning-free：包含 CUDA 不可用并按 CPU-only 处理、GPU broadphase 回退，以及 cooking 插件 registry/UJITSO 初始化警告／错误级日志。所有 worker 退出码为 0，primitive sphere/box 的质量、静态平衡、接触与位姿读回均通过上述独立核验。这些诊断不能套用正式 C0 的 runtime-log gate 宣称正式有效。

代码／清单 SHA-256：

- probe_contact_load_calibration.py: 4277d8041a336d67fda1ebe4dea390e513ab9af420f21e833f453ef714870a25
- probe_contact_dt_crossover.py: 9fc2225297f8af0f27b6c060ad6b05f357c1258056fd7b31618dc6ac1e993b71
- 安装包 api.py: cd0dd9890998571a97df9fe5f04610e4d654082e14d7ebab93a238a3b8777960
- 载荷实验 inventory.json: 67091c120ae98cd7fd1ef77c79f09f9f4879fe72178a9716b4f6c59d82367133
- 交叉实验 inventory.json: 0f4c517c373094770d974a8bf019329ccfa2b5fbb9c35f19e3ad174c086c1c41

本地核验：

```powershell
.\.venv\Scripts\python.exe scripts\verify_contact_load_calibration.py --root results\contact-load-calibration-20260929-a1
```

在安装固定 ovphysx 0.4.13 的 Linux CPU 环境中复跑（输出路径必须不存在）：

```bash
CUDA_VISIBLE_DEVICES='' python -B scripts/probe_contact_load_calibration.py --output results/load-calibration-new
CUDA_VISIBLE_DEVICES='' python -B scripts/probe_contact_dt_crossover.py --fixture-script scripts/probe_contact_load_calibration.py --output results/dt-crossover-new
```

GPU 运行前必须重新确认空闲卡；现已完成 GPU smoke 与受控交叉对照，未完成项见上方更新。当前没有外部发布、push 或 PR。
