# ContactSensor 运行对照：异常尺度被原样传递

2026-09-29，独立诊断 a2。非官方、simulation-only，不替换正式 C0。

## 结论

在固定的 IsaacLab / ovphysx 环境中，实际创建 ContactSensor 并读取公开的 `sensor.data.force_matrix_w` 后，得到的数值与其原生 ContactBinding 完全相同。同步步进中的旧 dt 尺度效应可以穿过传感器封装，封装没有把它纠正，也没有额外制造一个尺度变化。

因此，最小实验中的问题已定位到**原生 OVPhysX 同步步进与接触报告换算的交界处**，而不是必须由 Sharpa 资产、手控制器或 ContactSensor 的额外换算才能产生。内部 C++ 的具体代码行仍未定位；不能将此扩大为整个引擎有问题。

## a1 失败的处理

a1 在场景复制前缺少 root-layer `/World` 父节点。a2 仅修复 fixture 复制函数：当目标 root layer 没有 `/World` 时，先创建 Xform 父节点，再复制 Ball/Ground；不复制 fixture 的 PhysicsScene，以免与 SimulationContext 的场景重复。

使用同一固定 Python/USD 环境进行无 GPU 的 preflight：在空 stage 中复制成功，质量为 1 kg、Ground 为 Cube、没有额外 PhysicsScene。随后冻结提交 `83b368f` 并使用独立 a2-source/sensor-a2 目录。a1 五个文件的 SHA-256 在运行后再次读取，与之前记录一致，旧失败记录未被改写。

## 实验设置

- 固定 IsaacLab `ffff603eafc6b74264a5261cc0183d6a65390d78`、ovphysx 0.4.13，原 C0 的隔离环境。
- 实际 kit-less SimulationContext 与 ContactSensor，不是 mock、抽出的 kernel 或重写的传感器。
- 原生 1 kg 球／固定盒子 fixture，重力 9.81 m/s²；不是 Sharpa 手的完整 C0 场景。
- ContactSensor 配置与 C0 对应：update_period=0，history_length=0，debug_vis=False，单 Ground filter。
- 对每个样本：native-before → sensor.update(dt, force_recompute=True) → 公开 force_matrix_w → native-after。三份结果均保存。
- context.reset、绑定及首次张量读取完成后，先以 async 2 ms 建立已知基准，再切换 sync 1 ms，最后 async 1 ms。
- 步进使用底层 SDK，受控更换 dt；并非通过完整 C0 的 SimulationContext.step／手控制器链路，因此不冒充整套适配器端到端回归。
- 两次 fresh process；阶段预期比值 1/0.5/1、容差 2%，在运行前固定；每个 worker 限时 90 s。

## 实际结果

| 阶段 | 原生接口与 dt | native-before / mg | ContactSensor / mg | native-after / mg |
| --- | --- | ---: | ---: | ---: |
| 建立基准 | async + wait，2 ms | 1.000000043 | 1.000000043 | 1.000000043 |
| 改用同步 | sync，1 ms | 0.500000021 | 0.500000021 | 0.500000021 |
| 恢复异步 | async + wait，1 ms | 1.000000043 | 1.000000043 | 1.000000043 |

两个进程、6/6 阶段符合预期，共 1,500 个样本。所有样本的三份向量逐元素精确相等；对应阶段的两次 fresh-process 原始文件逐字节一致。质量／静态高度／速度准入通过，数据全部有限。传感器读取前后的原生值不变，未观察到此次读取顺序改变报告尺度。

场景无 Camera，未加载 Kit/renderer/replicator 模块；有 CUDA 张量与当前进程 GPU 使用证据。仍有 cooking registry Error 和插件警告，不能声称日志无错误级消息或正式 C0 runtime gate 已通过。

## 科学结论与下一步边界

证据链目前包括：旧 C0 原始信号与门槛的离线诊断、CPU 24-case 载荷矩阵、CPU/GPU 接口交叉、GPU 24-case 载荷矩阵，以及实际 ContactSensor 运行对照。它们共同支持“接触报告尺度与固定事件门槛相互作用”的解释，而非直接把 5.5 ms 说成物体实际分离延迟。

下一步若要验证规避方案，需单独冻结新的 C0 campaign，以经过校准的 async + wait 路径重新采集完整手部场景，再评估 dt-halving 与跨引擎差异。不能仅凭此诊断改判，也不能给旧读数统一除以 dt。

本轮没有修改正式适配器，没有重跑完整 C0，没有更改原始门槛。正式 C0 仍 VALID / INCONCLUSIVE、pass_ready=false，Gate 0/T1 仍 DIVERGENT。无硬件、Sim2Real 或官方背书；无外部发布、push、PR。

## 核验资料

- probe SHA-256：`d027e3b326312767965f8669700f912edf6738e6e5fcfa8a6ebc6f47741c04dc`。
- ContactSensor SHA-256：`7fbc14a9f5d37aff3c267a51c14e1e8ad50aee8b03eecd2485c27d8f7f2b26bf`。
- kernels SHA-256：`8d5d9625fe674df26d9111b4d2f6f287aba9e9eeccb0f6b9d98fa3517a214894`。
- inventory SHA-256：`95afd0af389c29ad925d97c1c56ae208de64f521a03e4f6ad255f23c38c20dac`；exact inventory 15 payload，大小和 SHA-256 通过。
- 传输归档 SHA-256：`011703540cdb48218fa61e515288788bfdf74e89da5fad6fd353d3f525675548`，解压前检查 hash 与路径。
- 原始资料只在 ignored `results/contact-sensor-diagnosis-20260929/a2-verified`；服务器／进程信息保留为 private evidence。
- 独立校验器重算三个输出、稳态、预测、时间轴、有限值与重复性。正确输入、篡改、额外文件、重新封装后输出不一致的四类测试，加上已有诊断测试合计 19 passed。
- GPU 分配成功后两次检查严格空闲，任务结束已自动释放；没有更改驱动或外来任务。

```powershell
.\.venv\Scripts\python.exe scripts\verify_contact_sensor_passthrough.py results\contact-sensor-diagnosis-20260929\a2-verified\sensor-a2
.\.venv\Scripts\python.exe -m pytest tests\test_contact_sensor_passthrough_verifier.py tests\test_contact_gpu_matrix_verifier.py tests\test_contact_gpu_crossover_verifier.py tests\test_contact_release_diagnosis.py tests\test_contact_metrics.py -q
```
