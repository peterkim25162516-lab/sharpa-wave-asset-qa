# GPU 接触读数诊断：物体没变，读数随步进历史变化

2026-09-29。新增原生 OVPhysX 诊断，不是正式 C0 重跑；非官方、simulation-only。

后续更新：GPU 24-case 矩阵已完成并核验；实际传感器对照 a1 在场景初始化失败。详见 [后续报告](CONTACT_GPU_MATRIX_DIAGNOSIS.md)。本文下方“尚未完成”描述保留为交叉实验完成时的范围。

## 结论（讲人话）

1 kg 球安静地放在盒子上，正常支持力应约为 9.81 N。GPU 实验中，球的质量、静态位置和速度没有变化，但换了步进接口和时间步长后，报告的接触力可以变成 4.905 N、2.4525 N 或 39.24 N。切回异步步进加等待后，又恢复为 9.81 N。

这说明至少在此最小实验里，变化的是**接触力报告的数值尺度**，不能解释成球真的突然受到四倍支持力却仍保持静止。原生 GPU 路径复现了 CPU 上相同的调用历史依赖：同步步进读数符合 `mg × 当前 dt / 上一次异步 dt`。

这是固定 ovphysx 0.4.13 的黑盒对照结果。尚未定位原生 C++ 的具体变量或代码行，不能据此把整个物理引擎或 Sharpa 资产判为有问题。

## 运行前固定的实验

- 源码提交 `2336f9d`，probe SHA-256 `7c3811b4eeba085a87e98f3e20abfab5f07349043fd48d33b0ec54903b8e072f`。
- Python 3.12.14；ovphysx 0.4.13；numpy 2.5.2；warp-lang 1.13.0；packaging 23.2。
- A800、原生 PhysX GPU dynamics、显式 DirectGPU 配置、CUDA 张量输出。无 Kit、renderer、camera、硬件实验。
- 1 kg 球／固定盒子沿用已校验的 CPU fixture，重力 9.81 m/s²，无 Sharpa 手或 IsaacLab ContactSensor。
- 自动首次张量读取预热完成后，显式 async 2 ms 建立基准；不假设 GPU 新实例的默认换算分母。
- 固定六阶段，两个 fresh process 独立运行，单 worker 上限 90 s。预测容差 2%。全部轨迹保存，不只保存符合预测的片段。
- 稳态条件：质量误差 <1e-5 kg，速度 <1e-3 m/s，z range <1e-4 m，球心 z 在 0.045–0.055 m。

本轮先做定向交叉实验，没有执行原计划的完整 24-case GPU 质量／步长矩阵。原因是 smoke 已验证 GPU async 静载，交叉顺序可直接检验旧 dt 假设并控制预热历史；不把未运行的矩阵列为完成。

## 两次运行的结果

| 顺序 | 接口 | 当前 dt | 读数 / mg 预测 | 实测读数 / mg |
| --- | --- | ---: | ---: | ---: |
| 1 | async + wait，建立基准 | 2 ms | 1 | 1.000000043 |
| 2 | sync | 1 ms | 0.5 | 0.500000021 |
| 3 | sync | 0.5 ms | 0.25 | 0.250000011 |
| 4 | async + wait，更新基准 | 0.5 ms | 1 | 1.000000043 |
| 5 | sync | 2 ms | 4 | 4.000000171 |
| 6 | async + wait | 2 ms | 1 | 1.000000043 |

12/12 阶段符合运行前预测。每阶段稳态速度、z range 都为 0，质量读回为 1 kg。两种接口 read_force_matrix / read_net_forces 给出相同尺度。共 3,500 个完整样本无非有限数值；两个 fresh process 的六份原始阶段文件逐字节一致。

## 对 C0 的意义和仍缺的证据

此前 CPU-only 的限制已经缩小：相同换算行为也存在于原生 GPU / DirectGPU 路径。结合旧 C0 接触信号在 dt 减半后约缩小到 0.498、固定门槛导致释放检测偏移，而几何分离差远小于 5.5 ms 的证据，接触报告尺度／检测门槛的解释更有支持。

但是本轮没有直接运行 C0 的 IsaacLab ContactSensor 封装，也没有重新采集 Sharpa 手接触轨迹。因此不能宣布旧 C0 的全部异常都已解决，更不能改判两个后端的接触行为一致。正式 C0 保持 VALID / INCONCLUSIVE、pass_ready=false；Gate 0/T1 的 DIVERGENT 不变。

下一步仍需：补齐 GPU 24-case 矩阵；以相同 fixture 对照 ContactSensor 和原生 binding；在获得充分校准证据后，单独冻结候选修正 campaign。候选规避方式是 `step + wait_op`，不是统一除以 dt。禁止覆盖旧证据或事后修改原门槛。

## 可核验性与限制

- 本地 ignored evidence：`results/contact-gpu-crossover-20260929/crossover-a1`。含服务器／进程信息的 provenance 和日志只留私有目录，不进入公开报告。
- exact inventory 23 payload 文件，大小和 SHA-256 均核对通过。
- inventory SHA-256：`40099a2b2945dc518a6f3bfc9a0468fde11ee15f819056b0db7642e93f459504`。
- 独立只读校验器从 raw samples 重算稳态、力比值、预测符合性与重复性，并核对固定 probe/API identity，不仅信任远端 summary。
- 日志含 cooking registry 的 Error、UJITSO／插件警告及预期 auto-warmup 提示；不声称 warning-free 或通过正式 C0 runtime gate。两 worker 均成功退出，CUDA 张量读取及进程 GPU 证据存在，未见 CPU fallback。
- 调度分配在启动前两次验证目标 GPU 空闲；运行完毕已释放，没有更改驱动或外来任务。
- 校验器含正确清单、篡改、额外文件、重新封装伪造 summary 四类测试。与已有诊断／指标测试合计 13 passed。

本地复核：

```powershell
.\.venv\Scripts\python.exe scripts\verify_contact_gpu_crossover.py results\contact-gpu-crossover-20260929\crossover-a1
.\.venv\Scripts\python.exe -m pytest tests\test_contact_gpu_crossover_verifier.py tests\test_contact_release_diagnosis.py tests\test_contact_metrics.py -q
```

本轮只作本地提交；没有外部发布、push 或 PR。
