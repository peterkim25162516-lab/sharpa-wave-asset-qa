# GPU 完整载荷矩阵与传感器对照进展

2026-09-29。新增诊断；非官方、simulation-only。不是正式 C0 的替换结果。

最新更新：a1 初始化问题已在独立 a2 修复，实际 ContactSensor 对照已核验通过，详见 [传感器运行报告](CONTACT_SENSOR_PASSTHROUGH_DIAGNOSIS.md)。下文保留 a1 失败时的历史状态。

## 已完成：24-case 原生 GPU 载荷矩阵

2 个质量（1/2 kg）× 3 个 dt（2/1/0.5 ms）× 2 种接口（sync/async + wait）× 2 次 fresh-process 重复，共 24/24 成功采集。

实验使用原生 ovphysx 0.4.13、A800、显式 DirectGPU、CUDA 张量；Python 3.12.14、numpy 2.5.2、warp-lang 1.13.0、packaging 23.2。无 Kit/renderer/camera。球体／盒子沿用已固定 fixture。先创建 bindings 并首次读张量预热，再执行指定步进接口 1 s，分析最后 0.5 s。

协议运行前已固定：异步读数与 mg 比较，容差 2%；纯同步的初始尺度仅作描述，不预先假定 GPU 自动预热会设定哪一个换算分母。

### 实测结果

1 kg 球，应有支持力 9.81 N：

| dt | 纯 sync 原始读数 | async + wait 原始读数 |
| --- | ---: | ---: |
| 2 ms | 0.0196200013 | 9.81000042 |
| 1 ms | 0.00981000066 | 9.81000042 |
| 0.5 ms | 0.00490500033 | 9.81000042 |

2 kg 读数相应翻倍。全部静载准入通过：质量误差 <1e-5 kg、速度 <1e-3 m/s、稳态 z range <1e-4 m、球心 z 在 0.045–0.055 m。12 个异步案例的力读数均符合 mg；12 个同步案例的观测结果均表现为 mg×dt 的尺度。

24 个独立进程，28,000 个原始样本；对应的 12 对 fresh repeat 完整 samples 文件逐字节相同。原始样本的有限数值、形状、时间轴和 summary 重算核验通过。

结合此前的 [GPU 接口交叉实验](CONTACT_GPU_CROSSOVER_DIAGNOSIS.md)，结论不只是“同步读数少了一个除以 dt”：同一实例混合调用时，同步读数还随上一次异步 dt 改变。因此不应把统一除以 dt 当成通用修复。

## 传感器源码审计：证据成立，但不冒充运行验证

固定 IsaacLab commit `ffff603eafc6b74264a5261cc0183d6a65390d78`：

- ContactSensor._update_buffers_impl 调用原生 read_net_forces / read_force_matrix。
- update_net_forces_ovphysx_kernel 将原生矩阵复制至 force_matrix_w；这条赋值路径没有乘除 dt。
- C0 bridge.filtered_force_matrix 调用 sensor.update(dt, force_recompute=True)，然后返回 sensor.data.force_matrix_w；这里也没有乘除 dt。
- C0 的配置为 update_period=0、history_length=0、单 target filter、debug_vis=False。

源文件 SHA-256 与原诊断记录一致：ContactSensor 为 `7fbc14a9f5d37aff3c267a51c14e1e8ad50aee8b03eecd2485c27d8f7f2b26bf`；kernels 为 `8d5d9625fe674df26d9111b4d2f6f287aba9e9eeccb0f6b9d98fa3517a214894`。

## 未完成：实际 ContactSensor 对照，attempt a1 保留失败

另行冻结真实传感器测试脚本，计划在相同原生 binding 上比较 native-before、公开 sensor.data.force_matrix_w、native-after，并在 async 2 ms → sync 1 ms → async 1 ms 下检查 1/0.5/1 比值和两次独立重复。

实际 a1 的第一个 worker 在 Sdf.CopySpec 时退出：目标 root layer 尚无 `/World`，不能向其添加 Ball/Ground。发生在传感器创建、context.reset 和正式接触轨迹之前；这是新增测试脚本的场景初始化遗漏，不是物理后端实验失败，也不产生“传感器原样传递”的运行结论。

错误日志、protocol、fixture、provenance、error.json 均保留并取回。该 attempt 已停止，没有改编号重试，没有覆盖失败目录，分配已释放。后续需先补建目标 root layer 的 `/World`、验证场景复制，再以明确的新源码／attempt 运行；当前没有修改正式 C0 适配器。

## 核验与复现

- 矩阵源码提交：`ccaa4bf`。
- probe SHA-256：`eb9a485969be51e02bbaf7923cdde2fdc68d8ed08a80e2bc103f8eaaaa85f051`。
- 矩阵 inventory SHA-256：`f09ac31ad0cb31ae62ff9ee92b2dcb7bcda1cffae82fea9d2ddabbbf8c7cac3e`，exact inventory 145 payload 文件，大小／hash 全部通过。
- 单文件传输归档 SHA-256：`f142d80dfafccbe46aa51ebfd8c6eebb071c46ef829e2eb38867909ab3093a8a`；解压前验证 hash 和路径。
- 完整核验目录：ignored `results/contact-gpu-matrix-20260929/verified-transfer/matrix-a1`。最早逐文件下载因延迟中止，仅停止本次自己的 scp 进程；上一级留下的部分下载不用于分析，未删除证据。
- 传感器测试源码提交：`7318f4a`，probe SHA-256 `861478b07c281aa5eb9ada48e76521dc32d74adb9715daecac103fe15ade92bf`。
- 失败证据：ignored `results/contact-sensor-diagnosis-20260929/sensor-a1`。所有带服务器路径／进程信息的原始日志与 provenance 均保持私有。
- 本地矩阵校验器的正常重算、篡改拒绝测试，加上交叉校验、原接触诊断／指标测试，合计 15 passed。

```powershell
.\.venv\Scripts\python.exe scripts\verify_contact_gpu_matrix.py results\contact-gpu-matrix-20260929\verified-transfer\matrix-a1
.\.venv\Scripts\python.exe -m pytest tests\test_contact_gpu_matrix_verifier.py tests\test_contact_gpu_crossover_verifier.py tests\test_contact_release_diagnosis.py tests\test_contact_metrics.py -q
```

日志仍有 cooking registry Error／UJITSO 警告；不宣称无警告，也不将诊断实验计入正式 C0 runtime gate。两项运行均走 Slurm，启动前严格检查空闲，未触碰外来任务、sudo、驱动或禁用存储。

正式 C0 保持 VALID / INCONCLUSIVE、pass_ready=false；Gate 0/T1 保持 DIVERGENT。本轮未证明跨模拟器接触一致，也没有完成修正 campaign。只作本地提交，没有外部发布、push 或 PR。
