# GPU 接触读数复现：待运行方案

2026-09-29，状态 NATIVE_GPU_AND_SENSOR_DIAGNOSIS_VERIFIED。GPU matrix、交叉及实际 ContactSensor a2 对照均已独立核验；a1 失败证据保持不变。详见 [传感器运行对照](CONTACT_SENSOR_PASSTHROUGH_DIAGNOSIS.md)、[矩阵报告](CONTACT_GPU_MATRIX_DIAGNOSIS.md)、[交叉报告](CONTACT_GPU_CROSSOVER_DIAGNOSIS.md)。正式 C0 修正 campaign 尚未实施。以下保留原计划和排查历史，不应把旧阻塞描述当作当前状态。

## 调度入口恢复与前述判断更正

本日后续检查发现服务器已启用 Slurm；普通 SSH 会话的 EPERM 不能证明 GPU/驱动损坏，也不足以要求管理员修复。通过正常 `salloc` 分配后再使用站点 `srun`，已成功访问 Server 3 GPU。下面“需要管理员”的旧判断被此发现取代，保留作为排查过程，不再是当前阻塞。

站点 `--gres=gpu:4` 表示物理 GPU 索引 4 的一个 shard，不是申请四张卡，也不保证整卡独占。分配内 nvidia-smi 可重编号为 0，必须核对 UUID。job 38664 查询到该卡 0 MiB / 0%；job 38665 的 compute-process 清单为空，两次检查后均自动释放分配。不能把这些历史读数当作后来启动时的空闲证明。

新隔离环境已按 Python 3.12.14、ovphysx 0.4.13、numpy 2.5.2、packaging 23.2、warp-lang 1.13.0 安装。工程脚本 `probe_contact_gpu_smoke.py` 使用 CUDA 张量、显式 DirectGPU 配置、首次张量读取预热后 async + wait；校验 1 kg 球的静载读数、稳态、进程 GPU 证据。启动脚本在同一分配内两次核对目标 UUID、显存、利用率与进程，失败立即退出。

job 38666 工程 smoke 已成功退出并自动释放分配：500 个 2 ms 步；稳态质量 1 kg、速度 0、z range 0、raw/mg=1.0000000427743831，约 9.81000042 N。CUDA 张量读取与进程 GPU 证据检查通过。脚本 SHA-256 为 `a98fa7250d434693259cef2cb7409fab88a995f33ac4a694c6085f49f4895909`。日志仍有 cooking registry/UJITSO 警告及预期的 auto-warmup 提示，不声称无警告。私有证据保存在 ignored results/contact-gpu-smoke-20260929，不能公开其中的进程／服务器信息。

该 smoke 只证明已知静载的 GPU async 路径可运行；不是 fresh-process repeatability、完整矩阵、同步／异步交叉试验、ContactSensor 封装对照或 C0 修正完成，不能据此确认 GPU 根因。正式 C0 仍为 INCONCLUSIVE。

取回后本地已验证 exact inventory（5 个 payload 文件）、文件大小及各自 SHA-256；inventory SHA-256 为 `53c29af4bef075662c28026a81f7fdeef558653d7451749aff657150b620530b`。相关既有诊断／接触指标测试 9 passed，新脚本通过 Python 编译与 Bash 语法检查。

## 当前阻塞

本轮通过已有 SSH key 登录 Server 6，GPU 清单与 compute-process 查询均以非零状态退出，输出 Failed to initialize NVML: Unknown Error。无法确认任何 GPU 满足 memory.used <=32 MiB、utilization.gpu=0、无 compute process。因此没有创建远端实验文件、占用 GPU、更改驱动或重跑 C0。

需要服务器的 GPU 可见性恢复。必须重新读取两类查询，并在启动前立即二次检查目标卡。NVML 失败不能当作“没有进程／GPU 空闲”。

### Server 3 备用节点检查（用户指定后）

已用已有 SSH key 登录 Server 3，核对 hostname=example-gpu-node-3、user=exampleuser。全卡查询、仅 GPU 0/1 查询、清空登录环境后直接运行 /usr/bin/nvidia-smi 均失败。内核驱动和 libnvidia-ml 链接版本均为 570.158.01；这不足以断言驱动损坏。

进一步只读检查得到具体阻塞：文件 ACL 允许 exampleuser 读写 /dev/nvidiactl、GPU 0/1 和 nvidia-uvm，但 strace 观察到当前会话打开 /dev/nvidiactl（O_RDWR 和 O_RDONLY）均返回 EPERM。说明当前会话在文件 ACL 之外仍受设备访问限制；具体是 cgroup/BPF 还是其他策略尚未确认。GPU 2–7 的 ACL 未授予 exampleuser 权限。

访问说明未提供 GPU 调度／授权会话的替代入口。本轮未尝试绕过设备策略，未使用 sudo，未修改权限或驱动，也未在 Server 3 创建项目文件／安装环境／启动计算。需要管理员检查 exampleuser 的会话设备访问策略及授权 GPU 0/1 的访问链路，恢复后重新验证空闲状态。用户报告有空闲 A800 与此账户访问阻塞并不矛盾。

## 已确认与待确认

- 已确认：固定 ovphysx 0.4.13 原生 CPU 路径的同步步进接触读数呈现沿用上次异步 dt 的行为；详见 CONTACT_LOAD_CALIBRATION.md。
- 待确认：同版本 GPU/DirectGPU 路径是否有相同行为，以及 C0 当前封装是否完整继承该行为。
- 正式 C0 仍为 INCONCLUSIVE，不改旧数据或阈值。

## 特别需要控制的 GPU 预热

本轮只读检查安装包 developer_guide.md 的 GPU Warmup and Determinism：

- device="gpu" 与 DirectGPU TensorAPI 是独立配置。
- IsaacLab 风格张量路径需在实例创建前设置 /physics/suppressReadback=true 和 /physics/suppressFabricUpdate=true。
- 首次张量操作可能自动执行 GPU warmup step。

因此不能把 CPU 新实例中的“默认换算分母为 1”的推断直接移植到 GPU。所有 warmup、binding 创建、首次 tensor read 必须记录顺序；在正式交叉段开始前，显式执行一个已知 dt 的 step + wait 建立可核验的基准历史。

## 复现顺序与判定

1. 校验固定依赖版本和源码 hash，使用新实验目录和 fresh process。
2. 实际 GPU 执行证据必须成立，CPU fallback、不可见 GPU 或无法确认设备均终止，不能算 GPU 结果。
3. 先做一个 1 kg sphere/box 工程 smoke，检查 runtime mass、pose、velocity、filter count 和每步 tensor 更新。失败时保留日志，修订代码后必须使用新的 attempt identity。
4. 运行 2 质量 × 3 dt × 2 step 接口 × 2 fresh repeats 的 24-case native matrix。稳态窗口及载荷准入沿用 CPU 诊断：质量误差 <1e-5 kg，速度 <1e-3 m/s，z range <1e-4 m，球心 0.045–0.055 m，mg 比例容差 2%。记录初始 sync 的尺度但不强制假定 GPU 的默认分母。
5. 完成 warmup 后，做两个 fresh-process 的受控接口交叉段：

| 调用 | dt | stale-dt 假设下 raw/mg 的预测 |
| --- | ---: | ---: |
| async + wait（建立基准） | 2 ms | 1 |
| sync | 1 ms | 0.5 |
| sync | 0.5 ms | 0.25 |
| async + wait（更新基准） | 0.5 ms | 1 |
| sync | 2 ms | 4 |
| async + wait | 2 ms | 1 |

6. 若 native GPU 复现，再用同 fixture 核对 IsaacLab ContactSensor 与原生 binding 的读数，记录读取次序，避免读取或预热本身成为干预因素。不得把尚未实现的封装比较记为完成。
7. 只有 GPU 支持且新控制路径通过校准，才提出独立 C0 修正 campaign。候选是 step + wait；不得直接统一对读数补除 dt，亦不得覆盖旧 a4 evidence。

若预测不成立，结果保留为不支持／未解决，检查 GPU 专有的数据更新与换算路径；不得只选符合预期的案例报告。已知载荷通过不等于 Sharpa 接触验证通过。

## 资源与边界

只使用一张经过严格确认的空闲 GPU，不抢卡、不等待外来进程、不杀外来进程。拟定新诊断硬上限 15 分钟 GPU wall time，单 worker 90 秒，数据不超过 100 MB；超时／运行错误保留证据并停止该 attempt，不自动更换编号。

不修改现有固定依赖、不使用 Kit/renderer/camera、不进入硬件实验。此轮未创建自动重试任务，未向服务器管理员发送消息，未对外发布。
