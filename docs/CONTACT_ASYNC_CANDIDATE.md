# C0 async + wait 候选：调用链先导验证与完整复验准备

2026-09-30。状态：COMPLETED_VALID_DIVERGENT。

## 最终状态

候选完整复验已经完成并严格归档：32/32 案例，32 个 fresh process，44/44 joints，10/10 frames，`VALID / DIVERGENT`，`pass_ready=false`。OVPhysX 自身 dt-halving 释放差由 5.5 ms 降至 0.5 ms，但跨引擎四项指标仍超阈值。GPU 已释放，不再有待执行的候选案例。完整解释、数值和证据身份见 [最终结果](CONTACT_ASYNC_RESULT.md)。

下文保留实施过程与先前阶段状态，属于历史记录；“尚待接入／尚未执行”等表述不代表本节最终状态。

## 完整复验执行协议

执行 corrigendum（2026-09-30）：首轮源码 `9ae68f7` 的 local-a1 首个 worker 完成 3500 步，但 launcher 二次进程归属校验仍按旧字段集合读取，退出码 125、整批完成数 0。修正仅让归属校验识别显式候选字段，不放松 PID/父进程验证；新增两种模式的归属与拒绝外来进程测试，本地组 18 passed。该失败目录保持不变且不作为 active evidence；后续必须以修订后的同一源码重采两端。首轮没有启动远端轨迹。

本次显式候选入口已接通远端采集与整批 finalizer。远端仍复用原完整 launcher 的资产、依赖、native CCD helper 构建、三次逐案例空闲检查、进程归属、超时、快照、日志和精确清单保护；只有指定 `--experimental-campaign contact-c0-async-v1` 才切换独立 worker。默认 C0 的科学步进未变。

候选固定 Server 6 的物理 7 号卡 UUID。Slurm live job 必须属于 exampleuser、处于 RUNNING、单节点／单任务、`gres/shard:idx7:1`；可见设备必须恰好一个、query index 为 0、UUID 为已固定设备。私有 provenance 单独保留 physical/query 编号、调度原文和继承环境；worker 使用由唯一设备 UUID 校验绑定的逻辑编号 0，不把物理 7 当成进程内 7。该 shard 并非整卡独占，仍逐案例检查是否出现其他进程。只读实测 job 38696 验证了上述映射并释放，没有执行轨迹。

第二次执行 corrigendum：`14f8b96` local-a2 完成并验收 16/16；remote-a2 job 38697 完成 native helper 准备，但首例在物理仿真前被既有 adapter 拒绝，因为其 CUDA_VISIBLE_DEVICES 合约仅接受数字，不接受 UUID 字符串。没有轨迹产生，完成数 0，377 秒 allocation 后释放。修订只将 worker 选择器改为经 Slurm 唯一 UUID 绑定的逻辑 0；不改变 adapter／仿真语义，不绕过 UUID 校验。两端 a2 保持不变，不作为最终 active；以新提交重新运行两端。

`finalize_contact_async.py` 显式选择候选身份，重用严格 32 案例验证与科学评估，公共报告标记干预，且声明不替代原 C0。不改 manifest、阈值、旧证据或既有 Gate 0。HOME 保持账户原值，各类可写 cache 继续由显式 XDG/CUDA/工具路径限定。相关 116 项测试通过；完整采集仍须执行，不能把准备测试写成轨迹完成。

## 最新实现进度：本地候选采集与只读验收

`run_contact_c0_local.py` 新增显式 `--experimental-campaign contact-c0-async-v1`，为 16 个 MuJoCo 案例使用独立候选 worker。默认仍为原 C0。源码／资产检查、逐案例 fresh process、归属检查、超时、输出预算和失败证据保留机制均保留；成功／失败状态与精确清单都标明候选身份。未知 campaign 在创建输出目录前被拒绝。

新增 `validate_contact_async_local.py`：检查本地整批的 launcher 身份、精确目录／清单、运行日志、案例覆盖、进程身份和候选 envelope。它仅返回 local_half_only / NOT_EVALUATED，不生成跨后端科学判定。源码 archive hash 来自真实 Git archive，不使用占位 hash。

共享 finalizer 的内部加载函数仅在显式选择候选时接受新增字段；原正式入口不启用候选。候选 OVPhysX 整批加载目前明确拒绝，直到 Slurm UUID 身份绑定与远端采集验证完成，不能把这个本地验证入口当成完整 campaign finalizer。

本地 launcher 单测 17 passed，包含模拟 16 案例派发、未知身份拒绝及两种模式的失败记录。模拟测试不是实际 MuJoCo 轨迹证据。本轮没有启动新 16/32 案例采集，也未使用 GPU；原 manifest hash 与正式判定不变。

验证记录：全套回归 988 passed；其运行开始后补充的候选失败记录参数化用例另由相关测试组覆盖，最终相关组 93 passed。只读验收 CLI 的 `--help` 与 `git diff --check` 通过。

## 最新实现进度：候选 bridge / worker 已接入

新增 `AsyncCandidateBridge` 保持 write_data_to_sim → SimulationContext.step(render=False) → articulation.update 顺序，逐步应用已经先导验证的 async + wait 干预。`AsyncCandidateContactAdapter` 显式使用该 bridge，并补回自定义 bridge 接口可能跳过的 CUDA、无 Kit、无 renderer、无 Camera、仅 ContactSensor 安全检查。

独立入口 `scripts/run_contact_async_worker.py --backend mujoco|ovphysx` 选择 `contact-c0-async-v1`。原来的两个正式 C0 入口未改为异步。MuJoCo 候选案例仍使用原 adapter，不做 OV 步进干预；只是同样标记为新 campaign，避免混入旧 campaign。

候选输出新增 private-campaign.json：绑定 case、源码、manifest 语义 hash、原始 run／private adapter／process 文件 hash。OV 的 runtime fingerprint 额外声明 intervention ID、helper hash、步数和审计 hash；私有记录保存每步调用／等待次数、dt 和原生仿真时间。缺失等待、重复步进、时间不按 dt 前进、错误 helper 身份均被拒绝。

新增 `load_candidate_case` 严格加载候选 envelope、重算文件／fresh-process／runtime hashes，并复用原 trace 验证。这只是**单案例的身份和数据绑定检查**，不是完整 launcher 身份、native helper、日志审计、exact campaign inventory 或科学判定的替代品。旧 formal finalizer 的闭合字段集合仍会拒绝新私有字段；未削弱其规则来强行接受候选。

本轮 145 项测试通过（候选 bridge／审计、原 worker、原 adapters、原 C0 finalizer）；候选 CLI 可加载。原 contact_c0.json 文件 hash 仍为 `73cc9dedffc1ea0600052fc1b78696eda292f1f34c7987ec1f433043e65bf215`。新增 bridge 目前通过模拟对象的调用顺序／安全测试，真实 GPU 证据仍限于上一轮 SimulationContext primitive 先导，不能写成完整手部 bridge 已运行。

尚待接入：Slurm 下的完整采集／监测、候选 campaign 汇总 finalizer 和双端 32 案例 fresh 重采。本轮未占用 GPU，未创建新的远端采集目录，也未产生新的科学结果。

## 已完成

新增 opt-in `async_wait_intervention`，在单线程 fresh worker 内临时拦截 SDK 实例的 `step_sync`，转为 `step` + `wait_op`。保留 SimulationContext.step 和原 manager 的运动学、仿真时钟更新路径；不直接跳过这些调用层。作用域退出（含异常）后删除临时实例覆盖并验证恢复。拒绝嵌套覆盖、无效 dt、重复步进或缺少步进。

默认 C0 bridge、默认 worker、原 manifest、阈值、完整采集与 finalizer 均未改变。本轮仅为候选先导，不是正式修复发布。

真实 GPU 先导沿用 1 kg 球／盒子及实际 ContactSensor。对照三阶段：原生 async 2 ms → 通过候选作用域的 SimulationContext.step 1 ms → 原生 async 1 ms。第二阶段仍由 manager 调用 step_sync 名称，但实际执行路径已被显式转换为 async + wait；不能把它标成未经修改的同步接口结果。

与前一次实际传感器 a2 的 1 / 0.5 / 1 相比，此次读数 / mg 为 1 / 1 / 1（每阶段约 1.00000004277）。两个 fresh process、共 1,500 个样本，native-before / public-sensor / native-after 全部精确相等，重复阶段原始文件逐字节相同。500 个候选步进样本的审计均为 calls=1、waits=1、dt=1 ms。质量、静态位置／速度准入及有限值检查通过。

这说明候选能在真实 SimulationContext 调用链中消除本最小实验的读数减半，而非仅在直接原生调用时有效。仍未经过 Sharpa 手、控制器及完整 C0 reset/trajectory/collector 链路，不能据此推断正式 5.5 ms 已改善。

## 新完整复验的边界（尚未执行）

拟定独立标识 `contact-c0-async-v1`，与原 C0 a4 分开存放和解释。

1. 复用原 contact_c0.json 的全部场景／阈值语义：左右手、contact/sham、dt/dt-half、两次 fresh repeat；共 16 MuJoCo + 16 OVPhysX。重采两端，避免把旧源码身份混入新 active campaign。
2. 只有 OVPhysX 的步进干预发生变化，保留 write_data_to_sim → SimulationContext.step(render=False) → articulation.update 的顺序。需要新 bridge 接入及逐步调用审计，不得偷偷更改默认实现。
3. 单独的实验协议／private evidence 必须声明 intervention ID、helper hash、原始 manifest hash、源码／环境／资产身份；修改后的采集、验证和 finalizer 必须认识该新身份。未完成这些防混淆工作前，不调用旧正式入口假装普通 C0。
4. 保持原 pair_active 的 1e-4 N 门槛和 4 ms dt-halving 判据，不根据新结果调参；保持映射、碰撞对、质量／惯量、CCD、日志、NaN、repeatability、exact inventory 与隐私检查。任何不通过均如实报告。
5. Slurm 物理卡与逻辑 CUDA 重编号需用 UUID 显式绑定。旧 launcher 的“物理编号直接等于可见编号”假设不能原封照搬；先验证调度下的身份与进程监测，不得放松空闲检查来启动。
6. 原正式 C0 a4、先导失败 a1、诊断 a2、CPU/GPU 矩阵与交叉数据保持不变；候选先导仅作准备证据，不进入完整 C0 active traces。

尚未完成的实现（以顶部最新进度为准）：Slurm 适配的完整采集封装、campaign 级身份／环境／日志／inventory 闭合验证、对应 finalizer、32 个案例执行及结果归档。候选 bridge/worker 和单案例绑定已实现。此文不是“已经一键可跑”的承诺。

## 可核验记录

- 候选实现／probe 提交 `e47dbec`；固定 launcher 提交 `7634ed0`。
- helper SHA-256：`8a4f8b091eb1112550f4b2a4b252644f5be5e18efe277b2e0fc6205943a34995`。
- probe SHA-256：`3ad0534f8c0cdf4a6ee4e460f61dc2ac484a207678a38486d4cd149ee12daa67`。
- inventory SHA-256：`a2f1a2f398fb84bb3e92f1d5ad84be5fad817c9bb3c7653e583daec9e765b1fa`；15 个 payload 的 exact inventory、大小与 hash 通过。
- 传输归档 SHA-256：`08cadf549dcfbbb8952e1c79eee858063183f2c1d85dd4cfa0a083fd528690b9`。
- 私有证据：ignored `results/contact-sensor-diagnosis-20260929/context-candidate-a1-verified/evidence`；验证器必须显式指定 `--context-candidate`，旧模式拒绝候选源码身份。
- 固定依赖和原生传感器源码与前一次 a2 相同。已有 cooking/plugin 日志限制仍适用，不声称满足正式 runtime-log gate。
- GPU 已释放；没有更改原 C0 verdict、外部发布、push 或 PR，无硬件／Sim2Real。
- 本轮单测与既有诊断校验测试合计 31 passed，含候选身份必须显式选择、缺少等待时拒绝、异常恢复和重复步进拒绝。

```powershell
.\.venv\Scripts\python.exe scripts\verify_contact_sensor_passthrough.py results\contact-sensor-diagnosis-20260929\context-candidate-a1-verified\evidence --context-candidate
```
