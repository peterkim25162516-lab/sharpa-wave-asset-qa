# 接触差异诊断：完整复验结果（2026-09-30）

## 一句话结论

**原先的 5.5 ms 步长敏感问题，定位到了接触读数／事件判定这一层；候选调用方式将它降到 0.5 ms。但两个仿真器并没有因此一致。** 完整候选实验的结论是 `VALID / DIVERGENT`，`pass_ready=false`。

这轮采集、验收、对比和归档已经完成。DIVERGENT 是如实记录的研究结果，不是必须继续调参消除的程序错误，也不是官方 bug 认定。

## 实际做了什么

- 独立实验标识：`contact-c0-async-v1`。只在 OVPhysX 中把每次 `SimulationContext.step(render=False)` 内的同步 SDK 步进替换成 async + wait；保留原来的关节写入、运动学更新和仿真时钟路径，每步记录一次调用、一次等待和 dt。
- 未修改场景、控制轨迹、资产、接触判定门槛或验收阈值。没有直接给最终数据乘系数、移动时间轴或挑选有利案例。
- 重新采集 MuJoCo 16 个、OVPhysX 16 个案例：左右手分别运行，contact/sham、2 ms/1 ms、两次 fresh process。active campaign 完成率 32/32，32 个独立进程，44/44 joints、10/10 distal frames。
- 覆盖 168,032 个采样点；16 对 fresh-process 重复的完整 samples（含各自后端原生观测）逐项精确一致。严格加载器的有限值、完整轨迹、参数、来源和清单检查通过，没有发现 NaN 或数值爆炸。
- 源码固定为 `b1c69f591cf02ba5865ce9892c366bbdeec2fc1b`。远端正常结束并释放 GPU；没有 push、PR、硬件或 Sim2Real。

## 为什么能定位到读数层

新旧 OVPhysX 的全部 16 个案例、84,016 个采样点，在下面字段的最大绝对差都为 **0**：关节位置／速度、目标位置、指端位姿、球心位置、有符号间隙、步号与时间。

但接触读数尺度发生了变化：在同一后端的新旧对照中，2 ms 案例的新／旧读数比中位数约为 500，1 ms 案例约为 1000，分别对应 `1 / dt`。该比值统计只使用旧读数大于 1e-7 的点，属于事后诊断筛选，不参与任何准入判据。结合此前已知载荷实验，这与旧同步调用路径的冲量尺度／报告归一化行为一致，不能把旧字段名称中的 N 当成已经验证的单位。

固定的接触判定门槛因此跨过了不同的采样点：每个 2 ms 接触案例改变 3 个 pair-active bit，每个 1 ms 接触案例改变 11 个；无接触对照没有 bit 变化。**这里不是球实际多移动了几毫秒，而是用不同尺度的读数判断“还在接触”时，判定时刻变了。**

MuJoCo 对照端的 16 个案例、84,016 个采样点，以及 case、execution、mapping、fixture readback、contact observation 字段也与原 C0 完全一致。

这一证据链将问题定位到该固定软件版本的调用路径／原生报告行为层，并不是对所有 PhysX 用法的结论；尚未定位到 SDK 底层 C++ 的具体缺陷代码行。

## 稳定性问题解决了什么

| 检查 | 原 C0 | 候选完整复验 | 原门槛 |
| --- | --- | --- | --- |
| OVPhysX 步长减半后的释放时差 | 5.5 ms | 0.5 ms | 4.0 ms |
| MuJoCo 步长减半后的释放时差 | 1.5 ms | 1.5 ms | 4.0 ms |
| fresh-process 重复 | 原结果保留 | 16/16 对 samples 精确一致 | 原精度／精确 bitset 判据 |

候选两端的 onset/release 合并最大步长敏感度为 1.5 ms。所有 admission、repeatability、dt-halving 检查均通过，因而可以继续做跨后端比较。

## 为什么最终仍然 DIVERGENT

共 24 个 gate checks：20 个通过，下面 4 个跨后端检查超过预先冻结的阈值。

| 跨后端指标 | 实测最大差 | 原门槛 |
| --- | --- | --- |
| 接触事件时刻 | 56 ms | 10 ms |
| 关节接触效应 | 0.034306 rad | 0.020000 rad |
| 被接触阻挡的行程 | 2.580752 mm | 2.000000 mm |
| 朝向接触效应 | 0.050860 rad | 0.050000 rad |

“效应”是先在每个后端计算 contact 减 sham，再跨后端比较，不是直接把空跑本来就有的偏差算成接触效应。56 ms 最大事件差来自 onset；例如左手 2 ms 的接触起始时刻为 MuJoCo 2.487 s、OVPhysX 2.431 s。同例释放时刻分别为 4.625 s 和 4.611 s。

剩余差异的单一底层原因未在本轮确定，不能直接归罪于某个默认参数、某个求解器或某家厂商。这轮已经能排除“把 5.5 ms 全部当作真实运动分离延迟”的解释，并证实通过稳定性准入后仍有跨后端响应差异。

## 限制与旧结论

- 这是固定基座、位置控制、零重力、无摩擦的合成球—盒接触；球附在单手 index_DP，原生手部碰撞禁用。左右手分别运行，不是双手任务，不覆盖真实手指碰撞网格、抓取、浮动基座或硬件。
- 日志中保留了旧正式 C0 同样存在的 cooking `registry is false` 错误提示及插件／材质警告。固定日志门槛检查的是 traceback、CUDA error 和 OOM 等明确类别；“该门槛通过”不代表日志零错误，更不证明其他日志无影响。本轮没有删除日志、放宽模式或证明这些提示完全无害。
- 原正式 C0 的 `VALID / INCONCLUSIVE` 保持不变；新候选是不同的实验，不追溯改判旧报告。原无接触 Gate 0/T1 的 `DIVERGENT` 和 `pass_ready=false` 也不变。
- 不把该结果写成官方修复、完整 Isaac Sim 对比、真实硬件验证或 Sim2Real 成功。

## 证据与复现

- [机器生成的完整报告](../results/contact-async/report.md)、[脱敏 summary](../results/contact-async/summary.json)。这两个文件是严格 finalizer 产生的原样公共投影。
- 源码 tree：`f8abb9618a1cf347bc882a450da3753f7527c960`。
- 源码 archive SHA-256：`1d1e632b1d2935cba041542aab29ec33cd2c04dca896dbad448643c4b66eb383`。
- 远端 snapshot SHA-256：`9092b6980284322f64bed783a6576310c8a4ad1e60eca487f5633c895272d2f3`。
- 新私有 bundle root：`90572b41c1b1427cde6a939f4ac7a87bf4f7a5a8c0b35ce7bac5cfa0e8799c4c`。
- local-a3 清单 root：`0f1d7f0d68087a8c3a93082a4a666960f3206a39a1ede98e425a0ec4497ea9b9`；114 files，236,051,249 bytes。
- remote-a3 清单 root：`44474980a8e61d83dc9ca6b9eb96384006f6f8be97cb0a7b2fd03be712413637`；601 files，252,336,479 bytes。
- 补充诊断由 `scripts/analyze_contact_async_result.py` 在严格验收后重算；私有结果 SHA-256：`a51430e500865270334ab104df20f41d79ec12bd77c86111355d67ae67bc2952`。其逐案例输入文件 hash 可连接到 sealed bundle；它不替代 gate。
- 原正式 C0 bundle root 已重新逐文件验证不变：`7f69ff9416c755401a7b2915684c457930feb13cb0803effc2955c90468dd69d`。
- 最终代码全套回归：1010 passed。封存包 721 条清单记录复算一致，公共两文件与包内投影逐字节一致，脱敏检查通过；调度队列中已无本轮任务。

复验必须使用干净的上述源码提交、固定依赖及资产；本地使用 `run_contact_c0_local.py --experimental-campaign contact-c0-async-v1`，远端同名候选选项配合受控 Slurm 单卡环境，最后运行 `finalize_contact_async.py`。Python 入口提供 `--help`，远端 shell 入口的 `usage` 列出必填参数；源码 revision/archive/snapshot 参数必须对应同一次冻结，不使用报告更新后的 HEAD 冒充实验源码。全部输出使用新目录，旧证据禁止覆盖。

## 排除的准备尝试

local-a1 的 worker 算完首例，但进程归属检查不识别候选字段，整批拒收；local-a2 完成 16 例，但配对的 remote-a2 在仿真前因数字卡号合约拒绝 UUID 字符串。修订后两端全部重采为 a3。三个旧目录完整保留并验证清单，不拼接到 active 32 案例中。

修订内容仅为候选标识贯通和 UUID 校验后的逻辑卡号传递；两次失败原因与修订记录见 [执行记录](CONTACT_ASYNC_CANDIDATE.md)。
