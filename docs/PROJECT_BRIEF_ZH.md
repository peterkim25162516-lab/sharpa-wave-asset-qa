# 项目简述（中文）

`Sharpa Wave Asset QA` 是一个独立、非官方、仅面向仿真的机器人资产质量验证工具。它针对 Sharpa 官方公开的 Wave 灵巧手资产，把原本依赖人工检查的 URDF、MuJoCo MJCF 与 ASCII USDA 一致性审查变成可重复执行的测试和报告。

第一阶段不依赖 Isaac Sim，普通 CPU 电脑即可完成：

- 自动发现 left/right/dual/float-base 及 wrist/flange 变体；
- 检查关节、限位、坐标轴、树结构、网格路径和执行器绑定；
- 对同一变体做跨格式 joint schema 与 limit 对比；
- 固定随机种子采样合法关节角，比较 URDF 与 MJCF 的末端运动学；
- 用 MuJoCo 做无界面加载、有限状态、确定性和实时率 smoke test；
- 生成可机器读取的 JSON 和便于评审的 Markdown 报告。

公开时必须明确：这是特定 commit 和软件版本下的 validator observation，不是对真机缺陷、Sim2Real 能力或 Sharpa 官方合作关系的声明。项目不复制任何私有仓库的代码、配置、数据、截图或指标。

## 面向简历的最终证据（发布后再填写）

- 覆盖的模型入口数量与格式矩阵；
- 自动测试数量和 CI 环境；
- 跨格式 joint/limit 匹配结果；
- FK 的 median / p95 / max 误差；
- MuJoCo 成功加载比例、确定性误差与 CPU real-time factor；
- 被公开记录并可复现的 known finding；
- GitHub release、演示视频和技术报告链接。

在仓库公开、结果复跑完成以前，不把上述指标写成已发布成果。
