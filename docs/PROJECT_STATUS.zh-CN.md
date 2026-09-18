# A10 v1 交付清单

2026-09-18。工程交付完成，范围是 **SGLang 量化推理 profiling、保守的算子二次开发和可复现实验**，
不是“重造推理引擎”，也不是全模型/全硬件最优或秋招录用保证。

| 项目 | 完成情况 |
|---|---|
| 固定环境与模型 | SGLang v0.5.19、容器 digest、模型 revision/校验信息 |
| 量化部署与质量 | FP16 / 校准 W8A8 / AWQ 对照；扩展质量测试后明确记录 AWQ 失败 |
| 运行时开发 | 7 个可重放补丁；选中 SM86 形状守卫 INT8 GEMM，其他实验默认关闭 |
| GPU 检查 | 44 个数值 case、8 个原生分派边界、11 个 M/JIT 检查、64 个 Norm 集成检查 |
| 主性能结果 | 三轮短 prefill：128/1、并发 1、相同 no-overlap 配置，吞吐 +13.15% |
| 回归边界 | 32 输出、1K decode、8K prefill、共享前缀、多轮及高并发分别报告 |
| 次要正结果 | graph128 高并发三轮 +7.40%，低于原门槛，不替换主验收标准 |
| 执行与机制证据 | 独立 Torch CPU/GPU trace；PTX INT8 MMA；一秒 NVML 采样 |
| 正确的计量 | 固定原生 token、流式 completion usage、公开多轮输入指标限制 |
| 可复核交付 | 43 个 CPU 测试；CI 校验补丁树、公开哈希，并重算主结果 |
| 面试材料 | 简历段落、系统/流程图、源码地图、问答和四个动手练习 |

最终五组三轮、未采集 profiler 的服务实验共 11,136 请求全部成功。动态并发部分存在输出
差异，完整记录在报告，不宣称全场景文本等价。40 条质量任务输出一致也不等于 40/40 正确。

## 从哪里开始看

1. [最终实验报告](FINAL_REPORT.zh-CN.md)：数据、对照、归因和范围。
2. [简历介绍](RESUME.zh-CN.md)：可直接改写，但保留硬件和负载限定。
3. [系统讲解与面试问答](PROJECT_GUIDE.zh-CN.md)：从请求到 kernel 的路径及追问。
4. [源码地图](SGLANG_CODE_MAP.md) 与 [运行手册](RUNBOOK.md)：定位代码并亲自复现。
5. [证据核验](EVIDENCE_GUIDE.md)：无需 GPU 重算主结果；有 GPU 再完整重复实验。

公开仓库：[Biaogezi/sglang-agentperf](https://github.com/Biaogezi/sglang-agentperf)。
本次 Windows 工作目录为 `F:\code\Codex\sglang-agentperf`：源码、补丁和文档在仓库中，
完整原始 JSONL/日志位于 `results/`，质量原始结果在 `quality/`，原始 trace 在 `profiles/`。
这些大文件不提交普通 Git；公开 `evidence/` 保存摘要、逐请求指标及原始文件哈希。
SSH 私钥和 `.env` 不进入仓库。

## 不计入已完成成果

- 没有自行重跑 checkpoint 的 SmoothQuant/GPTQ/AWQ 校准流程；配方归模型作者。
- 没有官方上游 PR 提交或合并；公开的是自己的实验仓库及源码补丁。
- 没有 A100 双卡、TP=2、NPU 移植或跨模型收益证明。
- 没有 Nsight Compute roofline/带宽/占用率计数器测量，也没有生产规模 SLO 保证。
- 没有穷尽所有 attention/backend/config 组合；对照是明确版本与配置下的原生实现。

后续扩展应作为新实验版本，不修改已经冻结的证据。交付代码不等于已经掌握：投递前请完成
问答文档末尾的四个练习，能独立解释、修改边界测试和复算数据，再在面试中陈述自己的工作。
