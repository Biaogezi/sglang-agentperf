# SGLang-AgentPerf：A10 量化推理优化报告

日期：2026-09-18。状态：最终源码的 GPU 检查、质量与短 prefill 性能通过，长输出和多轮回归仍在执行。

## 1. 项目到底交付什么

在真实 SGLang v0.5.19 上做性能工程，不重写简化推理引擎。主线是：固定实验环境 → 分阶段
profiling → 选定热点 → 修改真实运行时 → 数值和实际执行检查 → 同条件 A/B → 保存原始证据。

当前选中的源码优化是 **SM86 形状感知 W8A8 INT8 GEMM**。保留既有权重量化、激活量化和
KV cache；只替换有正收益的线性层形状。自己的贡献不包括 SGLang 的分页 KV、RadixAttention、
continuous batching、FlashInfer、Marlin、CUTLASS 或第三方模型的量化校准。

## 2. 可复现边界

| 项目 | 固定值或说明 |
|---|---|
| 硬件 | 单 NVIDIA A10 24 GiB，8 vCPU / 30 GiB 主机；TP=1 |
| 上游 | v0.5.19，`0bcd822377da7b5718e674eaf9c870d349424dd1` |
| 最终补丁源码树 | `bb70ae7f8fb7cede92b8655fbc503aeb5c42bcbd`，7 个可重放补丁 |
| 运行环境 | `CONTAINER.lock` 固定镜像 digest；宿主机驱动独立记录 |
| 模型 | Qwen3-8B；FP16、已校准 W8A8、AWQ 各自固定 revision |
| 自定义开关 | `SGLANG_A10_INT8_PREFILL=true`；默认 OFF |
| 对照 | 相同最终补丁源码树、相同量化权重、GEMM 开关 OFF |
| 未测范围 | A100、TP=2、NPU、新模型和其他运行环境，不计入项目成果 |

补丁提交 ID 因本地/远端提交时间而不同，源码 tree 和执行文件 SHA-256 才是内容标识。
本地上游目录可能保留未采用的构建实验；复现请使用 `UPSTREAM.lock` 的补丁序列，不能直接打包
那个工作目录。公开 CI 重建补丁树并校验公开证据，GPU 检查日志单独保存。

## 3. 为什么选这个热点

早期 W8A8 trace 中，CUTLASS INT8 GEMM 分别占混合 prefill 和单请求 decode 的约 64.7%、77.9%
kernel 时间。Norm 与激活量化的占比只有约 5.7%、3.5%，所以大幅加速一个小算子并不一定改变
服务性能。最终选择对 GEMM 做形状扫描，而不是先承诺端到端提速百分比。

支持范围为 INT8 输入/权重、FP32 row/channel scale、FP16/BF16 输出、SM86、M=80..128、
K=4096、N=6144 或 24576，并检查布局、设备和可选 bias。其他形状回退原 CUTLASS。
实际 M 是本次模型执行的 token 行数，可能受 batching、chunking 和 CUDA Graph padding 影响，
不必等于一条用户请求的原始长度。

kernel 使用 128×128×128 tile、4 warps、3 stages、INT32 MMA 累加，在 epilogue 复用原有
`acc * (row_scale * channel_scale)` 的顺序。最终 GPU 测试验证 PTX 内存在 INT8 `mma.sync`；
不是把量化权重还原成 FP16 再调用普通矩阵乘。邻近形状、实际模型权重和更宽的 tile 扫描均保留。

将 M 从编译常量改为运行时参数，并仅区分满块/边界块。固定 N/K/dtype/bias 下，11 个 M
测试复用 2 个编译二进制。不能将此表述为“整个模型只有两个 kernel”。

## 4. 最终版本的短 prefill 结果

运行 `20260918T044732Z__short`，三轮 AB/BA/AB 服务重启、每轮预热；每个输入长度 160 请求、
并发 1、输出 1 token；使用 native token IDs，不做 text roundtrip。两边同样关闭上游 overlap
scheduler，CUDA Graph 保持启用，Norm 融合关闭。这个已有调度开关不是本项目发明的优化。

| 输入 tokens | OFF input tok/s，均值 ± 样本 SD | ON input tok/s，均值 ± 样本 SD | 吞吐变化 | 各轮 p99 TTFT 均值，OFF → ON |
|---|---:|---:|---:|---:|
| 96 | 3302.26 ± 28.63 | 3502.28 ± 15.47 | +6.06% | 29.818 → 28.217 ms |
| 128 | 4143.97 ± 20.08 | 4688.91 ± 27.87 | **+13.15%** | **31.705 → 28.115 ms** |
| 160，回退 | 3904.22 ± 17.65 | 3920.42 ± 33.43 | +0.41% | 42.061 → 41.966 ms |

2,880 个请求全部成功，9 组配对输出逐条一致，所有启动的执行源码指纹一致。128-token 点通过
原定 10% 吞吐门槛，p99 TTFT 减少 11.32%。输出只有一个 token，TPOT 不适用。结果不是
所有聊天/agent 请求都加速 13.15%，也不是全局最优证明。三轮 p99 的均值不是合并样本后的 p99。

前一源码版本同配置独立三轮得到 +12.08%，保留为历史复测；不能挑选/混合两组里的最快轮次。
默认 overlap 下此前约 +7.9%，不把调度配置收益加到自定义算子收益里。

公开证据：`evidence/a10/final_short_off/`、`final_short_on/`。

## 5. 正确性与量化质量

最终树四组 GPU 检查全部通过：44 个矩阵正确性测试、8 个真实原生量化 method 的边界/回退测试、
11 个动态 M/JIT 变体检查、64 个原生 Norm 融合检查。后者验证备用候选，不意味着默认启用。
日志在 `evidence/a10/final_gpu_validation/`。

扩展质量集为固定 WikiText-2 test revision 的 64 个独立 128-token 窗口，共 8,128 个有效预测。
这不是标准拼接/滑窗 WikiText perplexity。FP16 NLL 为 2.94421470；AWQ 为 3.02027352，
delta +0.07605882，**未过**原定 +0.02 门槛。此前 335-token smoke 的通过结论已被更大测试修正。

固定任务回归有 strict JSON 16 题、严格输出格式算术 16 题、长 key 检索 8 题。FP16 和 W8A8
的早期测试在算术格式上为 0/16；不能因答案里包含正确数字就改评分规则。JSON/检索检查和
greedy 输出相等只证明这个小测试集上的表现，不代表广泛推理能力或真实工具调用正确率。

最终 GEMM-only 运行 `20260918T045756Z`：同权重 OFF NLL 为 2.9490910677，ON 为
2.9490919142，delta **+0.0000008466**，低于 +0.02；40 条 greedy 任务输出全部一致，
没有丢失原来正确的题。两边均为 JSON 16/16、检索 8/8、严格算术格式 0/16。ON 对 FP16
参考的 NLL delta 为 +0.00487721，指数化 NLL 相对增加约 0.489%，通过 checkpoint 质量门槛。
公开数据与原始合成任务回答在 `evidence/a10/final_quality_off/`、`final_quality_on/`。

## 6. 回归矩阵

最终代码还要分别报告：短输入 + 32-token 输出；并发 1K/512 decode；8K/64 prefill；共享前缀；
24 个合成会话 × 4 轮工具观察重放。未完成的结果不预填正收益。

多轮 fixture 是原创合成数据，实际模型回复会进入下一轮，不是真的执行 shell/搜索工具，也不是
SWE-bench。该版本上游客户端不正确汇总多轮输入 token 数/命中率，所以不报告这些字段。
只报告请求完成、输出吞吐、TTFT/TPOT/E2E，并检查输出历史是否改变。

## 7. 没有采用的候选与有条件的配置收益

| 候选/配置 | 观察 | 决策 |
|---|---|---|
| 自适应 prefill chunk | 没超过调好的静态 1024 对照 | 拒绝作为有效源码优化 |
| 静态 chunk 1024 | 历史 AWQ p99 TPOT -72.9%，输入吞吐 -7.8% | SLO/吞吐取舍，不是发明新 scheduler；AWQ 另有质量限制 |
| Marlin FP16 reduction | 热负载没有增益，长输出改变 | 拒绝 |
| 早期 Norm 服务实验 | 开关没有进入实际原生 W8A8 类 | 撤回因果结论，修复分派并重新验证 |
| 修复后的 GEMM + Norm | 默认 overlap 128-token 吞吐 +9.02%，部分输出改变 | 未过吞吐门槛；最终选 GEMM-only |
| 全部形状替换 GEMM | 支持区外多次更慢 | 用硬件/形状/布局守卫和回退 |
| 扩大 tile 搜索 | 微测收益不等于服务收益 | 固定最终 tile，报告全部对照，不挑最快服务轮次 |

这些负结果是筛选过程，不计入简历的已实现性能提升。

## 8. 如何复现与解释

- 从 [GPU runbook](RUNBOOK.md) 重建锁定环境和补丁；使用固定模型 revision。
- 查看 [优化 004](OPTIMIZATION_004_SM86_INT8_PREFILL.md) 的形状扫描/协议修正历史。
- 查看 [优化 006](OPTIMIZATION_006_LATENCY_CONTROL.md) 的 overlap 控制和端到端结果。
- 阅读 [项目讲解与问答](PROJECT_GUIDE.zh-CN.md)，能手算量化缩放、KV 容量并解释源码回退。
- `python scripts/verify_evidence.py` 检查公开文件；原始 JSONL、日志、trace 另行本地备份。

没有向上游提交或合并 PR，不写“已贡献 SGLang 官方优化”；没有实际 NPU 适配，不写跨芯片移植。
