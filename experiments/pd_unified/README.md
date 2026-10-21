# DynoSim 静态部署基线运行说明

这套实验用于建立固定模型、固定实例角色和副本数下的性能基线。先运行约 200 条请求的单实例用例，再比较等 GPU 预算的聚合与 PD 部署。

入口是 [run_baseline.py](run_baseline.py)。它复用已安装的 AISimulate／Dynamo SDK，负责 trace 筛选、配置编译、观测采集和结果整理。仿真仍在 WSL 的 CPU 上执行，配置中的 H200 是被模拟设备，无需本机具有 H200。

**DynoSim 的改造设计、实验代码、配置、测试、操作说明和结论分析统一由 AISimulate 仓库管理。**本项目材料集中在 `experiments/pd_unified/`；仿真器实现仍放在仓库对应的源码模块中。Dynamo 保持独立仓库，提供路由与 Replay SDK。

## 文档与实验材料入口

2026-10-21 起，材料按以下边界维护：

```text
experiments/pd_unified/
├── README.md                  实验入口、配置和运行方法
├── run_baseline.py            基线运行入口
├── analyze_results.py         结果分析脚本
├── configs/                   可复现的实验配置
├── tests/                     实验脚本的验证
├── docs/                      安装、源码能力边界、改造方案及接口设计
├── reports/                   实验结论、比较分析和后续问题
│   └── assets/                报告引用的图表与精简数据摘要
├── results/                   可选的本地原始输出，Git 忽略
└── .local-results/            本地预检与临时产物，Git 忽略
```

| 材料 | 入口 |
|---|---|
| 安装、构建工具、镜像源、官方用例和源码阅读顺序 | [WSL 安装与基线指南](docs/DynoSim_WSL安装与官方基线运行指南.md) |
| 当前源码能力、建模边界和 PD 一体化改造缺口 | [能力边界与 Gap 分析](docs/DynoSim_官方基线能力边界与池化状态架构Gap分析.md) |
| 937 条 Mooncake 请求的四 GPU 静态部署对照 | [首轮结果分析](reports/DynoSim_静态部署首轮结果分析.md) |
| 系统总体架构、目标场景和头脑风暴 | [项目资料入口](../../../../项目资料/README.md) |

`docs/`、`reports/` 及其图表、精简摘要纳入本仓库 Git；报告记录源码版本、配置、设备性能画像、trace 切片、指标口径及原始结果位置。完整日志和逐请求输出继续放在 `/home/qingya/pd-sim/results`，或 Windows 的 `仿真工具/tmp_logs`；不把批量原始输出归入 `reports/`。大 trace／权重仍由 `AI_Data_Trace_Weights` 按类别共享管理。

外部 `项目资料` 仓库维护总体架构、技术选型、跨仿真器调研与头脑风暴。架构稿提炼实验中的关键结论，并链接到这里的完整报告，避免重复维护实验参数和分析细节。Windows 仍是编辑与 Git 管理的主副本，WSL 按现有方式同步执行。

## 0. 当前实验与长期目标

**2026-10-20 确认：当前 Qwen／H200 的单实例与四 GPU 静态基线保持不变。**现阶段使用受支持、具有适用性能数据的 GPU 作为仿真 device，先研究固定部署瓶颈，再评估 P/D 角色和部署模型切换。结果按实际 GPU、后端版本、并行方式和数据覆盖解释；不将 GPU 的毫秒级切换阈值直接标成 950 结论。

长期目标是 **256／384 张昇腾 950 系列设备，每节点 8 卡，每实例 8／16 卡**。对应模型范围、实例数量、当前 PP=1 与 V4.1-Flash 的限制，以及 950 待补的 forward、内存、通信、运行时和切换 profiling，集中维护在 [架构讨论稿第 2.5 节](C:/projects/PD一体化无状态推理/项目资料/PD一体化池化推理集群_架构讨论稿.md)。

后续以替换和校准设备成本数据为主，若推理运行时的调度、KV 布局或并行行为存在差异，再补必要适配。本次仅登记目标和路线，不新增 256／384 卡或 950 YAML，也不修改现有脚本、配置和测试。

## 1. 当前配置

准备版本：AISimulate `0.13.0 / 221b816`，Dynamo `1.6.0 / d94072d`。运行结果另记录实际安装版本、源码位置、可发现的 Git HEAD 和已跟踪文件修改状态。

| 项目 | 初始设置 |
|---|---|
| 模型 | `Qwen/Qwen3-32B-FP8`，使用包内模型元数据，无需下载模型权重 |
| 硬件 | H200 SXM |
| 后端性能数据查询版本 | vLLM `0.24.0`，当前 H200 受维护的 `current` slot；实际执行的是当前 AIS 的仿真 scheduler |
| 耗时模型 | `op_level`，`SILICON`，`fallback_policy: deny` |
| 并行度 | 每个实例 TP1／PP1／Attention-DP1 |
| 上下文上限 | **40,960**，与内置模型的原生配置一致 |
| KV | 当前 Qwen/vLLM 默认 FP8 KV、BF16 attention；HBM-only、64-token page、prefix caching 开启，自动估算容量 |
| 路由 | Round Robin |
| PD 传输 | 按目标 D 缺失 token 计费，131,072 B/token，有效带宽暂设 50 GB/s |
| 暂定报表门限 | TTFT 5,000 ms、逐请求平均 ITL 50 ms |
| Telemetry | 每 1,000 ms 仿真时间采样一次 |

`SILICON` 表示所选成本数据库路径，不保证每次查询都是直接实测命中，仍可能涉及插值或已有复用规则。PD 的 50 GB/s 是抽象有效带宽，不包含共享链路争用；报表门限也还不是正式业务 SLA。

版本按 `systems/query_versions.yaml` 的可查询 slot 选择，不能只看原始数据目录。当前固定 `0.24.0`，不使用会随更新变化的 `current` 别名；它按上游明确的复用清单从 0.25.0 补充缺失 GEMM 键。0.25.0 的数据文件存在，并不意味着它本身是可查询入口。正式基线不设置 `AIC_ALLOW_UNLISTED_VERSIONS=1` 放开未维护入口，具体修复见第 10 节。

当前 Dynamo 1.6 runner 不接受显式 `engine.kvcache_quant_mode`／`engine.fmha_quant_mode`，即使它们是 AIS schema 的合法字段。五份配置省略这些字段，按固定源码版本的模型／后端默认解析。当前默认是 FP8 KV、BF16 attention；这不是对任意 FP8 模型或未来版本的通用推断。

三份 PD 配置显式设定 `engine.kv_transfer.bytes_per_token: 131072`，计算为 `2 × 64层 × 8个KV头 × 128维 × 1字节`。当前 `auto` 传输估算器仅根据 checkpoint 的 `torch_dtype=bfloat16` 推导字节数，会得到 262,144 B/token，与模型侧的默认 FP8 KV 不一致。更换模型、KV 精度或 TP/PP 后必须重新核对这个值。完整修复说明见第 9 节。

| `--case` | 部署 | GPU 数 | 用途 |
|---|---|---:|---|
| `agg1` | 1 个聚合实例 | 1 | 首次运行及链路检查 |
| `agg4` | 4 个聚合实例 | 4 | 静态对照 |
| `p1d3` | 1 P＋3 D | 4 | 静态对照 |
| `p2d2` | 2 P＋2 D | 4 | 静态对照 |
| `p3d1` | 3 P＋1 D | 4 | 静态对照 |

五份完整 YAML 位于 [configs](configs/)。其中的 trace 路径占位符由入口脚本替换，最终原生配置保存为每次运行的 `resolved.yaml`。

P 的 `max_sequences=1` 沿用社区角色默认值，聚合／D 为 256，batch token budget 均为 8192。上述配比和调度参数只是第一版起点，不能称为已经优化的“最佳静态部署”。`agg1` 不参加四 GPU 配比排名。

## 2. 同步到 WSL

Windows 主目录：

```text
实验代码：C:\projects\PD一体化无状态推理\仿真工具\aisimulate\experiments\pd_unified
原始数据：C:\projects\AI_Data_Trace_Weights\Traces\Mooncake_FAST25
```

以下命令由你在 WSL 的 root 终端运行：

```bash
mkdir -p /home/qingya/aisimulate/experiments/pd_unified
mkdir -p /home/qingya/pd-sim/results
mkdir -p /home/qingya/AI_Data_Trace_Weights/Traces/Mooncake_FAST25

rsync -rt \
  --exclude '__pycache__/' --exclude 'results/' --exclude '.local-results/' \
  '/mnt/c/projects/PD一体化无状态推理/仿真工具/aisimulate/experiments/pd_unified/' \
  /home/qingya/aisimulate/experiments/pd_unified/

rsync -rt --exclude 'derived/' \
  '/mnt/c/projects/AI_Data_Trace_Weights/Traces/Mooncake_FAST25/' \
  /home/qingya/AI_Data_Trace_Weights/Traces/Mooncake_FAST25/
```

保留 Mooncake 的 `upstream_manifest.json`，入口会核对源文件 SHA-256。首次只需同步这份约 8.6 MB 的数据，不需要同步 FreeInference。

如果已经使用仓库根目录的 Deployment，同步 AISimulate 仓库即可覆盖这个新目录。旧的 `/home/qingya/pd-sim/experiments` 副本不再作为入口；已有文件可保留，本次没有操作 WSL 中的目录。

目录关系：

```text
/home/qingya/
├── aisimulate/                         已 editable 安装的 AIS 源码及 Git 仓库
│   └── experiments/pd_unified/          本实验代码、配置和测试
├── dynamo/                             已 editable 安装的 Dynamo 源码
├── AI_Data_Trace_Weights/
│   └── Traces/Mooncake_FAST25/
│       ├── upstream/
│       └── derived/dynosim-static-v1/   脚本生成、可跨实验复用的切片
└── pd-sim/
    ├── .venvs/dynosim-src/              已安装的虚拟环境
    └── results/                        每次运行独立输出
```

## 3. PyCharm 的环境变量和入口参数

### 可直接粘贴的环境变量

在 Run/Debug Configuration 的 **Environment variables** 字段填写：

```text
PYTHONUNBUFFERED=1;PD_TRACE_ROOT=/home/qingya/AI_Data_Trace_Weights/Traces;PD_OUTPUT_ROOT=/home/qingya/pd-sim/results
```

| 变量 | 作用 |
|---|---|
| `PYTHONUNBUFFERED=1` | 立即显示 Python 进度和日志 |
| `PD_TRACE_ROOT` | 共享 Traces 根目录，**不要多加一层 `Mooncake_FAST25`** |
| `PD_OUTPUT_ROOT` | 日志、配置、报告及 CSV 的输出根目录 |

解释器选择已经启用了 venv 的包环境，无需在 PyCharm 再写 `source activate` 或 `VIRTUAL_ENV`。本轮不需要 Cargo、Rust、CUDA、NCCL、NATS 或 etcd 环境变量，也不需要人为覆盖 `PYTHONPATH`。HF offline 开关不作为本实验的环境要求；当前 Qwen 元数据已经内置。

### 解释器

使用已有 Linux 虚拟环境：

```text
/home/qingya/pd-sim/.venvs/dynosim-src/bin/python
```

它必须配置为 **On WSL** 或 **On SSH** 解释器，不能作为普通 Windows Local Interpreter。

### 首次运行参数

```text
--case agg1 --max-requests 200 --speedup 1
```

这是正式性能模型驱动的短 trace 测试，不使用固定 1 ms。约 200 条只用于先验证结果与观测链路，不能用作完整业务结论。

如果想先检查解释器、SDK 导入、配置编译、宿主资源准入及性能模型初始化，可先填：

```text
--case agg1 --max-requests 200 --speedup 1 --check-only
```

`--check-only` 现在还执行与 Dynamo 正式运行相同的启动参数构造：实际加载性能数据、检查模型 readiness、推导 KV 容量并做 native launch 校验。它不执行 Replay，也不证明实际 trace 的全部 batch／长度查询都受支持。

一次检查四 GPU 对照的全部角色，可以使用：

```text
--suite static --max-requests 200 --speedup 1 --check-only
```

预检通过后去掉 `--check-only` 再执行仿真。正常运行继续沿用 runner 原来的初始化路径，避免为每次正式 Replay 重复做一遍预检。

### 路线 A：手动同步后，在 PyCharm 打开 WSL 副本

这条路线不用先配置 SSH：

1. 完成上一节的 rsync。
2. 在 Windows 用 `wsl -l -q` 确认发行版名称。
3. PyCharm 打开 `\\wsl.localhost\<实际发行版名>\home\qingya\aisimulate`。
4. 添加 **On WSL → Existing environment**，选择上面的 venv Python。
5. 新建 **Python** Run Configuration，选择这个 WSL 项目中的 `experiments/pd_unified/run_baseline.py`。

实际 Linux 执行路径为：

| 字段 | Linux 侧值 |
|---|---|
| Script path | `/home/qingya/aisimulate/experiments/pd_unified/run_baseline.py` |
| Working directory | `/home/qingya/aisimulate/experiments/pd_unified` |
| Parameters | `--case agg1 --max-requests 200 --speedup 1` |
| Environment variables | 上述分号字符串 |

UI 若显示 UNC 路径，直接用文件选择器选 WSL 项目里的同一文件，让解释器完成路径映射。On WSL 使用发行版的执行用户，不保证等于终端中手动切换后的 root；日志会打印 UID，当前环境预期为 0。

Windows 原目录继续作为编辑主副本，保存后同步，再在 WSL 项目窗口运行。

### 路线 B：Windows 项目＋SSH interpreter＋Deployment

这条路线更符合“Windows 编辑、保存后同步到 Linux 执行”的习惯。前提是 WSL 已有可用 SSH 连接及 root 登录方式；本次准备不安装或启动 SSH 服务。

1. 使用当前 Windows AISimulate 项目，或打开 `C:\projects\PD一体化无状态推理\仿真工具\aisimulate`。
2. 添加 **On SSH** 解释器，使用实际 WSL 地址／端口，用户名 `root`。
3. 选择已有解释器 `/home/qingya/pd-sim/.venvs/dynosim-src/bin/python`。
4. 沿用仓库根级 Deployment。当前项目已经有以下映射，无需为实验目录再加一套：

| 本地仓库 | 远端仓库 |
|---|---|
| `C:\projects\PD一体化无状态推理\仿真工具\aisimulate` | `/home/qingya/aisimulate` |
| `C:\projects\PD一体化无状态推理\仿真工具\dynamo` | `/home/qingya/dynamo` |

解释器的 path mappings 也要对应相同的本地／远端仓库。新目录最终映射到 `/home/qingya/aisimulate/experiments/pd_unified`。先手动 Upload 验证，再启用保存时自动上传。

此路线在 Run Configuration 中选 **Windows 本地脚本路径和工作目录**，由映射执行 WSL 副本：

```text
Script path: C:\projects\PD一体化无状态推理\仿真工具\aisimulate\experiments\pd_unified\run_baseline.py
Working directory: C:\projects\PD一体化无状态推理\仿真工具\aisimulate\experiments\pd_unified
```

环境变量中的数据与输出路径仍填写 Linux 路径。

普通 WSL interpreter 对 Windows 项目默认使用 `/mnt/c/...` 映射，不会因为你另有 `/home/qingya` 副本就自动切过去。启动日志里的 Python、工作目录和源码位置可以检查究竟运行了哪一份。

当前项目配置已有 Deployment，但选中的解释器仍是 WSL 类型。若希望在 Windows 项目点 Run 后执行 `/home/qingya` 副本，应选上述 SSH interpreter；Deployment 本身不会改变 WSL interpreter 的运行映射。本次未改动 `.idea` 设置。

入口会用同一个 `sys.executable` 启动子进程，以完整保存 Rust/Python 日志。普通 Run 可直接使用；若以后需要在 SDK 子进程打断点，再启用 PyCharm 的子进程调试。

## 4. 终端运行方式

也可以直接在 WSL 用同一解释器运行，不要求先 activate：

```bash
export PYTHONUNBUFFERED=1
export PD_TRACE_ROOT=/home/qingya/AI_Data_Trace_Weights/Traces
export PD_OUTPUT_ROOT=/home/qingya/pd-sim/results

cd /home/qingya/aisimulate/experiments/pd_unified

/home/qingya/pd-sim/.venvs/dynosim-src/bin/python run_baseline.py \
  --case agg1 --max-requests 200 --speedup 1
```

后续可以更换 PyCharm Parameters，或将以下参数接在相同脚本之后：

| 目的 | 参数 |
|---|---|
| 短 trace 的 2P2D 检查 | `--case p2d2 --max-requests 200 --speedup 1` |
| 四 GPU 配比小规模对照，串行运行 | `--suite static --max-requests 1000 --speedup 1` |
| 全部长度合格请求的四 GPU 对照 | `--suite static --max-requests 0 --speedup 1` |
| 原到达间隔压缩为一半 | `--suite static --max-requests 0 --speedup 2` |
| 换 Tool/Agent 负载 | `--suite static --trace toolagent --max-requests 1000 --speedup 1` |

`speedup=2` 只提高到达强度，不使 forward 计算变快。每次运行内部保持模型、角色、数量不变；不接 Planner。

## 5. Trace 筛选的具体含义

`--max-requests N` 先取原始文件开头 N 行所在的完整时间戳批次，再做上下文筛选。保留同一时间戳的全部请求，避免人为拆散突发；筛掉超长请求后不补足数量，也不压缩剩余请求之间的时间间隔。

本次 Conversation 的实际准备结果：

| 参数 | 保留边界同刻请求后的条数 | 超过原生上下文而排除 | 最终请求数 |
|---|---:|---:|---:|
| `--max-requests 200` | 203 | 13 | **190** |
| `--max-requests 1000` | 1,001 | 64 | **937** |
| `--max-requests 0` | 12,031 | 574 | **11,457** |

默认 `--context-policy filter` 保留 `input < 40960` 且 `input + output <= 40960` 的请求。每次结果包含完整筛选记录；原始数据不被修改。若希望遇到任何超长请求都停止准备，使用 `--context-policy error`。

因此初版评估对象是 **Conversation 的 40K 上下文合格子集**，不是原始全量 workload。长上下文需求会在选定原生支持更长上下文、且有适用性能数据的模型后另做对照；不要仅修改 `context_length` 就认为已完成模型长上下文适配。

切片从冷缓存开始，当前无 warmup。使用相同参数的不同部署引用同一份衍生文件，并记录 SHA-256。脚本不设虚拟时间截止，让被选中的请求完整排空，避免只汇报成功子集。

## 6. 看哪些输出

输出形如：

```text
/home/qingya/pd-sim/results/<UTC时间>_static/
├── comparison.csv
├── agg4/
├── p1d3/
├── p2d2/
└── p3d1/
```

每个 case 目录包含：

| 文件 | 含义 |
|---|---|
| `run.log` | 子进程 Python／Rust stdout 和 stderr，失败先看这里 |
| `run-metadata.json`、`runtime.json` | 参数、请求数量、实际解释器、包／源码版本、运行状态 |
| `trace-selection.json` | 源文件／衍生文件哈希、筛选数量、时间范围 |
| `resolved.yaml` | 本次实际使用的原生配置 |
| `replay-spec.json`、`performance-model-config.json` | 编译后的配置与已有性能模型元数据 |
| `runner-capabilities.json` | 当前 runner 声明的能力；与编译后的 spec 一起在兼容性检查前保存 |
| `resource-plan.json` | 宿主资源准入估计，不是设备利用率 |
| `engine-preflight.json` | `--check-only` 各角色的性能数据／启动参数检查状态；失败时保留错误 |
| `engine-launch-configs.json` | `--check-only` 成功构造角色的实际启动参数及 KV block 数，不重新注入为显式容量 |
| `prediction.json` | 原生报告，包含 summary、请求和 telemetry |
| `requests.jsonl`、`requests.csv` | 每请求结果、输出完整性、阶段时间 |
| `telemetry.jsonl` | 原始周期观测 |
| `scheduler_samples.csv` | 按实例／角色／rank 展开的运行、等待请求及 KV 占用 |
| `traffic_samples.csv` | 区间到达／完成计数、抢占和路由队列 |
| `analysis.json` | 完整性结论、分位数与按暂定 SLA 计算的有效请求吞吐 |

数据准备模式 `--prepare-only` 和 SDK 检查模式 `--check-only` 不生成仿真结果文件。

`engine-preflight.json` 的 `kv_token_slots_per_rank` 是每个 Attention-DP rank 的 KV token 槽位数，不是 HBM 总字节数。预检输出只记录推导结果，不改变正式运行的自动容量语义。

解释指标时：

- `p99_request_mean_itl_ms` 是逐请求平均 ITL 的 P99，不是所有 token 间隔的 P99。
- `first_admit_wait_ms` 包含到首次准入的等待，可能同时包括路由与调度器排队。
- `post_prefill_wait_ms` 是 P 完成后到 D 准入的组合等待，不是纯传输时延。
- 聚合实例在原生 telemetry 的 decode 字段中，CSV 已改标为 `aggregated`。
- KV 占用不是 GPU 算力利用率；当前报告不生成 MFU 或 SM active 数字。

若有拒绝、未终态请求、实际输出短于 trace 要求等，分析器将 `valid` 标为 false，正式 goodput 置空，并保留诊断值和原始记录。脚本返回非零退出码，不把这类结果自动当作有效配比对照。

当前脚本复用 SDK 的资源准入估计，串行执行各 case；它没有复制 CLI 的持续内存监督器。先从短 trace 检查，再扩大请求规模，资源准入失败时按 `resource-plan.json` 处理。

## 7. 已完成的准备验证

- 5 份 YAML 及其渲染配置通过当前源码的配置 schema 检查。
- Windows `--prepare-only` 已生成单实例短切片和四 GPU 全量合格子集，四组对照使用相同 trace 哈希。
- 本地测试覆盖同时到达边界、上下文筛选、文件篡改检测、请求完整性、单输出 token 的 SLA、telemetry 口径和 FP8 KV 传输几何。
- 2026-10-20 补充了 runner 能力检查：本地以真实 capability 方法验证 model-control 子契约，复现原配置失败、修正配置通过；完整 SDK 合约测试需要在已安装扩展的 WSL 环境执行，Windows 会明确跳过。
- WSL 中的真实 SDK Replay 留给你执行；schema 和分析器测试不能替代性能数据加载、运行结果或实机准确性验证。

测试不需要真实 GPU，也不执行 Replay。基础测试仅依赖 Python／PyYAML；`test_sdk_contract.py` 另外需要已安装的 AIS／Dynamo Linux 扩展：

```bash
python -m unittest discover -s tests -v
```

这些脚本、YAML 和目录调整不需要重编译。以后修改 AIS Rust core 时，需重新构建相应扩展，并确认 Dynamo 构建引用本地 AIS 源码；Python editable 安装不会自动更新已经编译进 `.so` 的 Rust 实现。

## 8. 参考

- [AIS timing 配置契约](../../python/aisimulate/.claude/rules/perfmodel-api.md)
- [Dynamo Replay SDK](../../../dynamo/components/src/dynamo/replay/api.py)
- [Dynamo Runner](../../../dynamo/components/src/dynamo/replay/simulation.py)
- [JetBrains WSL 解释器与 WSL 目录](https://www.jetbrains.com/help/pycharm/using-wsl-as-a-remote-interpreter.html)
- [JetBrains SSH 解释器](https://www.jetbrains.com/help/pycharm/configuring-remote-interpreters-via-ssh.html)
- [JetBrains Deployment 映射](https://www.jetbrains.com/help/pycharm/creating-a-remote-server-configuration.html)
- [JetBrains Python Run Configuration](https://www.jetbrains.com/help/pycharm/run-debug-configuration-python.html)

## 9. Bug 分析报告

### 一、错误现象

1. 2026-10-20 首次 WSL 运行已使用 UID 0、AIS 0.13.0 和 Dynamo/runtime 1.6.0，完成 trace 准备与 SDK 导入后，在 `factory.capabilities().require_compatible(spec)` 失败：

   ```text
   runner does not support engine model controls ['kvcache_quant_mode', 'fmha_quant_mode']; use --stack engine
   ```

2. 复现：运行初版 `agg1` 配置，其中显式设置两个 `bfloat16` 字段。五份初版配置均包含该设置。
3. 错误日志和对应版本信息完整，定位不涉及用户目录权限或包缺失。
4. 属于本实验配置引入的问题，没有对应上游 PR，也不需要 Git 二分。

### 二、根因分析

采用从错误栈向上追踪的方法：

1. AIS schema 接受这些字段，compiler 将它们保留在 timing identity。
2. Dynamo factory 未声明 `supported_engine_model_controls`，其默认值为空。
3. AIS 的 `require_compatible` 检查发现两个显式控制不受支持并拒绝。之前的 YAML schema 校验及结果分析测试没有覆盖这一层。
4. 省略两个字段后，还需统一数据口径：Qwen3-32B-FP8 在当前 vLLM 模型默认路径下采用 FP8 KV／BF16 attention，但 PD 自动传输估算根据 `torch_dtype` 使用 BF16 字节数。

证据：[Dynamo 能力声明](../../../dynamo/components/src/dynamo/replay/simulation.py:62)、[AIS 能力校验](../../python/aisimulate/src/aisimulate/sweeper/replay.py:437)、[默认精度解析](../../python/aisimulate/src/aisimulate_core/sdk/models/helpers.py:713)、[vLLM attention 默认](../../python/aisimulate/src/aisimulate_core/sdk/models/helpers.py:817)、[PD 字节数编译](../../python/aisimulate/src/aisimulate/compiler.py:267)、[朴素 dtype 字节数估算](../../python/aisimulate/src/aisimulate_core/sdk/memory.py:682)。

### 三、最小化 Diff 格式修复补丁

以下删除应用于全部五份 YAML，传输修正应用于三份 PD YAML：

```diff
 engine:
-  fmha_quant_mode: bfloat16
-  kvcache_quant_mode: bfloat16
+  # Dynamo 1.6 rejects explicit precision controls; use model/backend defaults.
   kv_transfer:
-    bytes_per_token: auto
+    # Qwen3-32B automatic FP8 KV, TP1/PP1.
+    bytes_per_token: 131072
```

入口另外将 `replay-spec.json` 和新增的 `runner-capabilities.json` 提前保存，兼容性失败时也能复查实际输入。保留原能力校验和 Dynamo stack；没有修改上游 runtime 或宣称支持新的精度控制。

### 四、防回归测试用例

- `test_preparation.py`：检查五套配置使用默认精度，并独立按 checkpoint 几何核算三套 PD 的传输字节数。
- `test_sdk_contract.py`：在真实已安装环境中，经 schema、adapter、compiler 和 runner 检查五套配置；恢复两个显式 BF16 字段时应复现原错误，并检查 PD 编译参数。测试不调用 Replay。
- Windows 没有完整 native 扩展，因此本地子契约验证不等于完整 SDK／性能模型运行通过。

在 WSL 的本实验目录执行：

```bash
/home/qingya/pd-sim/.venvs/dynosim-src/bin/python -m unittest discover -s tests -v
```

### 五、预防建议

配置验收分为 schema、runner 能力、实际启动参数构造和整段 Replay 四个层次；在已安装环境中先运行 `--suite static --check-only`。KV 精度、容量和传输字节数需要共同核对。升级模型或源代码版本后，重新检查默认推导与性能数据覆盖。

## 10. Bug 分析报告：性能数据库查询版本

### 一、错误现象

1. 2026-10-20，移除不受支持的精度控制后，运行继续进入 `materialize_engine_launch_config`，在实际构造性能模型时失败：

   ```text
   vllm/'0.25.0' looks like an old-style raw version query
   h200_sxm ... slot versions {'current': '0.24.0', 'next': '0.27.1'}
   ```

2. 初版五份 YAML 均选择了 `backend_version: "0.25.0"`。
3. 用户日志提供了完整版本和调用链，权限、SDK 导入及 runner 能力检查均已通过。
4. 这是实验配置选错数据查询入口，不表示 Qwen 模型本身不受支持，也不需要重装 vLLM 或 Rust 扩展。

### 二、根因分析

采用自顶向下追踪：

`Dynamo _engine_args → materialize_engine_launch_config → materialize_aic_num_gpu_blocks → RustForwardPassPerfModel.best_available → resolve_query_version`。

`query_versions.yaml` 定义每个硬件／后端的受维护查询版本。当前 H200/vLLM 的 `current` 为 0.24.0；`next` 从数据目录推导为 0.27.1。0.25.0 是原始采集数据坐标，可作其他入口明确批准的补充来源，但不在允许直接查询的 slot 中。

此前根据 0.25.0 目录及采集元数据选择版本，没有检查查询资格，是本实验准备遗漏。此前 `--check-only` 又只到 schema、capabilities 和宿主准入，未触发真正的性能数据加载，因此抓不住这类错误。

证据：[查询版本配置](../../python/aisimulate/src/aisimulate_core/systems/query_versions.yaml)、[版本准入实现](../../python/aisimulate/src/aisimulate_core/sdk/perf_database.py:309)、[0.24.0 GEMM 复用清单](../../python/aisimulate/src/aisimulate_core/systems/data/h200_sxm/gemm/vllm/0.24.0/reuse.yaml)、[Dynamo 实际初始化路径](../../../dynamo/components/src/dynamo/replay/simulation.py:327)。

### 三、最小化 Diff 格式修复补丁

五份配置统一改为固定、受维护的具体版本：

```diff
-  backend_version: "0.25.0"
+  # Pin the maintained H200/vLLM current slot, not a raw collection directory.
+  backend_version: "0.24.0"
```

不修改模型、设备、请求切片、角色配比、FP8 KV 或 PD 的 131,072 B/token；不通过 `AIC_ALLOW_UNLISTED_VERSIONS` 绕过版本准入。

`--check-only` 增加真实的 `materialize_engine_launch_config → normalize_mocker_config` 路径，保持 canonical 性能模型构造方式。按角色保存初始化结果和失败信息；正式 Replay 的初始化及自动容量行为保持原样。

0.24.0 会优先使用其自身数据，并按显式规则从 0.25.0 补充缺失 GEMM 键。这种复用有来源记录，不等于所有数据都重新采集于 0.24.0，也不代表全部运行环境时延完全一致。

### 四、防回归测试用例

- `test_preparation.py` 读取当前仓库的 query slots，按硬件 override 优先，检查五份 YAML 固定到受维护的 `current` 具体版本；原 0.25.0 配置下五组均失败，修正后应通过。
- `test_sdk_contract.py` 在完整 Linux SDK 中，对五套配置逐角色构造真实启动参数，核对正的 KV block 容量、版本和 PD 字节数。
- 同一 SDK 测试恢复 0.25.0，验证 capabilities 虽通过，实际 materialization 仍拒绝该版本；测试期间关闭未维护版本放行变量。
- 本地 Windows 的基础测试可以执行；缺少原生扩展时 SDK 测试明确跳过，不能记作真实模型初始化成功。实际 WSL 预检和 Replay 结果由用户运行确认。

### 五、预防建议

选性能版本时先检查 query slots，再检查该入口的主数据、复用来源及具体查询覆盖。保留具体版本和两仓提交，避免浮动别名改变实验条件。启动预检成功后，仍需完整回放并检查请求数、实际输出长度及数据查询错误，才能确认该 workload 可运行。
