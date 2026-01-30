import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import random


def _add_repo_paths():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(this_dir, "..", ".."))
    geak_eval_root = os.path.join(repo_root, "GEAK-eval")

    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    if geak_eval_root not in sys.path:
        sys.path.insert(0, geak_eval_root)


_add_repo_paths()

from geak_eval.constants import NATIVE_PERF_GOLD_ROOT
from geak_eval.evaluators.interface import TestAllCloseEvaluatorTBG


REF_OP_PATH = r"/workspace/zibo/Geak-OPT/GEAK-eval/geak_eval/data/TritonBench/data/TritonBench_G_v1/dequantize_rowwise.py"
NUM_ROUNDS = 5
OFFSPRING_PER_ROUND = 10
OFFSPRING_REPEAT_PER_PLAN = 2

ATOL = 1e-3
RTOL = 1e-3
TIMEOUT_S = 5 * 60

OUTPUT_ROOT = r"/workspace/zibo/Geak-OPT/WIse-Agent/output"
REPORT_PREFIX = "optimize"
NCU_SET = "full"
NCU_FULL_REPORT = 0
MODEL_NAME = "qwen3-coder-plus"

API_MAX_RETRIES = 6
API_RETRY_BACKOFF_S = 2.0
API_RETRY_MAX_BACKOFF_S = 30.0


TESTS_SEP_LINE = "#" * 146


def _read_operator_impl(fpath: str) -> str:
    with open(fpath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if line.strip() == TESTS_SEP_LINE:
            return "".join(lines[:i]).rstrip() + "\n"

    return "".join(lines).rstrip() + "\n"


def _run(cmd, cwd=None, env=None):
    p = subprocess.run(cmd, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return p.returncode, p.stdout, p.stderr


def _extract_json_object(text: str) -> dict | None:
    if not text:
        return None

    m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.S | re.I)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass

    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                cand = text[start : i + 1]
                try:
                    obj = json.loads(cand)
                    if isinstance(obj, dict):
                        return obj
                except Exception:
                    return None

    return None


def _get_plan_priority(plan) -> int | None:
    if not isinstance(plan, dict):
        return None
    p = plan.get("priority")
    if p is None:
        return None
    try:
        return int(p)
    except Exception:
        return None


def _qwen_chat(messages: list[dict], temperature: float = 0.7) -> str:
    from openai import OpenAI

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY is not set")

    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    last_err = None
    for attempt in range(1, int(API_MAX_RETRIES) + 1):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                stream=True,
                temperature=temperature,
            )
            out = []
            for chunk in completion:
                delta = chunk.choices[0].delta.content
                if delta:
                    out.append(delta)
                    print(delta, end="", flush=True)
            print()
            return "".join(out)
        except Exception as e:
            last_err = e
            if attempt >= int(API_MAX_RETRIES):
                raise RuntimeError(f"LLM request failed after {API_MAX_RETRIES} attempts: {e}") from e

            backoff = min(float(API_RETRY_MAX_BACKOFF_S), float(API_RETRY_BACKOFF_S) * (2 ** (attempt - 1)))
            backoff = backoff * (0.7 + 0.6 * random.random())
            print(f"\nLLM request error (attempt {attempt}/{API_MAX_RETRIES}): {e}. Retry in {backoff:.1f}s", flush=True)
            time.sleep(backoff)

    raise RuntimeError(f"LLM request failed after {API_MAX_RETRIES} attempts: {last_err}")


def _truncate_first_operator_profile(text: str) -> str | None:
    if not text:
        return None

    lines = text.splitlines()
    sec_idx = None
    for i, ln in enumerate(lines):
        if ln.strip() == "Section: Source Counters":
            sec_idx = i
            break

    if sec_idx is None:
        return None

    def _is_sep_line(line: str) -> bool:
        s = line.strip()
        if not s:
            return False
        s2 = s.replace(" ", "")
        return s2 != "" and all(ch == "-" for ch in s2)

    sep_count = 0
    end_idx = None
    for j in range(sec_idx, len(lines)):
        if _is_sep_line(lines[j]):
            sep_count += 1
            if sep_count >= 3:
                end_idx = j
                break

    if end_idx is None:
        return None

    return "\n".join(lines[: end_idx + 1]).rstrip() + "\n"


def _strip_opt_lines(text: str) -> str:
    if not text:
        return ""
    out_lines = []
    for line in text.splitlines():
        # Drop any row/line that contains an OPT token (case-insensitive).
        if re.search(r"\bOPT\b", line, flags=re.I):
            continue
        out_lines.append(line.rstrip())
    return "\n".join(out_lines).strip() + "\n"


def _find_section_block(text: str, section_name: str) -> str | None:
    if not text:
        return None

    lines = text.splitlines()
    start_idx = None
    header = f"Section: {section_name}" if not section_name.startswith("Section:") else section_name

    for i, ln in enumerate(lines):
        if ln.strip() == header:
            start_idx = i
            break

    if start_idx is None:
        return None

    out = []
    for j in range(start_idx, len(lines)):
        ln = lines[j]
        if j > start_idx and ln.lstrip().startswith("Section:"):
            break
        out.append(ln)

    block = "\n".join(out).strip("\n")
    return block if block.strip() else None


def _parse_table_metric_value(block: str, metric_name: str) -> str | None:
    if not block:
        return None

    for line in block.splitlines():
        s = line.strip()
        if not s:
            continue
        if not s.startswith(metric_name):
            continue

        m = re.search(r"(-?[0-9]+(?:\.[0-9]+)?)\s*$", s)
        if not m:
            return None
        return m.group(1)

    return None


def _unknown_if_none(v: str | None) -> str:
    if v is None:
        return "unknown"
    s = str(v).strip()
    return s if s else "unknown"


def _parse_metric_percent(table: str, metric_name: str) -> float | None:
    if not table:
        return None
    for line in table.splitlines():
        if line.strip().startswith(metric_name):
            m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*$", line.strip())
            if m:
                try:
                    return float(m.group(1))
                except Exception:
                    return None
    return None


def _analyze_bottleneck(table: str) -> dict:
    mem = _parse_metric_percent(table, "Memory Throughput")
    dram = _parse_metric_percent(table, "DRAM Throughput")
    l1 = _parse_metric_percent(table, "L1/TEX Cache Throughput")
    l2 = _parse_metric_percent(table, "L2 Cache Throughput")
    sm = _parse_metric_percent(table, "Compute (SM) Throughput")

    reason = "unknown"
    if sm is not None and sm >= 50:
        reason = "compute_bound"
    elif dram is not None and dram >= 50:
        reason = "dram_bound"
    elif l1 is not None and l1 >= 50:
        reason = "l1_bound"
    elif mem is not None and mem >= 50:
        reason = "memory_bound"

    strategy = {
        "bottleneck": reason,
        "metrics": {"Memory Throughput": mem, "DRAM Throughput": dram, "L1/TEX Cache Throughput": l1, "L2 Cache Throughput": l2, "Compute (SM) Throughput": sm},
    }

    if reason in {"dram_bound", "memory_bound", "l1_bound"}:
        strategy["optimization_strategy"] = "Improve memory access efficiency: increase contiguous loads/stores, use tl.multiple_of/tl.max_contiguous, adjust BLOCK sizes and num_warps/num_stages to better utilize bandwidth, and reduce redundant loads."
    elif reason == "compute_bound":
        strategy["optimization_strategy"] = "Improve compute efficiency: consider using lower precision (fp16/bf16), fuse operations, reduce type conversions, and tune num_warps/num_stages."
    else:
        strategy["optimization_strategy"] = "Tune launch configuration first: adjust num_warps/num_stages, ensure contiguous memory accesses, and validate grid/block choices."

    return strategy


def _draft_step1_prompt(kernel_code: str, ncu_res: str) -> str:
    # Step1 uses the same prompt template as ncu_llm/parse_profile_to_prompt.py.
    # We first parse the NCU text into structured indicators, then embed them into the prompt.
    sol = _find_section_block(ncu_res, "GPU Speed Of Light Throughput") or ""
    launch = _find_section_block(ncu_res, "Launch Statistics") or ""
    sched = _find_section_block(ncu_res, "Scheduler Statistics") or ""
    occ = _find_section_block(ncu_res, "Occupancy") or ""
    mem_work = _find_section_block(ncu_res, "Memory Workload Analysis") or ""
    warp_state = _find_section_block(ncu_res, "Warp State Statistics") or ""
    inst_stats = _find_section_block(ncu_res, "Instruction Statistics") or ""
    src_cnt = _find_section_block(ncu_res, "Source Counters") or ""
    comp_work = _find_section_block(ncu_res, "Compute Workload Analysis") or ""

    grid = _parse_table_metric_value(launch, "Grid Size")
    block = _parse_table_metric_value(launch, "Block Size")
    sms = _parse_table_metric_value(launch, "# SMs")
    waves = _parse_table_metric_value(launch, "Waves Per SM")
    dur = _parse_table_metric_value(sol, "Duration")

    oep = _parse_table_metric_value(sched, "One or More Eligible")
    nep = _parse_table_metric_value(sched, "No Eligible")
    aws = _parse_table_metric_value(sched, "Active Warps Per Scheduler")
    ews = _parse_table_metric_value(sched, "Eligible Warps Per Scheduler")
    iws = _parse_table_metric_value(sched, "Issued Warp Per Scheduler")

    aop = _parse_table_metric_value(occ, "Achieved Occupancy")
    aaw = _parse_table_metric_value(occ, "Achieved Active Warps Per SM")
    bl_regs = _parse_table_metric_value(occ, "Block Limit Registers")
    bl_shmem = _parse_table_metric_value(occ, "Block Limit Shared Mem")
    bl_warps = _parse_table_metric_value(occ, "Block Limit Warps")
    if bl_regs is None and bl_shmem is None and bl_warps is None:
        block_limits = None
    else:
        parts = []
        if bl_regs is not None:
            parts.append(f"registers={bl_regs}")
        if bl_shmem is not None:
            parts.append(f"shared_mem={bl_shmem}")
        if bl_warps is not None:
            parts.append(f"warps={bl_warps}")
        block_limits = ", ".join(parts) if parts else None

    dram_pct = _parse_table_metric_value(sol, "DRAM Throughput")
    mem_pct = _parse_table_metric_value(sol, "Memory Throughput")
    l1_pct = _parse_table_metric_value(sol, "L1/TEX Cache Throughput")
    l2_pct = _parse_table_metric_value(sol, "L2 Cache Throughput")
    sm_pct = _parse_table_metric_value(sol, "Compute (SM) Throughput")

    mem_gbps = _parse_table_metric_value(mem_work, "Memory Throughput")
    mem_busy = _parse_table_metric_value(mem_work, "Mem Busy")
    max_bw = _parse_table_metric_value(mem_work, "Max Bandwidth")
    mem_pipes_busy = _parse_table_metric_value(mem_work, "Mem Pipes Busy")
    l1_hit = _parse_table_metric_value(mem_work, "L1/TEX Hit Rate")
    l2_hit = _parse_table_metric_value(mem_work, "L2 Hit Rate")

    avg_active_threads = _parse_table_metric_value(warp_state, "Avg. Active Threads Per Warp")
    avg_not_pred = _parse_table_metric_value(warp_state, "Avg. Not Predicated Off Threads Per Warp")
    warp_cyc_issued = _parse_table_metric_value(warp_state, "Warp Cycles Per Issued Instruction")
    warp_cyc_exec = _parse_table_metric_value(warp_state, "Warp Cycles Per Executed Instruction")

    exec_inst = _parse_table_metric_value(inst_stats, "Executed Instructions")
    issued_inst = _parse_table_metric_value(inst_stats, "Issued Instructions")

    branch_ratio = _parse_table_metric_value(src_cnt, "Branch Instructions Ratio")
    avg_div = _parse_table_metric_value(src_cnt, "Avg. Divergent Branches")
    branch_eff = _parse_table_metric_value(src_cnt, "Branch Efficiency")

    sm_busy = _parse_table_metric_value(comp_work, "SM Busy")
    issue_slots_busy = _parse_table_metric_value(comp_work, "Issue Slots Busy")
    issued_ipc = _parse_table_metric_value(comp_work, "Issued Ipc Active")
    exec_ipc = _parse_table_metric_value(comp_work, "Executed Ipc Active")

    metrics = {
        "GRID_SIZE": _unknown_if_none(grid),
        "BLOCK_SIZE": _unknown_if_none(block),
        "NUM_SMS": _unknown_if_none(sms),
        "WAVES_PER_SM": _unknown_if_none(waves),
        "DURATION_US": _unknown_if_none(dur),
        "ONE_OR_MORE_ELIGIBLE_PCT": _unknown_if_none(oep),
        "NO_ELIGIBLE_PCT": _unknown_if_none(nep),
        "ACTIVE_WARPS_PER_SCHED": _unknown_if_none(aws),
        "ELIGIBLE_WARPS_PER_SCHED": _unknown_if_none(ews),
        "ISSUED_WARP_PER_SCHED": _unknown_if_none(iws),
        "ACHIEVED_OCC_PCT": _unknown_if_none(aop),
        "ACHIEVED_ACTIVE_WARPS_PER_SM": _unknown_if_none(aaw),
        "BLOCK_LIMITS_OR_UNKNOWN": _unknown_if_none(block_limits),
        "DRAM_TPUT_PCT": _unknown_if_none(dram_pct),
        "MEM_TPUT_PCT": _unknown_if_none(mem_pct),
        "L1_TEX_TPUT_PCT": _unknown_if_none(l1_pct),
        "L2_TPUT_PCT": _unknown_if_none(l2_pct),
        "SM_TPUT_PCT": _unknown_if_none(sm_pct),
        "MEM_GBPS": _unknown_if_none(mem_gbps),
        "MEM_BUSY_PCT": _unknown_if_none(mem_busy),
        "MAX_BW_PCT": _unknown_if_none(max_bw),
        "MEM_PIPES_BUSY_PCT": _unknown_if_none(mem_pipes_busy),
        "L1_HIT_PCT": _unknown_if_none(l1_hit),
        "L2_HIT_PCT": _unknown_if_none(l2_hit),
        "AVG_ACTIVE_THREADS_PER_WARP": _unknown_if_none(avg_active_threads),
        "AVG_NOT_PRED_OFF_THREADS_PER_WARP": _unknown_if_none(avg_not_pred),
        "WARP_CYCLES_PER_ISSUED_INST": _unknown_if_none(warp_cyc_issued),
        "WARP_CYCLES_PER_EXEC_INST": _unknown_if_none(warp_cyc_exec),
        "EXEC_INST": _unknown_if_none(exec_inst),
        "ISSUED_INST": _unknown_if_none(issued_inst),
        "BRANCH_RATIO_PCT": _unknown_if_none(branch_ratio),
        "AVG_DIVERGENT_BRANCHES": _unknown_if_none(avg_div),
        "BRANCH_EFF_PCT": _unknown_if_none(branch_eff),
        "SM_BUSY_PCT": _unknown_if_none(sm_busy),
        "ISSUE_SLOTS_BUSY_PCT": _unknown_if_none(issue_slots_busy),
        "ISSUED_IPC_ACTIVE": _unknown_if_none(issued_ipc),
        "EXECUTED_IPC_ACTIVE": _unknown_if_none(exec_ipc),
    }

    prompt = f"""你是GPU性能分析助手。请严格按我给定的4步流程工作：Step2(瓶颈原型匹配)→Step3(计算分数，含门控与反证)→Step4(输出Top-K瓶颈与优化建议)。不要讨论roofline，不要引用任何speedup%或OPT文字，不要输出description内容。

【输入：Step1结构化指标（字段名必须来自NCU原文）】

1) Launch / Parallelism（来自 Launch Statistics + Speed Of Light）
- Grid Size = {{GRID_SIZE}}
- Block Size = {{BLOCK_SIZE}}
- # SMs = {{NUM_SMS}}
- Waves Per SM = {{WAVES_PER_SM}}
- Duration (us) = {{DURATION_US}}

2) Scheduler（来自 Scheduler Statistics）
- One or More Eligible (%) = {{ONE_OR_MORE_ELIGIBLE_PCT}}
- No Eligible (%) = {{NO_ELIGIBLE_PCT}}
- Active Warps Per Scheduler (warp) = {{ACTIVE_WARPS_PER_SCHED}}
- Eligible Warps Per Scheduler (warp) = {{ELIGIBLE_WARPS_PER_SCHED}}
- Issued Warp Per Scheduler = {{ISSUED_WARP_PER_SCHED}}

3) Occupancy（来自 Occupancy）
- Achieved Occupancy (%) = {{ACHIEVED_OCC_PCT}}
- Achieved Active Warps Per SM (warp) = {{ACHIEVED_ACTIVE_WARPS_PER_SM}}
- (可选) Block Limit Registers / Shared Mem / Warps = {{BLOCK_LIMITS_OR_UNKNOWN}}

4) Throughput / Busy（来自 GPU Speed Of Light Throughput + Memory Workload Analysis）
- DRAM Throughput (%) = {{DRAM_TPUT_PCT}}
- Memory Throughput (%) = {{MEM_TPUT_PCT}}              # Speed Of Light里的百分比
- L1/TEX Cache Throughput (%) = {{L1_TEX_TPUT_PCT}}
- L2 Cache Throughput (%) = {{L2_TPUT_PCT}}
- Compute (SM) Throughput (%) = {{SM_TPUT_PCT}}

- (Memory Workload Analysis)
  - Memory Throughput (Gbyte/s) = {{MEM_GBPS}}
  - Mem Busy (%) = {{MEM_BUSY_PCT}}
  - Max Bandwidth (%) = {{MAX_BW_PCT}}
  - Mem Pipes Busy (%) = {{MEM_PIPES_BUSY_PCT}}
  - L1/TEX Hit Rate (%) = {{L1_HIT_PCT}}
  - L2 Hit Rate (%) = {{L2_HIT_PCT}}

5) Warp / Instruction / Branch（用于tail_effect与"是否真的在跑"）
- (Warp State Statistics)
  - Avg. Active Threads Per Warp = {{AVG_ACTIVE_THREADS_PER_WARP}}
  - Avg. Not Predicated Off Threads Per Warp = {{AVG_NOT_PRED_OFF_THREADS_PER_WARP}}
  - Warp Cycles Per Issued Instruction = {{WARP_CYCLES_PER_ISSUED_INST}}
  - Warp Cycles Per Executed Instruction = {{WARP_CYCLES_PER_EXEC_INST}}

- (Instruction Statistics)
  - Executed Instructions (inst) = {{EXEC_INST}}
  - Issued Instructions (inst) = {{ISSUED_INST}}

- (Source Counters)
  - Branch Instructions Ratio (%) = {{BRANCH_RATIO_PCT}}
  - Avg. Divergent Branches = {{AVG_DIVERGENT_BRANCHES}}
  - (可选) Branch Efficiency (%) = {{BRANCH_EFF_PCT}}

6) Compute Workload（来自 Compute Workload Analysis，用于compute/issue判断）
- SM Busy (%) = {{SM_BUSY_PCT}}
- Issue Slots Busy (%) = {{ISSUE_SLOTS_BUSY_PCT}}
- Issued Ipc Active (inst/cycle) = {{ISSUED_IPC_ACTIVE}}
- Executed Ipc Active (inst/cycle) = {{EXECUTED_IPC_ACTIVE}}

【瓶颈原型集合（只能从中选择）】
MICRO-EXECUTION：
1) latency_bound
2) bandwidth_bound
3) compute_bound

【Step2：原型定义（用因果组合，不要用单阈值）】

说明：本任务仅允许从“微观执行”角度识别瓶颈（latency/bandwidth/compute），不允许输出任何结构性瓶颈分类。

A) latency_bound（微观执行）
 前提：Waves Per SM 不接近0，或 Active Warps Per Scheduler 不低
 证据组合：
 - No Eligible 高但 Active Warps 不低（说明有很多warp但都在等）
 - Warp Cycles Per (Issued/Executed) Instruction 很高
 - Mem Busy 有一定水平但 DRAM Throughput 未必高（更像latency而非带宽饱和）
 （注意：如果Grid很小导致Active本来就低，则不能判为latency主因）

B) bandwidth_bound（微观执行）
 前提：并行度足够
 证据组合：
 - DRAM Throughput (%) / Max Bandwidth (%) / Mem Busy (%) 较高
 - Mem Pipes Busy 较高
 - 同时 L2 Hit 低或访问压力大（但需要证据；若只有高hit也可能不带宽）
 反证：DRAM Throughput 极低 → bandwidth_bound接近0

C) compute_bound（微观执行）
 前提：并行度足够
 证据组合：
 - SM Busy (%) / Issue Slots Busy (%) 较高
 - Issued/Executed IPC Active 较高
 反证：SM Busy 极低 + Issued Warp Per Scheduler 极低 → compute_bound接近0

【Step3：打分规则】
- 对每个原型输出 score ∈ [0,100]，并列出"支持证据"和"反证/不确定性"。
- 必须使用：
  1) 逻辑与：多个条件同时满足才给高分
  2) 互斥/门控（gating，仅限微观三类之间）：
     - 若 DRAM Throughput(%) 很低 或 Mem Busy(%) 很低，则 bandwidth_bound 必须≤20
     - 若 SM Busy(%) 很低 或 Issue Slots Busy(%) 很低，则 compute_bound 必须≤20
     - 若 No Eligible(%) 不高 且 Warp Cycles Per (Issued/Executed) Instruction 不高，则 latency_bound 必须≤30
  3) 反证（counter-evidence）：
     - DRAM Throughput(%) 很低 → bandwidth_bound≈0
     - SM Busy(%) 很低 → compute_bound≈0
- 如果关键字段为unknown导致无法区分，请明确"需要补充哪些NCU字段"。

【Step4：输出格式（严格遵守）】
输出一个JSON（不要多余解释），字段如下：
{{
  "top_k": [
    {{
      "prototype": "...",
      "score": ...,
      "evidence": ["...", "...", "..."],
      "recommendations": [
        {{"priority": 1, "action": "...", "rationale": "..."}},
        {{"priority": 2, "action": "...", "rationale": "..."}}
      ],
      "next_metrics_to_check": ["...", "..."]
    }},
    {{
      "prototype": "...",
      "score": ...,
      "evidence": ["...", "...", "..."],
      "recommendations": [
        {{"priority": 1, "action": "...", "rationale": "..."}},
        {{"priority": 2, "action": "...", "rationale": "..."}}
      ],
      "next_metrics_to_check": ["...", "..."]
    }}
  ],
  "all_scores": [
    {{"prototype": "latency_bound", "score": ...}},
    {{"prototype": "bandwidth_bound", "score": ...}},
    {{"prototype": "compute_bound", "score": ...}}
  ]
}}

【优化建议约束】
- 建议必须可执行且贴近Triton/算子场景，按优先级输出：
  (a) 结构级：融合到下游、合并多个小调用(打包/批处理)、CUDA Graphs、改映射增大Grid
  (b) 映射级：2D grid、tile拆分、减少mask尾块
  (c) 微调级：num_warps/num_stages、寄存器与shared权衡、对齐/连续性提示
- 建议必须足够详细、具体、明确，不能只给出一个方向。
- 不要给"直接增大输入规模"这类建议；只能建议"合并多个调用/打包多个任务"来摊薄开销。

现在开始执行Step2-4并输出JSON。

【Triton kernel实现】
{kernel_code}
"""

    for k, v in metrics.items():
        prompt = prompt.replace("{" + k + "}", str(v))
    return prompt


def _draft_step2_prompt(kernel_code: str, triton_tuning_plan: str) -> str:
    return f"""你是一个经验丰富的 Triton kernel 编程专家。
请严格依据给定的 revision plan，对现有 Triton kernel 代码进行修改。
【Triton kernel实现】
{kernel_code}

【Triton调优计划】
{triton_tuning_plan}

【强制约束】
- 不得修改kernel的输入/输出接口与调用方式
- 不得改变kernel的语义（任何合法输入下，输出结果必须与原本实现完全相同）
- 不得引入任何Triton中不存在或未公开的APIs

- 除 revision plan 明确要求外
- Triton版本: Triton版本3.0.0 or later.

【输出要求】
- 仅输出修改后的完整 Triton kernel 代码
- 不得输出解释、分析、注释说明或任何无关内容
"""


def _write_ncu_runner(runner_path: str):
    with open(runner_path, "w", encoding="utf-8") as f:
        f.write("import importlib.util\n")
        f.write("import os\n")
        f.write("import runpy\n")
        f.write("import sys\n")
        f.write("\n")
        f.write("this_dir = os.path.dirname(os.path.abspath(__file__))\n")
        f.write("repo_root = None\n")
        f.write("cur = this_dir\n")
        f.write("for _ in range(12):\n")
        f.write("    if os.path.isdir(os.path.join(cur, 'GEAK-eval')):\n")
        f.write("        repo_root = cur\n")
        f.write("        break\n")
        f.write("    parent = os.path.dirname(cur)\n")
        f.write("    if parent == cur:\n")
        f.write("        break\n")
        f.write("    cur = parent\n")
        f.write("if repo_root is None:\n")
        f.write("    repo_root = os.path.abspath(os.path.join(this_dir, '..', '..'))\n")
        f.write("geak_eval_root = os.path.join(repo_root, 'GEAK-eval')\n")
        f.write("if repo_root not in sys.path:\n")
        f.write("    sys.path.insert(0, repo_root)\n")
        f.write("if geak_eval_root not in sys.path:\n")
        f.write("    sys.path.insert(0, geak_eval_root)\n")
        f.write("\n")
        f.write("import geak_eval.perf.performance_utils as _pu\n")
        f.write("_orig_run_benchmark = _pu.Performance_Metrics.run_benchmark\n")
        f.write("def _run_benchmark_first_only(self):\n")
        f.write("    if getattr(self, 'input_tensors', None):\n")
        f.write("        self.input_tensors = self.input_tensors[:1]\n")
        f.write("    return _orig_run_benchmark(self)\n")
        f.write("_pu.Performance_Metrics.run_benchmark = _run_benchmark_first_only\n")
        f.write("\n")
        f.write("gen_op_path = os.path.abspath(sys.argv[1])\n")
        f.write("op_name = sys.argv[2]\n")
        f.write("perf_script_path = os.path.abspath(sys.argv[3])\n")
        f.write("\n")
        f.write("name = os.path.splitext(os.path.basename(gen_op_path))[0]\n")
        f.write("spec = importlib.util.spec_from_file_location(name, gen_op_path)\n")
        f.write("mod = importlib.util.module_from_spec(spec)\n")
        f.write("spec.loader.exec_module(mod)\n")
        f.write("sys.modules[op_name] = mod\n")
        f.write("runpy.run_path(perf_script_path, run_name='__main__')\n")
        f.write("try:\n")
        f.write("    import torch\n")
        f.write("    if torch.cuda.is_available():\n")
        f.write("        torch.cuda.synchronize()\n")
        f.write("except Exception:\n")
        f.write("    pass\n")


def _profile_first_input(gen_op_path: str, op_name: str, out_dir: str, report_base: str) -> tuple[str, str, str]:
    os.makedirs(out_dir, exist_ok=True)

    perf_script = os.path.join(NATIVE_PERF_GOLD_ROOT, f"{op_name}_perf.py")
    if not os.path.isfile(perf_script):
        raise FileNotFoundError(f"Perf script not found: {perf_script}")

    rep_path = report_base + ".ncu-rep"
    txt_path = report_base + ".txt"

    rc, _, err = _run(["ncu", "--version"])
    if rc != 0:
        raise RuntimeError(f"ncu not available in PATH. stderr: {err}")

    runner_path = report_base + "__runner.py"
    _write_ncu_runner(runner_path)

    cmd = [
        "ncu",
        "--set",
        NCU_SET,
        "-o",
        report_base,
        sys.executable,
        runner_path,
        gen_op_path,
        op_name,
        perf_script,
    ]

    rc, out, err = _run(cmd)
    if rc != 0:
        raise RuntimeError(f"ncu profiling failed. stdout: {out}\nstderr: {err}")

    if not os.path.exists(rep_path):
        reps = [
            os.path.join(out_dir, fn)
            for fn in os.listdir(out_dir)
            if fn.endswith(".ncu-rep")
        ]
        if reps:
            reps.sort(key=lambda p: os.path.getmtime(p))
            rep_path = reps[-1]
            txt_path = os.path.splitext(rep_path)[0] + ".txt"
        else:
            raise FileNotFoundError(f"Expected report not found: {rep_path}")

    rc, out, err = _run(["ncu", "--import", rep_path])
    if rc != 0:
        raise RuntimeError(f"ncu --import failed. stdout: {out}\nstderr: {err}")

    out = _truncate_first_operator_profile(out) or out
    table = _strip_opt_lines(out)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(table)

    if os.path.exists(runner_path):
        try:
            os.remove(runner_path)
        except Exception:
            pass

    return rep_path, txt_path, table


def _evaluate_candidate(gen_op_path: str, ref_op_path: str) -> dict:
    fname = os.path.basename(ref_op_path)
    gt_root = os.path.dirname(ref_op_path)

    with open(gen_op_path, "r", encoding="utf-8") as f:
        code = f.read()

    work_dir = tempfile.mkdtemp(prefix="opt_eval_")
    log_root = os.path.join(work_dir, "log")
    exec_root = os.path.join(work_dir, "exec")
    os.makedirs(log_root, exist_ok=True)
    os.makedirs(exec_root, exist_ok=True)

    evaluator = TestAllCloseEvaluatorTBG(ground_truth_root=gt_root)
    call_status, exec_status, speedup, stdout, stderr = evaluator.execute(
        code=code,
        log_root=log_root,
        exec_root=exec_root,
        fname=fname,
        atol=ATOL,
        rtol=RTOL,
        timeout=TIMEOUT_S,
        verbose=False,
        custom_tests_path=None,
    )

    return {
        "compile_ok": bool(call_status),
        "diff_ok": bool(exec_status),
        "speedup": speedup,
        "stdout": stdout,
        "stderr": stderr,
        "work_dir": work_dir,
    }


def main():
    ref_op = os.path.abspath(REF_OP_PATH)
    if not os.path.isfile(ref_op):
        raise FileNotFoundError(f"REF_OP_PATH not found: {ref_op}")

    fname = os.path.basename(ref_op)
    op_name = os.path.splitext(fname)[0]

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    base_impl = ref_op
    best_speedup_so_far = 1.0
    history = []

    for r in range(1, int(NUM_ROUNDS) + 1):
        print(f"Round {r}")
        round_dir = None
        if r >= 2:
            round_dir = os.path.join(OUTPUT_ROOT, f"round_{r}")
            os.makedirs(round_dir, exist_ok=True)

        report_base = os.path.abspath(os.path.join(OUTPUT_ROOT, f"{REPORT_PREFIX}_r{r}_{op_name}"))
        rep_path, txt_path, table = _profile_first_input(base_impl, op_name, os.path.abspath(OUTPUT_ROOT), report_base)
        print(f"Round {r}, Profiling Finish")

        if round_dir:
            try:
                shutil.copyfile(rep_path, os.path.join(round_dir, "profile.ncu-rep"))
            except Exception:
                pass
            try:
                shutil.copyfile(txt_path, os.path.join(round_dir, "profile.txt"))
            except Exception:
                pass

        round_info = {
            "round": r,
            "base_impl": base_impl,
            "ncu_rep": rep_path,
            "ncu_txt": txt_path,
        }

        if r == 1:
            history.append(round_info)
            continue

        if not os.getenv("DASHSCOPE_API_KEY"):
            raise RuntimeError("DASHSCOPE_API_KEY is not set (required for Round>=2 LLM optimization)")

        analysis = _analyze_bottleneck(table)

        src = _read_operator_impl(base_impl)

        step1_prompt_full = _draft_step1_prompt(src, table)
        step1_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": step1_prompt_full},
        ]
        step1_raw = _qwen_chat(step1_messages).strip()
        step1_obj = _extract_json_object(step1_raw)
        plans_sorted = []
        if isinstance(step1_obj, dict):
            top_k = step1_obj.get("top_k")
            if isinstance(top_k, list):
                global_priority = 1
                for item in top_k:
                    if not isinstance(item, dict):
                        continue
                    proto = item.get("prototype")
                    score = item.get("score")
                    evidence = item.get("evidence")
                    recs = item.get("recommendations")
                    if not isinstance(recs, list):
                        continue
                    for rec in recs:
                        if not isinstance(rec, dict):
                            continue
                        local_p = rec.get("priority")
                        action = rec.get("action")
                        rationale = rec.get("rationale")
                        plan = {
                            # The LLM may output priorities that restart per-bottleneck (e.g., 1..2 for each top_k item).
                            # We need a unique global ordering for downstream sorting/task creation.
                            "priority": global_priority,
                            "local_priority": local_p,
                            "bottleneck": proto,
                            "score": score,
                            "metric_evidence": evidence,
                            "rationale": rationale,
                            "triton_tuning_plan": action,
                        }
                        plans_sorted.append(plan)
                        global_priority += 1

        plans_sorted.sort(key=lambda x: (_get_plan_priority(x) is None, _get_plan_priority(x) if _get_plan_priority(x) is not None else 10**9))

        step1_strategy_text = step1_raw

        if round_dir:
            with open(os.path.join(round_dir, "step1_prompt.txt"), "w", encoding="utf-8") as f:
                f.write(step1_prompt_full)
            with open(os.path.join(round_dir, "step1_strategy.txt"), "w", encoding="utf-8") as f:
                f.write(step1_strategy_text)
            with open(os.path.join(round_dir, "step1_plans.json"), "w", encoding="utf-8") as f:
                json.dump(plans_sorted, f, ensure_ascii=False, indent=2)

        repeat = int(OFFSPRING_REPEAT_PER_PLAN)
        if repeat <= 0:
            repeat = 1

        tasks = []
        for plan_idx, plan in enumerate(plans_sorted):
            for rep_idx in range(1, repeat + 1):
                tasks.append({"plan_index": plan_idx, "repeat_index": rep_idx, "plan": plan})

        k = int(OFFSPRING_PER_ROUND)
        if k <= 0:
            k = 1
        if tasks:
            k = min(k, len(tasks))

        best_child = None
        best_child_speedup = float("-inf")
        children = []
        for child_i in range(1, k + 1):
            child_dir = os.path.join(round_dir, f"child_{child_i}") if round_dir else OUTPUT_ROOT
            if round_dir:
                os.makedirs(child_dir, exist_ok=True)

            task = tasks[child_i - 1] if tasks and len(tasks) >= child_i else {"plan": {}}
            plan = task.get("plan")
            plan_text = plan.get("triton_tuning_plan") if isinstance(plan, dict) else str(plan)
            if not plan_text:
                plan_text = json.dumps(plan, ensure_ascii=False, indent=2) if isinstance(plan, dict) else str(plan)
            plan_priority = _get_plan_priority(plan)
            plan_index = task.get("plan_index")
            repeat_index = task.get("repeat_index")
            step2_prompt_full = _draft_step2_prompt(src, plan_text)
            step2_messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": step2_prompt_full},
            ]
            gen_code = _qwen_chat(step2_messages)
            m = re.search(r"```python\s*(.*?)```", gen_code, flags=re.S)
            if m:
                gen_code = m.group(1)
            gen_code = gen_code.strip() + "\n"

            candidate_path = os.path.abspath(
                os.path.join(child_dir, "candidate.py" if round_dir else f"{op_name}_r{r}_child{child_i}.py")
            )
            with open(candidate_path, "w", encoding="utf-8") as f:
                f.write(gen_code)

            if round_dir:
                with open(os.path.join(child_dir, "step2_prompt.txt"), "w", encoding="utf-8") as f:
                    f.write(step2_prompt_full)
                with open(os.path.join(child_dir, "plan.json"), "w", encoding="utf-8") as f:
                    f.write(plan_text)
                with open(os.path.join(child_dir, "task.json"), "w", encoding="utf-8") as f:
                    json.dump(task, f, ensure_ascii=False, indent=2)

            eval_res = _evaluate_candidate(candidate_path, ref_op)
            runnable_ok = bool(eval_res.get("compile_ok"))
            correct_ok = bool(eval_res.get("diff_ok"))
            try:
                speedup_val = float(eval_res.get("speedup", 0) or 0)
            except Exception:
                speedup_val = 0.0

            print(f"Round {r}, Child {child_i}, Runnable Test {'Success' if runnable_ok else 'Fail'}")
            print(f"Round {r}, Child {child_i}, Correct Test {'Success' if correct_ok else 'Fail'}")
            print(f"Round {r}, Child {child_i}, Performance Test {speedup_val}")

            if round_dir:
                with open(os.path.join(child_dir, "eval.json"), "w", encoding="utf-8") as f:
                    json.dump(eval_res, f, ensure_ascii=False, indent=2)

            child_info = {
                "child": child_i,
                "candidate": candidate_path,
                "plan": plan,
                "priority": plan_priority,
                "plan_index": plan_index,
                "repeat_index": repeat_index,
                "eval": eval_res,
                "runnable_ok": runnable_ok,
                "correct_ok": correct_ok,
                "speedup": speedup_val,
            }
            children.append(child_info)

            if runnable_ok and correct_ok and speedup_val > best_child_speedup:
                best_child_speedup = speedup_val
                best_child = child_info

        accept = bool(best_child) and best_child_speedup > float(best_speedup_so_far)
        if accept:
            base_impl = best_child["candidate"]
            best_speedup_so_far = float(best_child_speedup)
        else:
            base_impl = base_impl

        if round_dir:
            with open(os.path.join(round_dir, "best.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "round": r,
                        "base_speedup_so_far": best_speedup_so_far,
                        "step1_raw": step1_raw,
                        "plans": plans_sorted,
                        "offspring_repeat_per_plan": repeat,
                        "tasks": tasks,
                        "best_child": best_child,
                        "children": children,
                        "accept": accept,
                        "next_base_impl": base_impl,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

        round_info.update(
            {
                "analysis": analysis,
                "step1_raw": step1_raw,
                "plans": plans_sorted,
                "offspring_repeat_per_plan": repeat,
                "children": children,
                "best_child": best_child,
                "best_speedup_so_far": best_speedup_so_far,
                "accept": accept,
                "next_base_impl": base_impl,
            }
        )
        history.append(round_info)

    print(json.dumps({"op": op_name, "rounds": history}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
