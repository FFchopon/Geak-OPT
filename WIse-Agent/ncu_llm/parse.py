import argparse
import os
import re


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


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

        # Capture the last numeric token in the line.
        m = re.search(r"(-?[0-9]+(?:\.[0-9]+)?)\s*$", s)
        if not m:
            return None
        return m.group(1)

    return None


def _parse_table_last_token(block: str, metric_name: str) -> str | None:
    if not block:
        return None

    for line in block.splitlines():
        s = line.strip()
        if not s:
            continue
        if not s.startswith(metric_name):
            continue

        parts = s.split()
        if not parts:
            return None
        return parts[-1]

    return None


PROMPT_TEMPLATE = """你是GPU性能分析助手。请严格按我给定的4步流程工作：Step2(瓶颈原型匹配)→Step3(计算分数，含门控与反证)→Step4(输出Top-K瓶颈与优化建议)。不要讨论roofline，不要引用任何speedup%或OPT文字，不要输出description内容。

【输入：Step1结构化指标（字段名必须来自NCU原文）】

1) Launch / Parallelism（来自 Launch Statistics + Speed Of Light）
- Grid Size = {GRID_SIZE}
- Block Size = {BLOCK_SIZE}
- # SMs = {NUM_SMS}
- Waves Per SM = {WAVES_PER_SM}
- Duration (us) = {DURATION_US}

2) Scheduler（来自 Scheduler Statistics）
- One or More Eligible (%) = {ONE_OR_MORE_ELIGIBLE_PCT}
- No Eligible (%) = {NO_ELIGIBLE_PCT}
- Active Warps Per Scheduler (warp) = {ACTIVE_WARPS_PER_SCHED}
- Eligible Warps Per Scheduler (warp) = {ELIGIBLE_WARPS_PER_SCHED}
- Issued Warp Per Scheduler = {ISSUED_WARP_PER_SCHED}

3) Occupancy（来自 Occupancy）
- Achieved Occupancy (%) = {ACHIEVED_OCC_PCT}
- Achieved Active Warps Per SM (warp) = {ACHIEVED_ACTIVE_WARPS_PER_SM}
- (可选) Block Limit Registers / Shared Mem / Warps = {BLOCK_LIMITS_OR_UNKNOWN}

4) Throughput / Busy（来自 GPU Speed Of Light Throughput + Memory Workload Analysis）
- DRAM Throughput (%) = {DRAM_TPUT_PCT}
- Memory Throughput (%) = {MEM_TPUT_PCT}              # Speed Of Light里的百分比
- L1/TEX Cache Throughput (%) = {L1_TEX_TPUT_PCT}
- L2 Cache Throughput (%) = {L2_TPUT_PCT}
- Compute (SM) Throughput (%) = {SM_TPUT_PCT}

- (Memory Workload Analysis)
  - Memory Throughput (Gbyte/s) = {MEM_GBPS}
  - Mem Busy (%) = {MEM_BUSY_PCT}
  - Max Bandwidth (%) = {MAX_BW_PCT}
  - Mem Pipes Busy (%) = {MEM_PIPES_BUSY_PCT}
  - L1/TEX Hit Rate (%) = {L1_HIT_PCT}
  - L2 Hit Rate (%) = {L2_HIT_PCT}

5) Warp / Instruction / Branch（用于tail_effect与“是否真的在跑”）
- (Warp State Statistics)
  - Avg. Active Threads Per Warp = {AVG_ACTIVE_THREADS_PER_WARP}
  - Avg. Not Predicated Off Threads Per Warp = {AVG_NOT_PRED_OFF_THREADS_PER_WARP}
  - Warp Cycles Per Issued Instruction = {WARP_CYCLES_PER_ISSUED_INST}
  - Warp Cycles Per Executed Instruction = {WARP_CYCLES_PER_EXEC_INST}

- (Instruction Statistics)
  - Executed Instructions (inst) = {EXEC_INST}
  - Issued Instructions (inst) = {ISSUED_INST}

- (Source Counters)
  - Branch Instructions Ratio (%) = {BRANCH_RATIO_PCT}
  - Avg. Divergent Branches = {AVG_DIVERGENT_BRANCHES}
  - (可选) Branch Efficiency (%) = {BRANCH_EFF_PCT}

6) Compute Workload（来自 Compute Workload Analysis，用于compute/issue判断）
- SM Busy (%) = {SM_BUSY_PCT}
- Issue Slots Busy (%) = {ISSUE_SLOTS_BUSY_PCT}
- Issued Ipc Active (inst/cycle) = {ISSUED_IPC_ACTIVE}
- Executed Ipc Active (inst/cycle) = {EXECUTED_IPC_ACTIVE}

【瓶颈原型集合（只能从中选择）】
STRUCTURAL:
1) insufficient_parallelism
2) tail_effect
MICRO-EXECUTION（仅当并行度足够时才允许高分）:
3) latency_bound
4) bandwidth_bound
5) compute_bound

【Step2：原型定义（用因果组合，不要用单阈值）】

A) insufficient_parallelism（结构性）
典型证据组合：
- Grid Size 显著小于 #SMs 或 Waves Per SM≈0
- Scheduler：Active Warps Per Scheduler 很低，Eligible/Issued 很低，No Eligible 很高
- 同时 SM Busy / Mem Busy / Throughput 都偏低（说明不是打满，而是没活）

B) tail_effect（结构性）
必须有证据才高分（否则≤30）：
- Avg. Active Threads Per Warp 明显偏低（大量mask/尾块）
- 或 Branch 发散：Avg. Divergent Branches > 0 / 分支相关指标异常
- 或 Warp Cycles Per Instruction 异常高且伴随 active threads 低（提示大量 predication/mask）

C) latency_bound（微观执行，需并行度足够）
前提：Waves Per SM 不接近0，或 Active Warps Per Scheduler 不低
证据组合：
- No Eligible 高但 Active Warps 不低（说明有很多warp但都在等）
- Warp Cycles Per (Issued/Executed) Instruction 很高
- Mem Busy 有一定水平但 DRAM Throughput 未必高（更像latency而非带宽饱和）
（注意：如果Grid很小导致Active本来就低，则不能判为latency主因）

D) bandwidth_bound（微观执行）
前提：并行度足够
证据组合：
- DRAM Throughput (%) / Max Bandwidth (%) / Mem Busy (%) 较高
- Mem Pipes Busy 较高
- 同时 L2 Hit 低或访问压力大（但需要证据；若只有高hit也可能不带宽）
反证：DRAM Throughput 极低 → bandwidth_bound接近0

E) compute_bound（微观执行）
前提：并行度足够
证据组合：
- SM Busy (%) / Issue Slots Busy (%) 较高
- Issued/Executed IPC Active 较高
反证：SM Busy 极低 + Issued Warp Per Scheduler 极低 → compute_bound接近0

【Step3：打分规则】
- 对每个原型输出 score ∈ [0,100]，并列出“支持证据”和“反证/不确定性”。
- 必须使用：
  1) 逻辑与：多个条件同时满足才给高分
  2) 门控（gating）：
     - 若 insufficient_parallelism score ≥ 70，则 bandwidth_bound/compute_bound 最高≤20；
       latency_bound 最高≤30，除非明确显示“Active Warps Per Scheduler不低但No Eligible高且Warp cycles很高”
  3) 反证（counter-evidence）：
     - DRAM Throughput(%) 很低 → bandwidth_bound≈0
     - SM Busy(%) 很低 → compute_bound≈0
     - 缺少tail证据（active threads不低、divergent branches=0）→ tail_effect≤30
- 如果关键字段为unknown导致无法区分，请明确“需要补充哪些NCU字段”。

【Step4：输出格式（严格遵守）】
输出一个JSON（不要多余解释），字段如下：
{
  "top_k": [
    {
      "prototype": "...",
      "score": ...,
      "evidence": ["...", "...", "..."],
      "recommendations": [
        {"priority": 1, "action": "...", "rationale": "..."},
        {"priority": 2, "action": "...", "rationale": "..."},
        {"priority": 3, "action": "...", "rationale": "..."}
      ],
      "next_metrics_to_check": ["...", "..."]
    }
  ],
  "all_scores": [
    {"prototype": "insufficient_parallelism", "score": ...},
    {"prototype": "tail_effect", "score": ...},
    {"prototype": "latency_bound", "score": ...},
    {"prototype": "bandwidth_bound", "score": ...},
    {"prototype": "compute_bound", "score": ...}
  ]
}

【优化建议约束】
- 建议必须可执行且贴近Triton/算子场景，按优先级输出：
  (a) 结构级：融合到下游、合并多个小调用(打包/批处理)、CUDA Graphs、改映射增大Grid
  (b) 映射级：2D grid、tile拆分、减少mask尾块
  (c) 微调级：num_warps/num_stages、寄存器与shared权衡、对齐/连续性提示
- 不要给“直接增大输入规模”这类建议；只能建议“合并多个调用/打包多个任务”来摊薄开销。

现在开始执行Step2-4并输出JSON。
"""


def _unknown_if_none(v: str | None) -> str:
    if v is None:
        return "unknown"
    s = str(v).strip()
    return s if s else "unknown"


def parse_metrics(profile_text: str) -> dict[str, str]:
    sol = _find_section_block(profile_text, "GPU Speed Of Light Throughput") or ""
    launch = _find_section_block(profile_text, "Launch Statistics") or ""
    sched = _find_section_block(profile_text, "Scheduler Statistics") or ""
    occ = _find_section_block(profile_text, "Occupancy") or ""
    mem_work = _find_section_block(profile_text, "Memory Workload Analysis") or ""
    warp_state = _find_section_block(profile_text, "Warp State Statistics") or ""
    inst_stats = _find_section_block(profile_text, "Instruction Statistics") or ""
    src_cnt = _find_section_block(profile_text, "Source Counters") or ""
    comp_work = _find_section_block(profile_text, "Compute Workload Analysis") or ""

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

    return {
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


def render_prompt(metrics: dict[str, str]) -> str:
    # NOTE: PROMPT_TEMPLATE contains JSON examples with many '{' / '}' braces.
    # Using str.format() would treat them as placeholders and may raise KeyError.
    # We only replace the specific metric placeholders we own.
    out = PROMPT_TEMPLATE
    for k, v in metrics.items():
        out = out.replace("{" + k + "}", str(v))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "profile.txt"),
        help="Path to NCU exported text report (e.g., profile.txt)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "filled_prompt.txt"),
        help="Path to write the filled prompt",
    )

    args = parser.parse_args()

    profile_text = _read_text(args.input)
    metrics = parse_metrics(profile_text)
    prompt = render_prompt(metrics)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(prompt)

    print(f"Saved prompt to: {args.output}")


if __name__ == "__main__":
    main()
