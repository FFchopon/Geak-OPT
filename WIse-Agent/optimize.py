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
    repo_root = os.path.abspath(os.path.join(this_dir, ".."))
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
TUNE_NUMS_PER_BOTTLENECK = 2
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


def _extract_json_array(text: str) -> list | None:
    if not text:
        return None

    m = re.search(r"```json\s*(\[.*?\])\s*```", text, flags=re.S | re.I)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass

    start = text.find("[")
    if start < 0:
        return None

    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                cand = text[start : i + 1]
                try:
                    return json.loads(cand)
                except Exception:
                    pass

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


def _extract_first_gpu_sol_table(text: str) -> str | None:
    if not text:
        return None

    header = "Section: GPU Speed Of Light Throughput"
    start = text.find(header)
    if start < 0:
        return None

    lines = text[start:].splitlines()
    out_lines = []

    def _is_sep_line(line: str) -> bool:
        s = line.strip()
        if not s:
            return False
        s2 = s.replace(" ", "")
        return s2 != "" and all(ch == "-" for ch in s2)

    sep_count = 0
    for line in lines:
        out_lines.append(line.rstrip())
        if _is_sep_line(line):
            sep_count += 1
            if sep_count >= 3:
                break

    return "\n".join(out_lines).strip() + "\n"


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


def _extract_table_from_section(lines: list[str], section_start_idx: int) -> str | None:
    if section_start_idx < 0 or section_start_idx >= len(lines):
        return None

    def _is_sep_line(line: str) -> bool:
        s = line.strip()
        if not s:
            return False
        s2 = s.replace(" ", "")
        return s2 != "" and all(ch == "-" for ch in s2)

    out_lines = [lines[section_start_idx].rstrip()]
    sep_count = 0
    for i in range(section_start_idx + 1, len(lines)):
        ln = lines[i]
        if ln.lstrip().startswith("Section:"):
            break
        out_lines.append(ln.rstrip())
        if _is_sep_line(ln):
            sep_count += 1
            if sep_count >= 3:
                break

    table = "\n".join(out_lines).strip() + "\n"
    table = _strip_opt_lines(table)
    return table


def _extract_first_n_sections(text: str, n: int) -> str | None:
    if not text:
        return None

    if n <= 0:
        return None

    lines = text.splitlines()
    section_starts = [i for i, ln in enumerate(lines) if ln.lstrip().startswith("Section:")]
    if not section_starts:
        return None

    blocks = []
    for start in section_starts[:n]:
        block = _extract_table_from_section(lines, start)
        if block:
            blocks.append(block.rstrip())

    if not blocks:
        return None
    return "\n\n".join(blocks).strip() + "\n"


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


def _draft_step1_prompt(kernel_code: str, ncu_res: str, tune_nums4each_bottleneck: int) -> str:
    return f"""你是一个Nsight Compute (ncu) 性能分析专家，熟悉NVIDIA Ampere微架构以及Triton kernel实现与调优。
以下profiling 结果来自Triton生成的单个GPU kernel在真实GPU上的执行，所有指标具有事实约束。

【输入信息】
硬件信息：GPU：Ampere RTX 3090，82 SM; Warp Size：32; 每SM：128 CUDA Cores
当前Triton kernel实现: {kernel_code}
ncu profile关键指标结果: {ncu_res}

【任务要求】
1. 基于上述Kernel实现和对应的ncu profile结果，分析当前单个Kernel实现的性能瓶颈。
\t- Kernel输入数据固定（shape/problem size固定不变）情况下的，Kernel实现的瓶颈
\t- 忽略任何从宏观角度的瓶颈分析与调优，包括输入规模、模型或pipeline调整、算子融合
2. 基于每个Kernel实现的瓶颈（注：输入固定），为当前Triton kernel给出{int(tune_nums4each_bottleneck)}种不同的kernel调优建议：
\t- 每个调优方案必须直接且唯一地针对该Kernel实现的瓶颈，忽略其他因素导致的瓶颈
\t- 优化建议仅限Triton kernel实现层面，包括且仅包括：block/tile size, program_id映射, memory access pattern（coalescing/reuse/stride）, L1/L2/shared memory行为, num_warps/num_stages, latency hiding, warp divergence, serialization
\t- 每个调优方案按"预期收益"×"实施复杂度"的综合优先级排序，其中，收益更大、修改更简单的调优plan优先级更高

【重要约束（必须遵守）】
- 你必须输出至少 2 个不同的 bottleneck。
- 对于每一个 bottleneck，你必须输出恰好 {int(tune_nums4each_bottleneck)} 条不同的 triton_tuning_plan（也就是数组中会出现同名 bottleneck 的多条元素）。
- priority 必须为从 1 开始的连续整数，并严格按 priority 从小到大排序输出。

【严格输出格式】
- 仅输出一个 JSON 数组，每个元素对应一个瓶颈及其调优方案：
- 不得输出解释、分析、注释说明或任何无关内容
- 允许多个元素的 bottleneck 相同，但 triton_tuning_plan 必须不同（对应同一瓶颈的多个方案）
- 必须按 priority 从小到大排序输出（priority 越小优先级越高）
输入形式如下：
[
  {{
    "priority": 1,
    "bottleneck": "...",
    "metric_evidence": "Explicitly list the supporting ncu metrics and explain their implications",
    "triton_tuning_plan": "Describe concrete Triton kernel code-level modifications"
  }},
  {{
    "priority": 2,
    "bottleneck": "...",
    "metric_evidence": "...",
    "triton_tuning_plan": "..."
  }},
    {{
    "priority": 3,
    "bottleneck": "...",
    "metric_evidence": "...",
    "triton_tuning_plan": "..."
  }}
]
"""


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

    cmd = ["ncu"]
    if int(NCU_FULL_REPORT) == 0:
        cmd += [
            "-f",
            "-o",
            report_base,
            sys.executable,
            runner_path,
            gen_op_path,
            op_name,
            perf_script,
        ]
    else:
        cmd += [
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

    if int(NCU_FULL_REPORT) == 0:
        table = _extract_first_n_sections(out, 4) or out
    else:
        table = _extract_first_n_sections(out, 1) or (_extract_first_gpu_sol_table(out) or out)
    table = _strip_opt_lines(table)
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

        step1_prompt_full = _draft_step1_prompt(src, table, int(TUNE_NUMS_PER_BOTTLENECK))
        step1_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": step1_prompt_full},
        ]
        step1_raw = _qwen_chat(step1_messages).strip()
        plans = _extract_json_array(step1_raw)
        if not isinstance(plans, list):
            plans = []

        plans_sorted = list(plans)
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
