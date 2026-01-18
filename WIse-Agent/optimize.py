import json
import os
import re
import shutil
import subprocess
import sys
import tempfile


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
NUM_ROUNDS = 10
OFFSPRING_PER_ROUND = 5

ATOL = 1e-3
RTOL = 1e-3
TIMEOUT_S = 2 * 60

OUTPUT_ROOT = r"/workspace/zibo/Geak-OPT/WIse-Agent/output"
REPORT_PREFIX = "optimize"
NCU_SET = "full"
NCU_FULL_REPORT = 0
MODEL_NAME = "qwen3-coder-plus"


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


def _qwen_chat(messages: list[dict]) -> str:
    from openai import OpenAI

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY is not set")

    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        stream=True,
    )
    out = []
    for chunk in completion:
        delta = chunk.choices[0].delta.content
        if delta:
            out.append(delta)
            print(delta, end="", flush=True)
    print()
    return "".join(out)


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


def _draft_prompts(op_name: str, table: str, strategy: str) -> tuple[str, str]:
    p1 = f"""你是 Nsight Compute (ncu) 性能分析专家，熟悉 NVIDIA Ampere（如 RTX 3090）微架构以及 Triton kernel 的实现与调优（block/tile size、program_id 映射、访存模式、num_warps、num_stages、pipeline、ILP、latency hiding 等）。

以下 profiling 结果来自 Triton 生成的单个 GPU kernel 在真实 GPU 上的执行，所有指标具有事实约束。

【严格前提约束（必须遵守）】
- 输入数据固定：shape / batch / problem size 不变
- 算子语义固定：数学语义不可改变
- 一次只分析一个 kernel
- 禁止宏观调优：
  - 算子融合/拆分、pipeline 调整
  - 改变输入规模、数值精度或算法语义

你给出的优化建议必须仅限 Triton kernel 实现层面，且必须能直接映射到代码修改点，允许的杠杆包括且仅包括：
- block / tile size（如 BLOCK_M/N/K 等）
- program_id 映射（pid 与 M/N 维度映射、swizzle）
- memory access pattern（coalescing / reuse / stride / vectorized load/store）
- L1 / L2 / shared memory 行为
- num_warps / num_stages / software pipeline
- latency hiding、warp divergence、serialization

【任务要求】
结合下面这张 ncu 表格（仅此表），判断该 Triton kernel 当前实现的主导性能瓶颈类型：
Memory-bound / Latency-bound / Instruction-bound / Occupancy-limited / Serialization-bound 等。

你必须：
1) 明确指出对应的 ncu 指标（来自表格中 Metric Name/Value），并给出因果链：指标 -> 微架构行为 -> 性能受限。
2) 仅分析可通过 Triton kernel 实现修改解决的瓶颈（禁止归因于不可变硬件或理论峰值）。
3) 给出 ONE 条具体可执行的优化策略句子，必须点名可修改的 Triton 代码层杠杆（例如 num_warps/num_stages、tl.multiple_of、tl.max_contiguous、向量化 load/store、tile/block 大小、pid 映射）。

【严格输出格式（必须完全遵守）】
当前 Triton kernel 的核心实现瓶颈在于 XXX-bound。
Evidence 包括：
ncu 指标 A 显示 ……
ncu 指标 B 表明 ……
这些指标反映 ……，因此性能主要受限于 ……
Optimization: <两到三条短句，必须是可直接映射到 Triton 代码修改的策略>

【ncu 表格】
{table}
"""

    p2 = f"""你是一个资深 Triton kernel 优化工程师，目标是在 NVIDIA GPU（CUDA）上优化 Triton 算子实现。

你将收到：
- 来自 step1 的优化策略（包含 2-3 条短句，带明确的调参/代码杠杆）
- 当前算子实现代码（仅包含实现，不含测试）

你的任务：
- 严格落实下方“优化策略”，对当前实现进行改写，以提升单 kernel 性能。

【优化策略（必须遵循并落实到代码改动）】
{strategy}

【硬性约束（必须遵守）】
1) 语义不变：数学语义、输出必须保持一致。
2) 接口不变：函数名、函数签名、输入输出类型保持一致。
3) 不允许宏观改动：禁止算子融合/拆分、禁止改变输入规模、禁止改变数值精度或算法语义。
4) 一次只针对一个 kernel 的实现做优化。

【允许的优化手段（仅限 Triton 实现层面）】
- block/tile size 调整（BLOCK_*、num_warps、num_stages）
- program_id 映射/线程块映射（pid -> (m,n) 的映射、swizzle、分块方式）
- 访存模式优化：coalescing、减少 stride、提高 reuse、向量化 load/store（例如把标量改为 tl.load/tl.store 的更大连续块）
- 使用 tl.multiple_of / tl.max_contiguous 传递对齐与连续性信息
- 合理使用 software pipeline / latency hiding（num_stages、ILP），降低 divergence/serialization
- 合理使用 shared memory / L1/L2 行为（仅在不改变语义前提下）

【输出要求（必须严格遵守）】
- 只输出修改后的完整 Python 代码，且必须放在一个 ```python 代码块``` 中。
- 不要输出解释、不要输出分析文字、不要输出多余的 Markdown。
- 如果你需要新增/修改 autotune config（如 triton.autotune / num_warps 候选），可以做，但必须保持接口不变。
"""

    return p1, p2


def _write_ncu_runner(runner_path: str):
    with open(runner_path, "w", encoding="utf-8") as f:
        f.write("import importlib.util\n")
        f.write("import os\n")
        f.write("import runpy\n")
        f.write("import sys\n")
        f.write("\n")
        f.write("this_dir = os.path.dirname(os.path.abspath(__file__))\n")
        f.write("repo_root = os.path.abspath(os.path.join(this_dir, '..', '..'))\n")
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
    if int(NCU_FULL_REPORT) == 1:
        cmd.append("-f")
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

    if int(NCU_FULL_REPORT) == 1:
        table = out
    else:
        table = _extract_first_gpu_sol_table(out) or out
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

        prompt1, _ = _draft_prompts(op_name, table, analysis["optimization_strategy"])

        src = _read_operator_impl(base_impl)

        step1_prompt_full = prompt1 + "\n\nCurrent operator implementation:\n```python\n" + src + "```\n"
        step1_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": step1_prompt_full},
        ]
        strategy_text = _qwen_chat(step1_messages).strip()
        strategy_text = re.sub(r"```.*?```", "", strategy_text, flags=re.S)
        strategy_text = strategy_text.strip()

        if round_dir:
            with open(os.path.join(round_dir, "step1_prompt.txt"), "w", encoding="utf-8") as f:
                f.write(step1_prompt_full)
            with open(os.path.join(round_dir, "step1_strategy.txt"), "w", encoding="utf-8") as f:
                f.write(strategy_text)

        k = int(OFFSPRING_PER_ROUND)
        if k <= 0:
            k = 1

        best_child = None
        best_child_speedup = float("-inf")
        children = []
        for child_i in range(1, k + 1):
            child_dir = os.path.join(round_dir, f"child_{child_i}") if round_dir else OUTPUT_ROOT
            if round_dir:
                os.makedirs(child_dir, exist_ok=True)

            _, prompt2 = _draft_prompts(op_name, table, strategy_text)

            step2_prompt_full = prompt2 + "\n\nCurrent operator implementation:\n```python\n" + src + "```\n"
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
                "strategy": strategy_text,
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
                        "strategy": strategy_text,
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
                "prompt_step1": prompt1,
                "strategy": strategy_text,
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
