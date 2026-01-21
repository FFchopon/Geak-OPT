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
OFFSPRING_PER_ROUND = 6

ANCESTOR_NUM = 3
TEMPERATURE = 0.7
MAX_PERF_DEBUG_NUM = 3
DESCENDANT_DEBUG = 1

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


def _read_operator_tests(fpath: str) -> str:
    with open(fpath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if line.strip() == TESTS_SEP_LINE:
            return "".join(lines[i + 1 :]).rstrip() + "\n"
    return ""


def _ensure_repo_imports():
    """Make sure we can import Geak src modules when this script runs standalone."""
    this_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(this_dir, ".."))
    src_root = os.path.join(repo_root, "src")
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    if src_root not in sys.path:
        sys.path.insert(0, src_root)



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


def _qwen_generate(messages: list[dict], temperature: float = 0.7, max_tokens: int | None = None) -> str:
    # Reuse the existing DashScope-compatible chat.
    # max_tokens is kept for compatibility with the GaAgent-like call signature.
    _ = max_tokens
    return _qwen_chat(messages=messages, temperature=temperature)


def _normalize_sigs(function_signatures: list[str] | None) -> str:
    if not function_signatures:
        return ""
    lines = []
    for sig in function_signatures:
        if not sig:
            continue
        s = str(sig).strip()
        if s.startswith("*"):
            s = s.lstrip("* ").strip()
        s = " ".join(s.split())
        lines.append(s)
    out = "\n".join(f"* {ln}" for ln in lines)
    return out + ("\n" if out else "")


def _build_history_prompt(history: list, max_items: int = 5) -> str:
    """Match GaAgent style (Attempt + Code + Test Results + Analysis)."""
    if not history:
        return ""
    history_template = """
### Attempt {attempt_number}
- Code: 
```python
{code}
```

- Test Results: 
{test_results}


- Analysis:
{reflection}
"""
    out = ""
    for i, rc in enumerate(history[-max_items:]):
        # NOTE: baseline evaluator returns (compile_ok, diff_ok, speedup)
        if getattr(rc, "pass_perf", False):
            test_txt = """
runnable test: Succeed
correctness test: Succeed
speedup: {speedup}
""".format(speedup=getattr(rc, "latency", 0.0))
        elif getattr(rc, "pass_exe", False):
            test_txt = """
runnable test: Succeed
correctness test: Succeed
"""
        elif getattr(rc, "pass_call", False):
            test_txt = """
runnable test: Succeed
correctness test: Failed
"""
        else:
            test_txt = """
runnable test: Failed
correctness test: Failed
"""
        out += history_template.format(
            attempt_number=i + 1,
            code=getattr(rc, "code", "") or "",
            test_results=test_txt.strip(),
            reflection=getattr(rc, "reflections", "") or "",
        )
    return out


def _call_llm_code(prompt: str, temperature: float):
    from utils.utils import clear_json, clear_code

    msg = [{"role": "user", "content": prompt}]
    response = _qwen_generate(msg, temperature=temperature, max_tokens=8192)
    opti = clear_json(response)
    if isinstance(opti, dict) and ("code" in opti) and ("strategy" in opti):
        return clear_code(opti["code"]), opti["strategy"]
    raise ValueError(f"LLM response is not a valid {{code,strategy}} JSON. clear_json={opti}.")


def _call_llm_reflection(prompt: str, temperature: float) -> str:
    # In GaAgent, reflections are stored raw; we do the same.
    msg = [{"role": "user", "content": prompt}]
    return _qwen_generate(msg, temperature=temperature, max_tokens=8192)


def _write_text(path: str, content: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content if content is not None else "")


def _generate_reflection_for_code(
    *,
    instruction: str,
    function_signatures_text: str,
    metrics_info: str,
    evolution_history: str,
    current_program: str,
    pass_exe: bool,
    speedup: float,
    err_msg: str,
    temperature: float,
):
    from prompts import prompt_for_reflection

    if pass_exe:
        result_txt = f"""
- runnable test: Succeed
- correctness test: Succeed
- speedup: {speedup}
"""
        reflect_txt = prompt_for_reflection.prompt_evolve_strategy_optimize.format(
            instruction=instruction,
            function_signatures=function_signatures_text,
            metrics_info=metrics_info,
            evolution_history=evolution_history,
            current_program=current_program,
            test_result=result_txt,
            reflection="",
        )
    else:
        result_txt = f"""
- runnable test: Failed
- correctness test: Failed
Error Message: {err_msg}
"""
        reflect_txt = prompt_for_reflection.prompt_evolve_reflect.format(
            instruction=instruction,
            function_signatures=function_signatures_text,
            metrics_info=metrics_info,
            evolution_history=evolution_history,
            current_program=current_program,
            test_result=result_txt,
            reflection="",
        )
    out = _call_llm_reflection(reflect_txt, temperature=temperature)
    return out, reflect_txt


def main():
    _ensure_repo_imports()

    from dataloaders.ProblemState import tempCode
    from prompts import prompt_for_generation
    from utils.utils import infer_function_signatures_from_test_code

    ref_op = os.path.abspath(REF_OP_PATH)
    if not os.path.isfile(ref_op):
        raise FileNotFoundError(f"REF_OP_PATH not found: {ref_op}")

    fname = os.path.basename(ref_op)
    op_name = os.path.splitext(fname)[0]

    os.makedirs(OUTPUT_ROOT, exist_ok=True)


    if not os.getenv("DASHSCOPE_API_KEY"):
        raise RuntimeError("DASHSCOPE_API_KEY is not set")

    # Controls (keep in top-level constants)
    temperature = float(TEMPERATURE)
    ancestor_num = int(ANCESTOR_NUM)
    descendant_num = int(OFFSPRING_PER_ROUND)
    max_perf_debug_num = int(MAX_PERF_DEBUG_NUM)
    descendant_debug = int(DESCENDANT_DEBUG)

    # Build instruction/signatures from dataset operator file.
    base_src = _read_operator_impl(ref_op)
    test_code = _read_operator_tests(ref_op)
    sigs = infer_function_signatures_from_test_code(test_code) if test_code else None
    fss_text = _normalize_sigs(sigs)
    instruction = (
        "Optimize the following Triton operator for NVIDIA GPUs. "
        "Preserve correctness strictly and improve speedup against the reference kernel."
    )

    metrics_info = """
- runnable test: test if the code can be successfully executed.
- correctness test: test if the output of the code is correct.
- speedup: measures latency compared with the golden reference code.
"""

    # State (match GaAgent memory fields but keep it local & single-op).
    perf_candidates: list[tuple] = []  # (code, speedup, eff, reflections, profiling)
    history: list[list[tempCode]] = [[] for _ in range(max(1, descendant_num))]
    raw_codes: list[tempCode] | None = None
    perf_debug_num = 0
    best_solution_code = base_src
    best_solution_speedup = 0.0
    best_solution_profiling = None

    # Evaluate & profile the dataset original impl as the initial baseline candidate.
    init_eval_dir = os.path.join(OUTPUT_ROOT, "round_0")
    os.makedirs(init_eval_dir, exist_ok=True)
    base_path0 = os.path.join(init_eval_dir, f"{op_name}_base.py")
    with open(base_path0, "w", encoding="utf-8") as f:
        f.write(base_src)
    eval0 = _evaluate_candidate(base_path0, ref_op)
    if bool(eval0.get("compile_ok")) and bool(eval0.get("diff_ok")):
        try:
            best_solution_speedup = float(eval0.get("speedup", 0.0) or 0.0)
        except Exception:
            best_solution_speedup = 0.0
        report_base0 = os.path.abspath(os.path.join(init_eval_dir, f"{REPORT_PREFIX}_r0_{op_name}"))
        try:
            _, _, table0 = _profile_first_input(base_path0, op_name, init_eval_dir, report_base0)
            best_solution_profiling = table0
        except Exception:
            best_solution_profiling = None
        # Seed perf_candidates with the dataset implementation.
        seed = (base_src, best_solution_speedup, 0.0, "", best_solution_profiling)
        perf_candidates.append(seed)
        perf_candidates.sort(key=lambda x: x[1], reverse=True)

    # Main optimization loop.
    for r in range(1, int(NUM_ROUNDS) + 1):
        round_dir = os.path.join(OUTPUT_ROOT, f"round_{r}")
        os.makedirs(round_dir, exist_ok=True)
        print(f"Round {r} | current best speedup={best_solution_speedup}")

        # Reset or keep raw_codes (GaAgent-style rewrite branch).
        if (perf_debug_num >= max_perf_debug_num):
            perf_debug_num = 0
            raw_codes = None

        # Build base prompt.
        text = prompt_for_generation.prompt.format(
            instruction=instruction,
            function_signatures=fss_text,
            reference_section="",
        )

        # Optimization mode prompt: keep exactly consistent with GaAgent.generate_solution.
        if perf_candidates and (not raw_codes):
            text += """\nThere are some Optimized codes(NO.1, NO.2 and so on) to solve the Problem. The Optimized codes are arranged in ascending order based on their performance, where higher speedup indicates better performance. According to their performance(speedup is the latency compared with golden reference code) and the corresponding analysis, you need to generate a new code with better performance. You should maintain code correctness during optimization."""
            text +="\nYou can use optimization strategies such as Memory access efficiency, Hardware resource utilization, IR analysis, Assembly analysis, Kernel occupancy, TorchInductor with Triton tuning knobs and Auto-tunable kernel configurations and environment variables."    
            for i, cand in enumerate(perf_candidates):
                text += f"\n### Reference {i+1}"
                text += f"\nOptimized code: {cand[0]}"
                text += f"\nOptimized speedup: {cand[1]}"
                if cand[3]:
                    text += f"\nStrategy: {cand[3]}"
                if cand[4]:
                    text += f"\nNsight Compute (ncu) profiling result:{cand[4]}"
                text += "\nAnalyze and compare all optimization strategies based on Optimized Implementation codes and give a better strategy motivated by them. Based on the better strategy generate a better optimization code to get a higher speedup."

        # Generate or rewrite offspring.
        if raw_codes:
            for i, rc in enumerate(raw_codes):
                if getattr(rc, "pass_perf", False):
                    continue
                history_text = _build_history_prompt(history[i], max_items=5)
                text_temp = text + f"\nPrevious attempt implementations:{history_text}" + prompt_for_generation.system_prompt
                rc.gen_prompt = text_temp
                code, strat = _call_llm_code(text_temp, temperature=temperature)
                rc.code = code
                rc.strategy = strat
                rc.reflections = None
                rc.pass_call = False
                rc.pass_exe = False
                rc.pass_perf = False
            perf_debug_num += 1
        else:
            raw_codes = []
            for i in range(descendant_num):
                text_temp = text + prompt_for_generation.system_prompt
                text_temp += "\nCRITICAL: All Triton kernels (e.g., _fwd_kernel/_bwd_*_kernel) MUST be decorated with @triton.jit and invoked as kernel[grid](...). Do NOT implement them as plain Python functions."
                gen_prompt = text_temp
                code, strat = _call_llm_code(text_temp, temperature=temperature)
                rc = tempCode(code=code, strategy=strat)
                rc.gen_prompt = gen_prompt
                raw_codes.append(rc)

        # Evaluate offspring.
        offspring_rows = []
        child_dirs = []
        pass_exe_count = 0
        for i, rc in enumerate(raw_codes):
            child_dir = os.path.join(round_dir, f"child_{i+1}")
            os.makedirs(child_dir, exist_ok=True)
            child_dirs.append(child_dir)
            cand_path = os.path.join(child_dir, "candidate.py")
            with open(cand_path, "w", encoding="utf-8") as f:
                f.write((rc.code or "").rstrip() + "\n")

            # Dump prompts used to produce this candidate.
            if getattr(rc, "gen_prompt", None):
                _write_text(os.path.join(child_dir, "gen_prompt.txt"), str(rc.gen_prompt))

            eval_res = _evaluate_candidate(cand_path, ref_op)
            rc.test_stdout = eval_res.get("stdout")
            rc.test_stderr = eval_res.get("stderr")
            rc.pass_call = bool(eval_res.get("compile_ok"))
            rc.pass_exe = bool(eval_res.get("diff_ok"))
            if rc.pass_exe:
                pass_exe_count += 1
            try:
                rc.latency = float(eval_res.get("speedup", 0.0) or 0.0)
            except Exception:
                rc.latency = 0.0

            # Optional profiling per child (only for correctness-passing ones).
            rc.profilig = None
            if rc.pass_exe:
                report_base = os.path.abspath(os.path.join(child_dir, f"{REPORT_PREFIX}_r{r}_{op_name}_c{i+1}"))
                try:
                    _, _, table = _profile_first_input(cand_path, op_name, child_dir, report_base)
                    rc.profilig = table
                except Exception:
                    rc.profilig = None

            # Perf pass criteria matches GaAgent: speedup>0 and correctness.
            rc.pass_perf = bool(rc.pass_exe and (rc.latency > 0.0))

            with open(os.path.join(child_dir, "eval.json"), "w", encoding="utf-8") as f:
                json.dump(eval_res, f, ensure_ascii=False, indent=2)

            offspring_rows.append(
                {
                    "i": i,
                    "pass_call": bool(rc.pass_call),
                    "pass_exe": bool(rc.pass_exe),
                    "speedup": float(rc.latency or 0.0),
                }
            )

        # Determine round-level pass_exe similar to GaAgent.
        round_pass_exe = pass_exe_count >= max(0, min(descendant_debug, len(raw_codes)))

        # Generate reflections for evaluated offspring.
        for i, rc in enumerate(raw_codes):
            if rc.reflections:
                continue
            history_text = _build_history_prompt(history[i], max_items=5)
            err = str(rc.test_stderr or rc.test_stdout or "")
            refl_out, refl_prompt = _generate_reflection_for_code(
                instruction=instruction,
                function_signatures_text=fss_text,
                metrics_info=metrics_info,
                evolution_history=history_text,
                current_program=rc.code or "",
                pass_exe=bool(rc.pass_exe),
                speedup=float(rc.latency or 0.0),
                err_msg=err,
                temperature=temperature,
            )
            rc.reflections = refl_out
            rc.reflection_prompt = refl_prompt
            if i < len(child_dirs):
                _write_text(os.path.join(child_dirs[i], "reflection_prompt.txt"), str(refl_prompt))

        # Update history and perf_candidates (GaAgent-like).
        for i, rc in enumerate(raw_codes):
            history[i].append(rc)
            # Keep the most recent 5 items per slot.
            if len(history[i]) > 5:
                history[i] = history[i][-5:]

            if rc.pass_perf:
                cand = (rc.code, float(rc.latency), 0.0, rc.reflections, rc.profilig)
                if len(perf_candidates) < ancestor_num:
                    perf_candidates.append(cand)
                else:
                    # Replace worst if better.
                    perf_candidates.sort(key=lambda x: x[1], reverse=True)
                    if perf_candidates[-1][1] <= cand[1]:
                        perf_candidates[-1] = cand
                perf_candidates.sort(key=lambda x: x[1], reverse=True)

        # Select best solution.
        if perf_candidates:
            best_solution_code = perf_candidates[0][0]
            best_solution_speedup = float(perf_candidates[0][1])
            best_solution_profiling = perf_candidates[0][4]

        # Dump round summary.
        with open(os.path.join(round_dir, "round_summary.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "round": r,
                    "offspring": offspring_rows,
                    "round_pass_exe": bool(round_pass_exe),
                    "perf_debug_num": perf_debug_num,
                    "best_speedup": best_solution_speedup,
                    "num_perf_candidates": len(perf_candidates),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        # When correctness passes for enough offspring, start a fresh batch next round.
        if round_pass_exe:
            raw_codes = None

    # Final output
    out = {
        "op": op_name,
        "best_speedup": best_solution_speedup,
        "best_code": best_solution_code,
        "best_profiling": best_solution_profiling,
        "perf_candidates": [
            {"speedup": c[1], "has_reflection": bool(c[3]), "has_profiling": bool(c[4])} for c in perf_candidates
        ],
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
