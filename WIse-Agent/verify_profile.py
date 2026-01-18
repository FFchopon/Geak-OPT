
import json
import os
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
from geak_eval.constants import Names
from geak_eval.evaluators.interface import TestAllCloseEvaluatorTBG
from geak_eval.helpers.helper import process_code


GEN_OP_PATH = r"./operator_4_test/dequantize_rowwise.py"
REF_OP_PATH = r"/workspace/zibo/Geak-OPT/GEAK-eval/geak_eval/data/TritonBench/data/TritonBench_G_v1/dequantize_rowwise.py"

ATOL = 1e-3
RTOL = 1e-3
TIMEOUT_S = 2 * 60
WORK_DIR = None

PROFILE_OUT_DIR = r"./profile_result"
REPORT_NAME = "verify_profile"
NCU_SET = "full"


def _run(cmd, cwd=None, env=None):
    p = subprocess.run(cmd, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return p.returncode, p.stdout, p.stderr


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
        f.write("\n")
        f.write("_orig_run_benchmark = _pu.Performance_Metrics.run_benchmark\n")
        f.write("\n")
        f.write("def _run_benchmark_first_only(self):\n")
        f.write("    if getattr(self, 'input_tensors', None):\n")
        f.write("        self.input_tensors = self.input_tensors[:1]\n")
        f.write("    return _orig_run_benchmark(self)\n")
        f.write("\n")
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


def main():
    gen_op = os.path.abspath(GEN_OP_PATH)
    ref_op = os.path.abspath(REF_OP_PATH)

    if not os.path.isfile(gen_op):
        raise FileNotFoundError(f"GEN_OP_PATH not found: {gen_op}")
    if not os.path.isfile(ref_op):
        raise FileNotFoundError(f"REF_OP_PATH not found: {ref_op}")

    with open(gen_op, "r", encoding="utf-8") as f:
        code_raw = f.read()

    fname = os.path.basename(ref_op)
    op_name = os.path.splitext(fname)[0]
    gt_root = os.path.dirname(ref_op)

    perf_script = os.path.join(NATIVE_PERF_GOLD_ROOT, f"{op_name}_perf.py")
    if not os.path.isfile(perf_script):
        raise FileNotFoundError(f"Perf script not found: {perf_script}")

    work_dir = os.path.abspath(WORK_DIR) if WORK_DIR else tempfile.mkdtemp(prefix="verify_profile_")
    log_root = os.path.join(work_dir, "log")
    exec_root = os.path.join(work_dir, "exec")
    os.makedirs(log_root, exist_ok=True)
    os.makedirs(exec_root, exist_ok=True)

    evaluator = TestAllCloseEvaluatorTBG(ground_truth_root=gt_root)

    gen_file = evaluator.get_gen_fpath(log_root, fname)
    tests = evaluator.get_tests_code(ref_op)
    code = process_code(code_raw)
    code = evaluator.format_gen_code(gen_file, code, tests)

    compile_ok, compile_stdout, compile_stderr = evaluator._call_file(gen_file, timeout=TIMEOUT_S)

    diff_ok = False
    diff_stdout_raw = None
    diff_stderr_raw = None
    diff_call_ok = None
    diff_exec_ok = None
    diff_stdout = None
    diff_stderr = None
    if compile_ok:
        diff_ok, diff_stdout_raw, diff_stderr_raw = evaluator._check_match(
            gen_fpath=gen_file,
            ref_fpath=ref_op,
            atol=ATOL,
            rtol=RTOL,
            timeout=TIMEOUT_S,
        )

        try:
            parts = str(diff_stdout_raw).split(Names.RET_SEPERATOR)
            if len(parts) >= 4:
                diff_call_ok = parts[0].strip().lower() == str(True).lower()
                diff_exec_ok = parts[1].strip().lower() == str(True).lower()
                diff_stdout = parts[2]
                diff_stderr = parts[3]
        except Exception:
            pass

    ncu_rep = None
    ncu_txt = None
    if diff_ok:
        out_dir = os.path.abspath(PROFILE_OUT_DIR)
        os.makedirs(out_dir, exist_ok=True)

        report_base = os.path.join(out_dir, REPORT_NAME)
        rep_path = report_base + ".ncu-rep"
        txt_path = report_base + ".txt"

        rc, _, err = _run(["ncu", "--version"])
        if rc != 0:
            raise RuntimeError(f"ncu not available in PATH. stderr: {err}")

        runner_path = os.path.join(out_dir, f"{REPORT_NAME}__runner.py")
        _write_ncu_runner(runner_path)

        cmd = [
            "ncu",
            "--set",
            NCU_SET,
            "-o",
            report_base,
            sys.executable,
            runner_path,
            gen_op,
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

        table = _extract_first_gpu_sol_table(out)
        if table is None:
            table = out

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(table)

        ncu_rep = rep_path
        ncu_txt = txt_path

        if os.path.exists(runner_path):
            try:
                os.remove(runner_path)
            except Exception:
                pass

    result = {
        "gen_op": gen_op,
        "ref_op": ref_op,
        "work_dir": work_dir,
        "compile_ok": bool(compile_ok),
        "compile_stdout": compile_stdout,
        "compile_stderr": compile_stderr,
        "diff_ok": bool(diff_ok),
        "diff_call_ok": diff_call_ok,
        "diff_exec_ok": diff_exec_ok,
        "diff_stdout": diff_stdout,
        "diff_stderr": diff_stderr,
        "diff_stdout_raw": diff_stdout_raw,
        "diff_stderr_raw": diff_stderr_raw,
        "ncu_rep": ncu_rep,
        "ncu_txt": ncu_txt,
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
