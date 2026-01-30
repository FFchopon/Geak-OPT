import os
import subprocess
import sys


def _add_repo_paths():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(this_dir, "..", ".."))
    geak_eval_root = os.path.join(repo_root, "GEAK-eval")

    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    if geak_eval_root not in sys.path:
        sys.path.insert(0, geak_eval_root)


_add_repo_paths()


GEN_OP_PATH = r"./operator_4_test/dequantize_rowwise.py"
OP_NAME = "dequantize_rowwise"
OUT_DIR = r"./profile_result"
REPORT_NAME = "ncu_report"
NCU_SET = "full"
PYTHON_BIN = sys.executable
PERF_SCRIPT_PATH = None


def _run(cmd, cwd=None, env=None):
    p = subprocess.run(cmd, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return p.returncode, p.stdout, p.stderr


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

def main():
    gen_op_path = os.path.abspath(GEN_OP_PATH)
    out_dir = os.path.abspath(OUT_DIR)

    if not os.path.isfile(gen_op_path):
        raise FileNotFoundError(f"GEN_OP_PATH not found: {gen_op_path}")

    os.makedirs(out_dir, exist_ok=True)

    report_base = os.path.join(out_dir, REPORT_NAME)
    rep_path = report_base + ".ncu-rep"
    txt_path = report_base + ".txt"

    from geak_eval.constants import NATIVE_PERF_GOLD_ROOT

    perf_script_path = os.path.abspath(PERF_SCRIPT_PATH) if PERF_SCRIPT_PATH else os.path.join(NATIVE_PERF_GOLD_ROOT, f"{OP_NAME}_perf.py")
    if not os.path.isfile(perf_script_path):
        raise FileNotFoundError(f"Perf script not found: {perf_script_path}")

    runner_path = os.path.join(out_dir, f"{REPORT_NAME}__runner.py")
    with open(runner_path, "w", encoding="utf-8") as f:
        f.write("import importlib.util\n")
        f.write("import os\n")
        f.write("import runpy\n")
        f.write("import sys\n")
        f.write("\n")
        f.write("this_dir = os.path.dirname(os.path.abspath(__file__))\n")
        f.write("repo_root = os.path.abspath(os.path.join(this_dir, '..', '..', '..'))\n")
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

    code, _, err = _run(["ncu", "--version"])
    if code != 0:
        raise RuntimeError(f"ncu not available in PATH. stderr: {err}")

    cmd_profile = [
        "ncu",
        "--set",
        NCU_SET,
        "-o",
        report_base,
        PYTHON_BIN,
        runner_path,
        gen_op_path,
        OP_NAME,
        perf_script_path,
    ]

    code, out, err = _run(cmd_profile)
    if code != 0:
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
            raise FileNotFoundError(
                f"Expected report not found: {rep_path}. "
                f"ncu stdout: {out}\n"
                f"ncu stderr: {err}\n"
                f"Hint: the target script must actually execute a GPU kernel; for pure-definition modules set ENTRY_FUNC to a function that runs the op." 
            )

    code, out, err = _run(["ncu", "--import", rep_path])
    if code != 0:
        raise RuntimeError(f"ncu --import failed. stdout: {out}\nstderr: {err}")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(_truncate_first_operator_profile(out) or out)

    print(f"Saved: {rep_path}")
    print(f"Saved: {txt_path}")

    if os.path.exists(runner_path):
        try:
            os.remove(runner_path)
        except Exception:
            pass


if __name__ == "__main__":
    main()
