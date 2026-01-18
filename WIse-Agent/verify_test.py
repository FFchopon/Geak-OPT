import json
import os
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

from geak_eval.evaluators.interface import TestAllCloseEvaluatorTBG
from geak_eval.constants import NATIVE_PERF_GOLD_ROOT

GEN_OP_PATH = r"./operator_4_test/gen.py"
REF_OP_PATH = r"/workspace/zibo/Geak-OPT/GEAK-eval/geak_eval/data/TritonBench/data/TritonBench_G_v1/dequantize_rowwise.py"

ATOL = 1e-3
RTOL = 1e-3
TIMEOUT_S = 2 * 60
VERBOSE = False
WORK_DIR = None


def main():
    gen_op = os.path.abspath(GEN_OP_PATH)
    ref_op = os.path.abspath(REF_OP_PATH)

    if not os.path.isfile(gen_op):
        raise FileNotFoundError(f"gen_op not found: {gen_op}")
    if not os.path.isfile(ref_op):
        raise FileNotFoundError(f"ref_op not found: {ref_op}")

    with open(gen_op, "r", encoding="utf-8") as f:
        code = f.read()

    fname = os.path.basename(ref_op)
    gt_root = os.path.dirname(ref_op)

    perf_fname = os.path.join(NATIVE_PERF_GOLD_ROOT, fname.replace(".py", "_perf.py"))
    if not os.path.exists(perf_fname):
        raise FileNotFoundError(
            f"Expected perf script not found: {perf_fname}. "
            f"REF_OP_PATH must point to a real TritonBench operator file (so its *_perf.py exists under golden_metrics)."
        )

    work_dir = os.path.abspath(WORK_DIR) if WORK_DIR else tempfile.mkdtemp(prefix="verify_test_")
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
        verbose=VERBOSE,
        custom_tests_path=None,
    )

    result = {
        "gen_op": gen_op,
        "ref_op": ref_op,
        "work_dir": work_dir,
        "compile_ok": bool(call_status),
        "diff_ok": bool(exec_status),
        "speedup": speedup,
        "stdout": stdout,
        "stderr": stderr,
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))

    if WORK_DIR is None:
        try:
            import shutil

            shutil.rmtree(work_dir, ignore_errors=True)
        except Exception:
            pass


if __name__ == "__main__":
    main()
