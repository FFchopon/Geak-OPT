from tqdm import tqdm
import os
import json
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from agents.reflexion_oneshot import Reflexion_Oneshot
from utils.utils import clear_code, extract_function_signatures, infer_function_signatures_from_test_code, clear_json
from memories.Memory import MemoryClassMeta
from prompts import prompt_for_generation, prompt_for_reflection
from loguru import logger
from tenacity import RetryError
from dataloaders.ProblemState import tempCode
from typing import List, Optional

class GaAgent(Reflexion_Oneshot):
    def __init__(self, model, dataset, corpus_path, max_perf_debug_num=5, mem_file=None, descendant_num=1):
        super().__init__(model, dataset, corpus_path, mem_file, descendant_num)
        self.max_perf_debug_num = max_perf_debug_num

    def _prompt_dump_enabled(self) -> bool:
        return str(os.environ.get("GEAK_DUMP_PROMPTS", "0")).lower() in {"1", "true", "yes", "y"}

    def memory_init(self, mem_file=None, descendant_num=1):
        """
        Args:
            mem_file: previous stored memories, which can be loaded to continue run
        """
        class Memory(metaclass=MemoryClassMeta, field_names=["ps", 
                                                             "call_err_msg", 
                                                             "exe_err_msg",
                                                             "reflection", 
                                                             "function_signatures", 
                                                             "oneshot", 
                                                             "perf_candidates",
                                                             "perf_strategy",
                                                             "raw_codes",
                                                             "call_candidate",
                                                             "exe_candidate",
                                                             "temp_strategy",
                                                             "perf_debug_num",
                                                             "pass_call", 
                                                             "pass_exe",
                                                             "pass_perf",
                                                             "offspring_summary",
                                                             "history"]):
            pass
        
        if mem_file is not None:
            assert mem_file.endswith(".json"), f"expect a json file, but got {mem_file} instead"
            with open(mem_file, "r") as f:
                input_mems = json.load(f)
            assert len(input_mems) == len(self.dataset), f"expect {len(self.dataset)} samples, but got {len(input_mems)} instead"

        for ps in self.dataset.problem_states:

            if ps.label:
                fs_mem = extract_function_signatures(ps.label)
            else:
                fs_mem = None

            if (not fs_mem) and getattr(ps, "test_code", None):
                fs_mem = infer_function_signatures_from_test_code(ps.test_code)
            elif fs_mem and getattr(ps, "test_code", None):
                inferred = infer_function_signatures_from_test_code(ps.test_code)
                if inferred:
                    existing_names = set()
                    for s in fs_mem:
                        try:
                            existing_names.add(s.split("def ", 1)[1].split("(", 1)[0].strip())
                        except Exception:
                            continue
                    for s in inferred:
                        try:
                            n = s.split("def ", 1)[1].split("(", 1)[0].strip()
                        except Exception:
                            n = None
                        if n and n not in existing_names:
                            fs_mem.append(s)
                            existing_names.add(n)
            raw_codes =None
            if mem_file is None:
                os_mem = self.instruction_retriever.query(ps.instruction)[0]
                tmp_mem = Memory(ps=ps, 
                                call_err_msg=None,
                                exe_err_msg=None, 
                                reflection=None, 
                                function_signatures=fs_mem, 
                                oneshot=os_mem["code"], 
                                perf_candidates=[],
                                perf_strategy=None,
                                raw_codes=raw_codes,
                                call_candidate=None,
                                exe_candidate=None,
                                temp_strategy=None,
                                perf_debug_num=0,
                                pass_call=False,
                                pass_exe=False,
                                pass_perf=False,
                                offspring_summary=None,
                                history=[[] for _ in range(descendant_num)]
                                )
            else:
                input_mem = input_mems[ps.filename]
                tmp_mem = Memory(
                    ps=ps,
                    call_err_msg=input_mem["call_err_msg"],
                    exe_err_msg=input_mem["exe_err_msg"], 
                    reflection=input_mem["reflection"], 
                    function_signatures=fs_mem, 
                    oneshot=input_mem["oneshot"], 
                    perf_candidates=input_mem["perf_candidates"],
                    perf_strategy=input_mem["perf_strategy"],
                    raw_codes=raw_codes,
                    call_candidate=input_mem["call_candidate"],
                    exe_candidate=input_mem["exe_candidate"],
                    temp_strategy=input_mem["temp_strategy"],
                    perf_debug_num=input_mem["perf_debug_num"],
                    pass_call=input_mem["pass_call"],
                    pass_exe=input_mem["pass_exe"],
                    pass_perf=input_mem["pass_perf"],
                    offspring_summary=input_mem.get("offspring_summary"),
                    history=[[] for _ in range(descendant_num)]
                )

            self.memories.append(tmp_mem)
    
    def write_memories(self, file_path):
        output_dict = {}
        with open(file_path, "w") as f:
            for mem in self.memories:
                output = {
                    "call_err_msg": str(mem.call_err_msg),
                    "exe_err_msg": str(mem.exe_err_msg),
                    "reflection": mem.reflection, 
                    "oneshot": mem.oneshot, 
                    "perf_candidates": [list(cand) for cand in mem.perf_candidates],
                    "perf_strategy": mem.perf_strategy,
                    "call_candidate": mem.call_candidate,
                    "exe_candidate": mem.exe_candidate,
                    "temp_strategy": mem.temp_strategy,
                    "perf_debug_num": mem.perf_debug_num,
                    "pass_call": mem.pass_call, 
                    "pass_exe": mem.pass_exe,
                    "pass_perf": mem.pass_perf,
                    "offspring_summary": mem.offspring_summary,
                }
                output_dict[mem.ps.filename] = output
            json.dump(output_dict, f)
    
    def _sanitize_path_component(self, s: str) -> str:
        s = str(s)
        s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s)
        return s[:180] if len(s) > 180 else s

    def _normalize_function_signatures(self, function_signatures) -> str:
        """Normalize signature display for prompts.
        Goal: one signature per line, stable 'def name(args)' form.
        """
        if not function_signatures:
            return ""

        lines = []
        for sig in function_signatures:
            if sig is None:
                continue
            s = str(sig).strip()
            if not s:
                continue
            if s.startswith("*"):
                s = s.lstrip("* ").strip()
            # Collapse multiline signatures into one line.
            s = " ".join(s.split())
            # Extract canonical `def name(args)` if possible.
            m = re.search(r"def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\)", s)
            if m:
                name = m.group(1)
                args = m.group(2).strip()
                # Remove trailing comma that can be introduced by noisy extraction.
                args = re.sub(r",\s*$", "", args)
                s = f"def {name}({args})"
            lines.append(s)

        out = "\n".join(f"* {ln}" for ln in lines)
        return out + ("\n" if out else "")

    def _build_example_snippet(self, mem) -> str:
        """Optionally include a small example snippet.
        Default: disabled to avoid injecting unrelated long code.
        """
        use_example = str(os.environ.get("GEAK_USE_EXAMPLE_SNIPPET", "0")).lower() in {"1", "true", "yes", "y"}
        if not use_example:
            return ""
        try:
            max_chars = int(os.environ.get("GEAK_EXAMPLE_SNIPPET_CHARS", "800"))
        except Exception:
            max_chars = 800

        snippet = None
        if not getattr(mem, "exe_candidate", None) and not getattr(mem, "call_candidate", None) and not getattr(mem, "raw_codes", None):
            snippet = getattr(mem, "oneshot", None)
        elif getattr(mem, "raw_codes", None):
            try:
                snippet = self.code_retriever.query(mem.raw_codes[0].code)[0]["code"]
            except Exception:
                snippet = None

        if not snippet:
            return ""
        s = str(snippet)
        if max_chars > 0 and len(s) > max_chars:
            s = s[:max_chars] + "\n...<truncated>...\n"
        return f"\nHere is an example snippet of code: {s}"

    def _compact_error_text(self, err: str) -> str:
        """Make reflection error text shorter and more informative.
        - De-duplicate repeated blocks (common when runnable/correctness share same stack)
        - Keep only tail lines where the exception and triton compile location appear
        """
        if not err:
            return ""
        s = str(err)
        # De-duplicate identical halves (very common in current formatting)
        mid = len(s) // 2
        if len(s) > 2000 and s[:mid].strip() == s[mid:].strip():
            s = s[:mid]
        # Keep tail lines for key error.
        try:
            keep_lines = int(os.environ.get("GEAK_REFLECTION_ERR_TAIL_LINES", "120"))
        except Exception:
            keep_lines = 120
        lines = s.splitlines()
        if keep_lines > 0 and len(lines) > keep_lines:
            lines = lines[-keep_lines:]
            s = "...<snip>...\n" + "\n".join(lines)
        return s

    def _dump_prompt(self, stage: str, mem, prompt: str, offspring_idx: Optional[int] = None):
        if not self._prompt_dump_enabled():
            return
        dump_dir = getattr(self, "_prompt_dump_dir", None)
        if not dump_dir:
            return

        try:
            console_chars = int(os.environ.get("GEAK_PROMPT_CONSOLE_CHARS", "2000"))
        except Exception:
            console_chars = 2000

        fn = getattr(getattr(mem, "ps", None), "filename", "unknown")
        safe_fn = self._sanitize_path_component(fn)
        ts = int(time.time() * 1000)
        off = f"_off{offspring_idx}" if offspring_idx is not None else ""
        fpath = os.path.join(dump_dir, f"{stage}_{safe_fn}{off}_{ts}.txt")

        meta = {
            "stage": stage,
            "filename": fn,
            "offspring_idx": offspring_idx,
            "timestamp_ms": ts,
        }
        try:
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(json.dumps(meta, ensure_ascii=False))
                f.write("\n\n")
                f.write(prompt)
        except Exception as e:
            logger.warning(f"failed to dump prompt to {fpath}: {e}")

        if console_chars and console_chars > 0:
            p = prompt
            if len(p) > console_chars:
                p = p[:console_chars] + "\n...<truncated>...\n"
            logger.info(f"[{stage}] prompt for {fn}{off}:\n{p}")
    
    def _log_iteration_summary(self, iter_idx: int, start_idx: int, data_len: int, profiling: bool):
        try:
            truncate_chars = int(os.environ.get("GEAK_LOG_TRUNCATE_CHARS", "0"))
        except Exception:
            truncate_chars = 0

        for mem in self.memories[start_idx:(start_idx + data_len)]:

            filename = getattr(mem.ps, "filename", "<unknown>")
            runnable = bool(getattr(mem, "pass_call", False))
            correctness = bool(getattr(mem, "pass_exe", False))

            best_speedup = None
            best_profiling = None
            if getattr(mem, "perf_candidates", None):
                try:
                    best_speedup = mem.perf_candidates[0][1]
                    best_profiling = mem.perf_candidates[0][4] if len(mem.perf_candidates[0]) > 4 else None
                except Exception:
                    best_speedup = None
                    best_profiling = None
            elif getattr(mem, "raw_codes", None):
                try:
                    rc0 = mem.raw_codes[0]
                    best_speedup = getattr(rc0, "latency", None)
                    best_profiling = getattr(rc0, "profilig", None)
                except Exception:
                    best_speedup = None
                    best_profiling = None

            msg = f"[Iter {iter_idx}] {filename} | runnable={runnable} | correctness={correctness}"
            if best_speedup is not None:
                msg += f" | speedup={best_speedup}"
            if profiling and best_profiling:
                snippet = str(best_profiling)
                if truncate_chars and len(snippet) > truncate_chars:
                    snippet = snippet[:truncate_chars] + "\n...<truncated>...\n"
                msg += f"\nprofiling:\n{snippet}"

            # Show per-offspring status (multi-offspring runs can otherwise look contradictory).
            off = getattr(mem, "offspring_summary", None)
            if off:
                off_s = str(off)
                if truncate_chars and len(off_s) > truncate_chars:
                    off_s = off_s[:truncate_chars] + "\n...<truncated>...\n"
                msg += f"\noffspring:\n{off_s}"
            # When correctness passes but speedup is 0, perf likely failed, did not emit speedup,
            # or speedup was computed as 0. Show perf-related stderr to avoid a silent 0.
            if correctness and (best_speedup == 0 or best_speedup == 0.0):
                perf_note = getattr(mem, "exe_err_msg", None)
                if perf_note and str(perf_note).strip() and str(perf_note).strip().lower() != "none":
                    note_s = str(perf_note)
                    if truncate_chars and len(note_s) > truncate_chars:
                        note_s = note_s[:truncate_chars] + "\n...<truncated>...\n"
                    label = "perf_warning" if "[GEAK-EVAL] speedup unavailable" in note_s else "perf_debug"
                    msg += f"\n{label}:\n{note_s}"
            if not correctness:
                err = getattr(mem, "exe_err_msg", None) or getattr(mem, "call_err_msg", None)
                if err:
                    err_s = str(err)
                    if truncate_chars and len(err_s) > truncate_chars:
                        err_s = err_s[:truncate_chars] + "\n...<truncated>...\n"
                    msg += f"\nerror:\n{err_s}"
            logger.info(msg)
    
    def run(self, output_path=None, multi_thread=True, datalen=None, iteration_num=0, temperature=0, ancestor_num=5, descendant_num=1, mutation=False, start_idx=0, gpu_id=0, start_iter=0, descendant_debug=1, target_gpu='MI250', profiling=False):
        """
        Args:
            output_path: the folder to store the final result
            multi_thread: whether use multithreading for generating
            datalen: for debug, to specify how many data from the dataset you want to use
            iteration_num: how many iterations you want to run
            temperature: LLM temperature
            ancestor_num: how many samples you want to add in the prompt when optimize the code
            descendant_num: how many codes you want to generate in one try
            start_idx: start idx of the data rows
            gpu_id: which gpu you want to use when you test the scripts
            start_iter: which iteration you want to start with. useful when you load previous result and memory
        """
        assert ancestor_num >= 0, f"expect ancestor_num to be larger than 0, bug got {ancestor_num}"
        assert descendant_num >= 0, f"expect descendant_num to be larger than 0, bug got {descendant_num}"
        assert descendant_debug >= 0, f"expect descendant_debug to be larger than 0, bug got {descendant_debug}"
        data_len = datalen if datalen else len(self.dataset)
        end_idx = start_idx + data_len
        if start_idx < 0:
            start_idx = 0
        if end_idx < start_idx:
            end_idx = start_idx
        if not self.memories[start_idx:end_idx]:
            logger.warning(f"No memories selected for run: start_idx={start_idx}, data_len={data_len}, end_idx={end_idx}, total_memories={len(self.memories)}")
        for iter in range(start_iter, iteration_num):
            logger.info(f"\n=== Iteration {iter} ===")
            if output_path is not None:
                root, extension = os.path.splitext(output_path)
                iter_path = f"{root}_{iter}{extension}"
                mem_output_path = f"{root}_mem_{iter}.json"
            if self._prompt_dump_enabled():
                if output_path is not None:
                    root, _ = os.path.splitext(output_path)
                    self._prompt_dump_dir = os.path.abspath(f"{root}_prompts_{iter}")
                else:
                    self._prompt_dump_dir = os.path.abspath(f"prompts_{iter}")
                os.makedirs(self._prompt_dump_dir, exist_ok=True)
            else:
                self._prompt_dump_dir = None

            if multi_thread:
                thread_num = 3
            # generate solution
            logger.info(f"\ngenerate solution")
            with tqdm(total=data_len) as pbar:
                if multi_thread:
                    
                    with ThreadPoolExecutor(max_workers=thread_num) as executor:
                        futures = {executor.submit(self.generate_solution, mem, temperature, descendant_num, mutation): mem for mem in self.memories[start_idx:end_idx]}
                        for future in as_completed(futures):
                            pbar.update(1)
                else:
                    for mem in self.memories[start_idx:end_idx]:
                        self.generate_solution(mem, temperature=temperature, descendant_num=descendant_num, mutation=mutation)
                        pbar.update(1)
            
            # generate reflections
            logger.info(f"\ngenerate LLM evaluation")
            with tqdm(total=data_len) as pbar:
                if multi_thread:
                    with ThreadPoolExecutor(max_workers=thread_num) as executor:
                        futures = {executor.submit(self.generate_llm_evaluate, mem, temperature): mem for mem in self.memories[start_idx:end_idx]}
                        for future in as_completed(futures):
                            pbar.update(1)
                else:
                    for mem in self.memories[start_idx:end_idx]:
                        self.generate_llm_evaluate(mem, temperature=temperature)
                        pbar.update(1)
            
            # run scripts
            logger.info(f"\nrun scripts on gpu")
            if output_path is not None:
                root, extension = os.path.splitext(output_path)
                tmp_dir = os.path.abspath(f"{root}_tmp")
                exe_dir = os.path.abspath(f"{root}_pass_exe")
                perf_result_dir = os.path.abspath(f"{root}_perf_results")
                perf_log_dir = os.path.abspath(f"{root}_perf_logs")

            else:
                tmp_dir = "tmp"
                exe_dir = "pass_exe"
                perf_result_dir = "perf_results"
                perf_log_dir = "perf_logs"
            os.makedirs(tmp_dir, exist_ok=True)
            os.makedirs(exe_dir, exist_ok=True)
            os.makedirs(perf_result_dir, exist_ok=True)
            os.makedirs(perf_log_dir, exist_ok=True)
            for mem in tqdm(self.memories[start_idx:end_idx]):
                if mem.raw_codes:
                    offspring_rows = []
                    for i in range(len(mem.raw_codes)):
                        raw_code = mem.raw_codes[i]
                        speedup = 0.0
                        if raw_code.pass_perf:
                            continue
                        try:
                            if raw_code.code :
                                pass_call, pass_exe, speedup, stdout, stderr = self.dataset.test_opt_correctness(raw_code.code, filename=mem.ps.filename, tmp_dir=tmp_dir, exe_dir=exe_dir, gpu_id=gpu_id)
                            else:
                                pass_call, pass_exe, speedup, stdout, stderr = False, False, 0.0, "", "Code is empty"
                        except Exception as e:
                            print(f"failed to test the code for {mem.ps.filename}")
                            raw_code.test_stdout = f"failed to test the code due to: {e}"
                            raw_code.test_stderr = f"failed to test the code due to: {e}"
                            continue

                        # Always persist evaluator outputs for debugging (stdout/stderr come from perf stage in TBG evaluator).
                        raw_code.test_stdout = stdout
                        raw_code.test_stderr = stderr

                        if not pass_call:
                            raw_code.profilig = None
                        elif pass_call and not pass_exe:
                            raw_code.pass_call = True
                            raw_code.test_stderr = stderr if stderr else stdout
                            mem.call_candidate = raw_code.code
                            mem.temp_strategy = raw_code.strategy
                            mem.pass_call = True
                            raw_code.profilig= None
                        else:
                            raw_code.pass_call = True
                            raw_code.pass_exe = True
                            mem.pass_call = True
                            mem.exe_candidate = raw_code.code
                            mem.call_candidate = raw_code.code
                            mem.temp_strategy = raw_code.strategy
                            if profiling:
                                pass_prfiler, stdout_profile, stderr_profile, stdout_analyze = self.dataset.test_kernel_profiling(raw_code.code, mem.ps.filename, tmp_dir, exe_dir=exe_dir, target_gpu=target_gpu, timeout=30*60, gpu_id=gpu_id)
                                raw_code.profilig = stdout_analyze
                        mem.call_err_msg = raw_code.test_stdout
                        # Preserve any existing perf note across offspring to avoid losing it when later offspring overwrites stderr.
                        prev_exe_err = getattr(mem, "exe_err_msg", None)
                        new_exe_err = raw_code.test_stderr
                        if prev_exe_err and "[GEAK-EVAL] speedup unavailable" in str(prev_exe_err) and (not new_exe_err or "[GEAK-EVAL] speedup unavailable" not in str(new_exe_err)):
                            mem.exe_err_msg = prev_exe_err
                        else:
                            mem.exe_err_msg = new_exe_err
                        # If perf ran but did not provide speedup, evaluator now appends an explanatory note to stderr.
                        # Surface it in logs by keeping it on mem.exe_err_msg even when correctness passes.
                        if pass_exe and (speedup == 0.0) and raw_code.test_stderr and "[GEAK-EVAL]" in str(raw_code.test_stderr):
                            mem.exe_err_msg = raw_code.test_stderr

                        if speedup > 0.0 and pass_exe:
                            raw_code.pass_perf = True
                            mem.pass_perf = True
                            raw_code.latency = speedup
                            raw_code.eff = 0.0
                        else:
                            # Always record measured speedup (including 0) for logging/debugging.
                            raw_code.latency = speedup

                        try:
                            offspring_rows.append({
                                "i": i,
                                "pass_call": bool(getattr(raw_code, "pass_call", False)),
                                "pass_exe": bool(getattr(raw_code, "pass_exe", False)),
                                "speedup": float(getattr(raw_code, "latency", 0.0) or 0.0),
                            })
                        except Exception:
                            offspring_rows.append({"i": i})
                    descendant_debug = min(descendant_debug, len(mem.raw_codes))
                    if sum(rc.pass_exe for rc in mem.raw_codes) >= descendant_debug:
                        mem.pass_exe = True
                    mem.offspring_summary = offspring_rows

            # generate reflections
            logger.info(f"\ngenerate reflections")
            with tqdm(total=data_len) as pbar:
                if multi_thread:
                    
                    with ThreadPoolExecutor(max_workers=thread_num) as executor:
                        futures = {executor.submit(self.generate_reflexion, mem, temperature): mem for mem in self.memories[start_idx:end_idx]}
                        for future in as_completed(futures):
                            pbar.update(1)
                else:
                    for mem in self.memories[start_idx:end_idx]:
                        self.generate_reflexion(mem, temperature=temperature)
                        pbar.update(1)

            # update perf_candidates

            for mem in self.memories[start_idx:end_idx]:
                if mem.raw_codes:
                    for i in range(len(mem.raw_codes)):
                        raw_code = mem.raw_codes[i]
                        mem.history[i].append(raw_code)
                        codes_sorted = sorted(mem.history[i], key=lambda x: x.llm_metric, reverse=True)
                        mem.history[i] = codes_sorted[:5]
                        if raw_code.pass_perf and raw_code.strategy:
                            raw_code.strategy = None
                            self.update_perf_candidates(mem=mem, raw_code=raw_code, ancestor_num=ancestor_num)
                if len(mem.perf_candidates) > 0:
                    mem.ps.solution = mem.perf_candidates[0][0]
                    mem.ps.speedup = mem.perf_candidates[0][1]
                elif mem.exe_candidate:
                    mem.ps.solution = mem.exe_candidate
                elif mem.call_candidate:
                    mem.ps.solution = mem.call_candidate
                elif mem.raw_codes:
                    mem.ps.solution = mem.raw_codes[0].code

            logger.info(f"\niteration summary")
            self._log_iteration_summary(iter_idx=iter, start_idx=start_idx, data_len=data_len, profiling=profiling)

            if output_path is not None:
                self.dataset.write_file(iter_path)
                self.write_memories(mem_output_path)

            os.system(f'rm -rf {exe_dir}')
            os.system(f'rm -rf {perf_result_dir}')
            os.system(f'rm -rf {perf_log_dir}')
            os.system(f'rm -rf {tmp_dir}')
    
    def generate_solution(self, mem, temperature=0, descendant_num=1, mutation=False):

        tab = "\n"
        fss_text = self._normalize_function_signatures(getattr(mem, "function_signatures", None))

        use_reference = str(os.environ.get("GEAK_USE_REFERENCE", "0")).lower() in {"1", "true", "yes", "y"}
        reference_code = getattr(mem.ps, "reference_code", "") or ""
        if use_reference and reference_code:
            reference_section = (
                "**REFERENCE IMPLEMENTATION (STRICTLY FOLLOW):**\n"
                "The following is the reference implementation from the dataset (may be AMD-maintained). "
                "You MUST strictly follow its API usage patterns and overall structure. "
                "You MUST NOT invent new public APIs or helper names beyond the required function signatures above.\n"
                "```python\n"
                f"{reference_code}\n"
                "```"
            )
        else:
            reference_section = ""
        text = prompt_for_generation.prompt.format(
            instruction=mem.ps.instruction,
            function_signatures=fss_text,
            reference_section=reference_section,
        )
        
        # for the one that has perf_candidates, and the code generated in this round pass_exe, we need to generate a new code
        # for the one that has perf_candidates, but the code generated in this round not pass_exe, if the debug_num has exceeds the man_debug_num, then generate a new code
        # otherwise, go to debug
        if (mem.perf_debug_num >= self.max_perf_debug_num) or mem.pass_exe:
            mem.perf_debug_num = 0
            mem.raw_codes =None
        if len(mem.perf_candidates) > 0 and not mem.raw_codes:
            text += """\nThere are some Optimized codes(NO.1, NO.2 and so on) to solve the Problem. The Optimized codes are arranged in ascending order based on their performance, where higher speedup indicates better performance. According to their performance(speedup is the latency compared with golden reference code) and the corresponding analysis, you need to generate a new code with better performance. You should maintain code correctness during optimization."""
            text +="\nYou can use optimization strategies such as Memory access efficiency, Hardware resource utilization, IR analysis, Assembly analysis, Kernel occupancy, TorchInductor with Triton tuning knobs and Auto-tunable kernel configurations and environment variables."    
            for i, cand in enumerate(mem.perf_candidates):
                text += f"\n### Reference {i+1}"
                text += f"\nOptimized code: {cand[0]}"
                text += f"\nOptimized speedup: {cand[1]}"
                if cand[3]:
                    text += f"\nStrategy: {cand[3]}"
                if cand[4]:
                    text += f"\nNsight Compute (ncu) profiling result:{cand[4]}"
            if mutation:
                text += "\nGenerate a better strategy completely different from Optimized Implementation. Based on the better strategy generate a better optimization code."
            else:    
                text += "\nAnalyze and compare all optimization strategies based on Optimized Implementation codes and give a better strategy motivated by them. Based on the better strategy generate a better optimization code to get a higher speedup."
        else:
            text += self._build_example_snippet(mem)
        
        if mem.raw_codes :
            for i in range(len(mem.raw_codes)):
                raw_code = mem.raw_codes[i]
                if not raw_code.pass_perf:
                    text_temp = text
                    history_text = self._build_history_prompt(mem.history[i])
                    text_temp += f"\nPrevious attempt implementations:{history_text}"
                    text_temp += prompt_for_generation.system_prompt
                    if raw_code.reflections:
                        raw_code.reflections = None
                    try:
                        raw_code.code, raw_code.strategy = self.call_llm_code(prompt=text_temp, temperature=temperature)
                    except:
                        logger.info(f"failed to call LLM for {mem.ps.filename}")
            mem.perf_debug_num +=1
            return
        
        gens_codes: List[tempCode] = []
        for i in range(descendant_num):
            gen_code = tempCode()
            try:
                text_temp = text
                text_temp += prompt_for_generation.system_prompt
                text_temp += "\nCRITICAL: All Triton kernels (e.g., _fwd_kernel/_bwd_*_kernel) MUST be decorated with @triton.jit and invoked as kernel[grid](...). Do NOT implement them as plain Python functions."
                self._dump_prompt("generate_solution", mem=mem, prompt=text_temp, offspring_idx=i)
                gen_code.code, gen_code.strategy = self.call_llm_code(prompt=text_temp, temperature=temperature)
            except:
                logger.info(f"failed to call LLM for {mem.ps.filename}")
            gens_codes.append(gen_code)
        mem.raw_codes = gens_codes
        mem.pass_exe = False
        mem.pass_call = False
        mem.pass_perf = False
        return
    
    
    def generate_reflexion(self, mem, temperature):
        
        tab = "\n"
        fss_text = self._normalize_function_signatures(getattr(mem, "function_signatures", None))

        m_info = """
- runnable test: test if the code can be successfully executed.
- correctness test: test if the output of the code is correct, i.e. if the code does implement the functionality required in the original problem.
- speedup: measures the total time from kernel launch to completion, reflecting the responsiveness and overhead of executing a single instance of the kernel on the GPU. And compare the time with golden reference code to get speedup.
"""
        
        if mem.raw_codes :
            for i in range(len(mem.raw_codes)):
                raw_code = mem.raw_codes[i]
                if  raw_code.reflections:
                    continue
                history_text = self._build_history_prompt(mem.history[i])
                if raw_code.pass_exe:
                    result_txt = f"""
- runnable test: Succeed
- correctness test: Succeed
- speedup: {raw_code.latency}
"""                 
                    reflect_txt = prompt_for_reflection.prompt_evolve_strategy_optimize.format(
                        instruction=mem.ps.instruction,
                        function_signatures=fss_text,
                        metrics_info=m_info,
                        evolution_history=history_text,
                        current_program=raw_code.code,
                        test_result=result_txt,
                        reflection=raw_code.reflections
                    )
                else:
                    if raw_code.pass_call:
                        result_txt = f"""
- runnable test: Succeed
- correctness test: Failed
Error Message: {self._compact_error_text(raw_code.test_stderr)}
"""
                    else:
                        result_txt = f"""
- runnable test: Failed
- correctness test: Failed
Error Message: {self._compact_error_text(raw_code.test_stderr)}
"""
                    reflect_txt = prompt_for_reflection.prompt_evolve_reflect.format(
                        instruction=mem.ps.instruction,
                        function_signatures=fss_text,
                        metrics_info=m_info,
                        evolution_history=history_text,
                        current_program=raw_code.code,
                        test_result=result_txt,
                        reflection=raw_code.reflections
                    )

                
                reflect_msg = [
                    {
                        "role": "user",
                        "content": reflect_txt
                    }
                ]
                self._dump_prompt("generate_reflexion", mem=mem, prompt=reflect_txt, offspring_idx=i)
                raw_code.reflections = self.model.generate(reflect_msg, temperature=temperature)
    

    
    def call_llm_code(self, prompt, temperature):
        msg = [{"role": "user", "content": prompt}]
        try:
            max_tokens = 8192
            response = self.model.generate(msg, temperature=temperature, max_tokens=max_tokens)
            opti = clear_json(response)
            if isinstance(opti, dict) and ('code' in opti) and ('strategy' in opti):
                code = clear_code(opti['code'])
                strategy = opti['strategy']
                return code, strategy
            # If JSON parsing failed or model returned unexpected schema, raise a helpful error.
            resp_s = response if isinstance(response, str) else str(response)
            resp_s = resp_s.replace("\r\n", "\n")
            if len(resp_s) > 1500:
                resp_s = resp_s[:1500] + "\n...<truncated>...\n"
            raise ValueError(
                f"LLM response is not a valid {{code,strategy}} JSON. clear_json={opti}. Raw response (truncated):\n{resp_s}"
            )
        except Exception as e:
            logger.exception("failed to call LLM")
            raise ValueError(f"failed to call LLM: {type(e).__name__}: {e}")


    def call_llm_reflecion(self, prompt, temperature):
        msg = [{"role": "user", "content": prompt}]
        try:
            max_tokens = 8192
            response = self.model.generate(msg, temperature=temperature, max_tokens=max_tokens)
            opti = clear_json(response)
            if 'reflection' in opti.keys():
                reflection = opti['reflection']
                return reflection 
        except Exception as e:
            logger.exception("failed to call LLM")
            raise ValueError(f"failed to call LLM: {type(e).__name__}: {e}")

    def update_perf_candidates(self, mem, raw_code: tempCode, ancestor_num):
        if len(mem.perf_candidates) < ancestor_num:
            candidate = [raw_code.code, raw_code.latency, raw_code.eff, raw_code.reflections, raw_code.profilig]
            mem.perf_candidates.append(tuple(candidate))
            mem.perf_candidates = sorted(mem.perf_candidates, key=lambda x: x[1], reverse=True)

        elif mem.perf_candidates[-1][1] <= raw_code.latency:
            candidate = [raw_code.code, raw_code.latency, raw_code.eff, raw_code.reflections, raw_code.profilig]
            mem.perf_candidates[-1] = tuple(candidate)
            # order the candidates in ascending order with regard to speedups
            mem.perf_candidates = sorted(mem.perf_candidates, key=lambda x: x[1], reverse=True)

    def _build_history_prompt(self, history):
        text = ""
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
        for i, raw_code in enumerate(history):
            if raw_code.pass_perf:
                test_txt = """
runnable test: Succeed
correctness test: Succeed
speedup: {latency}
"""
                test_txt = test_txt.format(
                    speedup=raw_code.latency
                )
            elif raw_code.pass_exe and not raw_code.pass_perf:
                test_txt = """
runnable test: Succeed
correctness test: Succeed
"""
            elif raw_code.pass_call and not raw_code.pass_exe:
                test_txt = """
runnable test: Succeed
correctness test: {err_msg}
"""
                test_txt = test_txt.format(
                    err_msg=raw_code.test_stderr
                )
            elif not raw_code.pass_call:
                test_txt = """
runnable test: {err_msg}
"""
                test_txt = test_txt.format(
                    err_msg=raw_code.test_stderr
                )
            text += history_template.format(
                attempt_number=i+1,
                code=raw_code.code,
                test_results=test_txt,
                reflection=raw_code.reflections
            )
        
        return text
    
    
    def generate_llm_evaluate(self, mem, temperature=1.0):
        if mem.raw_codes :
            for i in range(len(mem.raw_codes)):
                raw_code = mem.raw_codes[i]
                if not raw_code.pass_perf:
                    text = ""
                    text += prompt_for_generation.llm_evaluate_prompt.format(current_program=raw_code.code)
                    msg = [{"role": "user", "content": text}]
                    try:
                        max_tokens = 8192
                        response = self.model.generate(msg, temperature=temperature, max_tokens=max_tokens)
                        llm_eval = clear_json(response)
                        metric = 0.0
                        for k, v in llm_eval.items():
                            if k == "reasoning":
                                continue
                            if isinstance(v, float) or isinstance(v, int):
                                metric += float(v)
                        raw_code.llm_metric = metric
                        raw_code.llm_eval = llm_eval
                    except:
                        logger.info(f"failed to generate LLM evaluation")
                        raise ValueError("failed to generate LLM evaluation")