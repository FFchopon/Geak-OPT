import ast
import os
import subprocess
import sys
from random import randint
from tqdm import tqdm
from shutil import copyfile
import datetime
import json
from parse_llm_code import extract_code_blocks
import numpy as np
import re

def get_temp_bash_file(prefix='temp_code'):
    # Generate a unique temporary file name
    temp_file_name = f'{prefix}_{randint(999, 999999)}.sh'
    while os.path.exists(temp_file_name):
        temp_file_name = temp_file_name.replace('.sh', f'_{randint(999, 999999)}.sh')
    return temp_file_name

def parse_profiler_content(profile_content):
    delimiter = "--------------------------------------------------------------------------------"
    
    parts = profile_content.split(delimiter)
    
    section_data = {}

    section_pattern = re.compile(r"^\s*(\d+)\..*$", re.MULTILINE)

    for part in parts:
        trimmed_part = part.strip()
        if not trimmed_part:
            continue
            
        match = section_pattern.search(trimmed_part)
        if match:
            section_number = match.group(1)
            full_section_content = delimiter + part
            section_data[section_number] = full_section_content
            
    return section_data

## Implementation from https://arxiv.org/pdf/2107.03374
def passk(n, c, k):
    if n -c < k: return 1.0
    return 1 - np.prod(
        1 - k/ np.arange(
            n-c+1, n+1
        )
    )

def get_time():
    # Get the current time in the format YYYY-MM-DD_HH-MM-SS
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def get_temp_file(prefix='temp_code'):
    # Generate a unique temporary file name
    temp_file_name = f'{prefix}_{randint(999, 999999)}.py'
    while os.path.exists(temp_file_name):
        temp_file_name = temp_file_name.replace('.py', f'_{randint(999, 999999)}.py')
    return temp_file_name

def code_call_exec_success_stdout(code, fname, temp_root="tmp2", tolerance=2, verbose=False):
    # Save the code to a temporary file
    tmp_triton_folder = os.path.join(temp_root, "triton") #f"{temp_root}_triton"
    tmp_gen_folder = os.path.join(temp_root, "gen") #f"{temp_root}_gen"
    os.makedirs(tmp_triton_folder, exist_ok=True)
    os.makedirs(tmp_gen_folder, exist_ok=True)
    

    triton_root = "dataloaders/TB_eval/TritonBench/data/TritonBench_G_v1"
    RAND_FILE = os.path.join(triton_root, "rand_utils.py")

    copyfile(RAND_FILE, os.path.join(tmp_triton_folder, "rand_utils.py"))
    copyfile(RAND_FILE, os.path.join(tmp_gen_folder, "rand_utils.py"))

    gen_file = get_temp_file(prefix=f'{fname}_gen_triton_code')
    triton_file = os.path.join(triton_root, fname)
    temp_triton_file = get_temp_file(prefix=f'{fname}_temp_triton')

    gen_file = os.path.join(tmp_gen_folder, gen_file)
    temp_triton_file = os.path.join(tmp_triton_folder, temp_triton_file)

    IMPORT_STATEMENT = f"""
from rand_utils import torch_rand, torch_randint, torch_randn
import torch
torch.set_printoptions(precision={tolerance},profile='full',sci_mode=False)
"""

    hash_line = "#"*146
    ## from triton_file copy everything after the hash_line into gen_file
    with open(triton_file, 'r') as f:
        lines = f.readlines()
        # lines.append(
        #     '\nprint(result_gold)'
        # )
        for iL, line in enumerate(lines):
            if line.strip() == hash_line:
                break
        test_code_lines = lines[iL+1:]
        test_code_lines = IMPORT_STATEMENT.split('\n') + test_code_lines
        test_code_lines_procs = []
        for line in test_code_lines:
            if "torch.rand" in line:
                line = line.replace("torch.rand", "torch_rand")
            test_code_lines_procs.append(line)

    with open(temp_triton_file, 'w') as f:
        triton_lines = lines[:iL] +  [hash_line] + test_code_lines_procs
        for line in triton_lines:
            f.write(line + "\n")

    code =  code + '\n\n' + hash_line + '\n' + '\n' + '\n'.join(test_code_lines_procs)

    code += "\n\nimport os\n"
    code += "try:\n    import torch\nexcept Exception:\n    torch = None\n"
    code += "if os.environ.get('GEAK_PROFILE_DIAG', '0') in {'1','true','yes','y'}:\n"
    code += "    try:\n"
    code += "        if torch is not None and hasattr(torch, 'cuda') and torch.cuda.is_available():\n"
    code += "            torch.cuda.synchronize()\n"
    code += "            print('GEAK_PROFILE_DIAG: cuda_synchronized')\n"
    code += "        else:\n"
    code += "            print('GEAK_PROFILE_DIAG: cuda_not_available')\n"
    code += "    except Exception as _e:\n"
    code += "        print('GEAK_PROFILE_DIAG: error', type(_e).__name__, str(_e))\n"

    # NCU self-test: ensure at least one known CUDA kernel is launched inside the profiled process.
    # If ncu still reports 'No kernels were profiled' with this enabled, the issue is with ncu/CUPTI/permissions.
    code += "if os.environ.get('GEAK_NCU_SELFTEST', '0') in {'1','true','yes','y'}:\n"
    code += "    try:\n"
    code += "        if torch is not None and hasattr(torch, 'cuda') and torch.cuda.is_available():\n"
    code += "            x = torch.randn((1024,), device='cuda')\n"
    code += "            y = x + 1\n"
    code += "            _ = y.sum()\n"
    code += "            torch.cuda.synchronize()\n"
    code += "            print('GEAK_NCU_SELFTEST: launched_cuda_ops')\n"
    code += "        else:\n"
    code += "            print('GEAK_NCU_SELFTEST: cuda_not_available')\n"
    code += "    except Exception as _e:\n"
    code += "        print('GEAK_NCU_SELFTEST: error', type(_e).__name__, str(_e))\n"

    with open(gen_file, 'w') as f:
        f.write(code)

    # Persist the generated script outside temp_root so users can reproduce profiling manually.
    # temp_root is often deleted by the agent at the end of an iteration.
    persist_enabled = str(os.environ.get("GEAK_PERSIST_PROFILE_SCRIPTS", "1")).lower() in {"1", "true", "yes", "y"}
    persist_dir = os.environ.get("GEAK_PROFILE_PERSIST_DIR", "").strip()
    persisted_gen_file = None
    if persist_enabled:
        if not persist_dir:
            # Default to a stable folder under CWD.
            persist_dir = os.path.abspath(os.path.join(os.getcwd(), "geak_profile_artifacts"))
        try:
            os.makedirs(persist_dir, exist_ok=True)
            persisted_name = os.path.basename(gen_file)
            persisted_gen_file = os.path.join(persist_dir, persisted_name)
            copyfile(gen_file, persisted_gen_file)
        except Exception:
            persisted_gen_file = None

    ## Execute two codes gen_file and triton_file using subprocess. 
    ## 1. If gen_file return error then return status as False, and stdout and stderr from gen file
    ## 2. If triton_file return error then return status as True and stdout and stderr as None
    ## 3. If gen_file and triton_file both return success then compare stdout from gen_file and triton_file. If stdout matches then return status as True, and stdout and stderr as None else return status as False and stdout and stderr as test cases mismatched.

    try:
        # Execute the generated code
        result_gen = subprocess.run([sys.executable, gen_file], capture_output=True, text=True, timeout=2*60)
        stdout_gen = result_gen.stdout
        stderr_gen = result_gen.stderr

        # Check if the generated code executed successfully
        if result_gen.returncode != 0:
            if verbose:
                print(f"Error in generated code: {stderr_gen}")
            return False, False, stdout_gen, stderr_gen

        # Execute the Triton code
        result_triton = subprocess.run([sys.executable, temp_triton_file], capture_output=True, text=True, timeout=2*60)
        stdout_triton = result_triton.stdout
        stderr_triton = result_triton.stderr

        # Check if the Triton code executed successfully
        if result_triton.returncode != 0:
            if verbose:
                print(f"Error in Triton code: {stderr_triton}")
            return None, None, None, None

        with open(gen_file+".out", 'w') as f:
            f.write(stdout_gen)
        with open(temp_triton_file+".out", 'w') as f:
            f.write(stdout_triton)

        with open(gen_file+".err", 'w') as f:
            f.write(stderr_gen)
        with open(temp_triton_file+".err", 'w') as f:
            f.write(stderr_triton)

        # Compare the outputs
        if stdout_gen == stdout_triton:
            return True, True, None, None
        else:
            return True, False, stdout_gen, "Error: not all test cases passed. The generated code and ground truth code produced different outputs."
    except Exception as e:
        if verbose:
            print(f"File: {fname}, Execution error: {e}")
        return False, False, None, str(e)
    # Clean up the temporary file
    except subprocess.TimeoutExpired:
        if verbose:
            print(f"File: {fname} timed out!")
        return None, None, None, "Time out"
    finally:
        pass
        # print(f"temp file for File: {fname} removed!")
        # if os.path.exists(gen_file):
        #     os.remove(gen_file)
    return False, False, None, None

def code_kernel_profiling(code, fname, py_folder, target_gpu, temp_root="tmp2", atol=1e-3, rtol=1e-1, timeout=6*60, verbose=False, gpu_id=None):
    tmp_gen_folder = os.path.join(temp_root, "gen")
    os.makedirs(tmp_gen_folder, exist_ok=True)
    
    
    triton_root = py_folder
    triton_file = os.path.join(triton_root, fname)

    gen_file = get_temp_file(prefix=f'{fname}_gen_triton_code')
    gen_file = os.path.join(tmp_gen_folder, gen_file)
    
    fname_split = fname.split('.')[0]
    hash_line = "#"*146

    harness_source = "hash_line"
    harness_found = False
    with open(triton_file, 'r') as f:
        lines = f.readlines()
        iL = None
        for idx, line in enumerate(lines):
            if line.strip() == hash_line:
                iL = idx
                harness_found = True
                break

        if harness_found and iL is not None:
            test_code_lines_procs = lines[iL + 1:]
        else:
            # Fallback: if the file does not contain the expected delimiter, try to include
            # the executable entrypoint so the script actually runs (and launches kernels).
            harness_source = "__main__"
            main_idx = None
            for idx, line in enumerate(lines):
                if "if __name__" in line and "__main__" in line:
                    main_idx = idx
                    break
            if main_idx is not None:
                test_code_lines_procs = lines[main_idx:]
                harness_found = True
            else:
                # Last resort: include nothing; we'll surface this explicitly in stdout_analyze.
                test_code_lines_procs = []

    # code = process_code(code)

    code =  code + '\n\n' + hash_line + '\n' + '\n' + '\n'.join(test_code_lines_procs)
    
    code += "\n\nimport os\n"
    code += "try:\n    import torch\nexcept Exception:\n    torch = None\n"
    code += "if os.environ.get('GEAK_PROFILE_DIAG', '0') in {'1','true','yes','y'}:\n"
    code += "    try:\n"
    code += "        if torch is not None and hasattr(torch, 'cuda') and torch.cuda.is_available():\n"
    code += "            torch.cuda.synchronize()\n"
    code += "            print('GEAK_PROFILE_DIAG: cuda_synchronized')\n"
    code += "        else:\n"
    code += "            print('GEAK_PROFILE_DIAG: cuda_not_available')\n"
    code += "    except Exception as _e:\n"
    code += "        print('GEAK_PROFILE_DIAG: error', type(_e).__name__, str(_e))\n"

    # NCU self-test: ensure at least one known CUDA kernel is launched inside the profiled process.
    # If ncu still reports 'No kernels were profiled' with this enabled, the issue is with ncu/CUPTI/permissions.
    code += "if os.environ.get('GEAK_NCU_SELFTEST', '0') in {'1','true','yes','y'}:\n"
    code += "    try:\n"
    code += "        if torch is not None and hasattr(torch, 'cuda') and torch.cuda.is_available():\n"
    code += "            x = torch.randn((1024,), device='cuda')\n"
    code += "            y = x + 1\n"
    code += "            _ = y.sum()\n"
    code += "            torch.cuda.synchronize()\n"
    code += "            print('GEAK_NCU_SELFTEST: launched_cuda_ops')\n"
    code += "        else:\n"
    code += "            print('GEAK_NCU_SELFTEST: cuda_not_available')\n"
    code += "    except Exception as _e:\n"
    code += "        print('GEAK_NCU_SELFTEST: error', type(_e).__name__, str(_e))\n"

    with open(gen_file, 'w') as f:
        f.write(code)

    # Persist the generated script outside temp_root so users can reproduce profiling manually.
    # temp_root is often deleted by the agent at the end of an iteration.
    persist_enabled = str(os.environ.get("GEAK_PERSIST_PROFILE_SCRIPTS", "1")).lower() in {"1", "true", "yes", "y"}
    persist_dir = os.environ.get("GEAK_PROFILE_PERSIST_DIR", "").strip()
    persisted_gen_file = None
    if persist_enabled:
        if not persist_dir:
            # Default to a stable folder under CWD.
            persist_dir = os.path.abspath(os.path.join(os.getcwd(), "geak_profile_artifacts"))
        try:
            os.makedirs(persist_dir, exist_ok=True)
            persisted_name = os.path.basename(gen_file)
            persisted_gen_file = os.path.join(persist_dir, persisted_name)
            copyfile(gen_file, persisted_gen_file)
        except Exception:
            persisted_gen_file = None

    try:
        ncu_exe = os.environ.get("NCU_BIN", "ncu")
        # Default to a rich metric set so the prompt has actionable info.
        # Users can override via NCU_SET (e.g., "launch", "speedOfLight", etc.).
        ncu_set = os.environ.get("NCU_SET", "full")
        ncu_sections = os.environ.get("NCU_SECTIONS", "")
        ncu_metrics = os.environ.get("NCU_METRICS", "")
        ncu_extra_args = os.environ.get("NCU_EXTRA_ARGS", "")

        # Some ncu flags (notably --target-processes all) can change behavior significantly.
        # Default to the simplest manual workflow unless explicitly enabled.
        ncu_target_processes = os.environ.get("NCU_TARGET_PROCESSES", "").strip()
        ncu_force_overwrite = str(os.environ.get("NCU_FORCE_OVERWRITE", "0")).lower() in {"1", "true", "yes", "y"}
        ncu_profile_from_start = os.environ.get("NCU_PROFILE_FROM_START", "").strip()

        # Use a deterministic report path so downstream code can store it as an artifact.
        # ncu produces a .ncu-rep report for both --export and -o.
        report_base = os.path.join(tmp_gen_folder, f"{fname_split}_ncu_report")
        report_rep = report_base + ".ncu-rep"

        # Ensure output directory exists (especially when report_base includes a path).
        try:
            os.makedirs(os.path.dirname(report_base), exist_ok=True)
        except Exception:
            pass

        # Match the simplest (and most commonly working) CLI pattern by default:
        #   ncu --set full -o report python script.py
        # Some environments behave better with `-o` than `--export`.
        # You can restore the old behavior via: NCU_OUTPUT_MODE=export
        ncu_output_mode = os.environ.get("NCU_OUTPUT_MODE", "o").strip().lower()

        ncu_args = [ncu_exe]
        if ncu_target_processes:
            ncu_args += ["--target-processes", ncu_target_processes]
        if ncu_force_overwrite:
            ncu_args += ["--force-overwrite", "true"]

        if ncu_output_mode == "export":
            ncu_args += ["--profile-from-start", "on", "--export", report_base]
        else:
            # Default: generate a .ncu-rep report file named `<report_base>.ncu-rep`.
            ncu_args += ["-o", report_base]

        # Allow overriding profile-from-start for both modes.
        if ncu_profile_from_start:
            ncu_args += ["--profile-from-start", ncu_profile_from_start]

        if ncu_set:
            ncu_args += ["--set", ncu_set]
        if ncu_sections:
            for section in [s.strip() for s in ncu_sections.split(",") if s.strip()]:
                ncu_args += ["--section", section]
        if ncu_metrics:
            ncu_args += ["--metrics", ncu_metrics]
        if ncu_extra_args:
            ncu_args += ncu_extra_args.split()

        ncu_args += [sys.executable, gen_file]

        run_env = dict(os.environ)
        if gpu_id is not None:
            run_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        run_env["GEAK_NCU_SELFTEST"] = run_env.get("GEAK_NCU_SELFTEST", "1")

        result_profile = subprocess.run(
            ncu_args,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
            env=run_env,
        )

        profile_status = result_profile.returncode == 0
        stdout_profile = result_profile.stdout
        stderr_profile = result_profile.stderr

    except Exception as e:
        if verbose:
            print(f"File: {fname}, Execution error: {e}")
        return None, None, str(e), None

    # Clean up the temporary file
    except subprocess.TimeoutExpired:
        if verbose:
            print(f"File: {fname} timed out!")
        return None, None, "Time out", None
    finally:
        pass

    # Check if the generated code executed successfully
    if result_profile.returncode != 0:
        if verbose:
            print(f"Error in profiling kernel")
    else:
        if verbose:
            print(f"Success in in profiling kernel")
    try:
        max_chars = int(os.environ.get("NCU_OUTPUT_CHARS", "8000"))
        cmd_str = " ".join(f'"{a}"' if (" " in a or "\t" in a) else a for a in ncu_args)

        no_kernels_profiled = "No kernels were profiled" in str(stdout_profile or "") or "No kernels were profiled" in str(stderr_profile or "")

        stdout_analyze = "\nBelow is Nsight Compute (ncu) profiling output for this kernel on an NVIDIA GPU."
        stdout_analyze += "\nCommand:\n" + cmd_str
        stdout_analyze += "\nReport (base path):\n" + report_base
        stdout_analyze += "\nReport (.ncu-rep):\n" + report_rep
        stdout_analyze += f"\nNCU output mode: {ncu_output_mode}"
        stdout_analyze += f"\nNCU target-processes: {ncu_target_processes if ncu_target_processes else '<unset>'}"
        stdout_analyze += f"\nNCU force-overwrite: {ncu_force_overwrite}"
        stdout_analyze += f"\nNCU profile-from-start: {ncu_profile_from_start if ncu_profile_from_start else '<unset>'}"
        stdout_analyze += f"\nNCU returncode: {result_profile.returncode}"
        stdout_analyze += f"\nCUDA_VISIBLE_DEVICES: {run_env.get('CUDA_VISIBLE_DEVICES', '<unset>')}"
        stdout_analyze += f"\nHarness extraction: source={harness_source}, found={harness_found}, appended_lines={len(test_code_lines_procs)}"
        if persisted_gen_file:
            stdout_analyze += "\nPersisted gen_file:\n" + persisted_gen_file
        else:
            stdout_analyze += "\nPersisted gen_file:\n<disabled_or_failed>"

        # Capture the exact ncu version used (important when multiple CUDA toolkits exist).
        try:
            ver = subprocess.run(
                [ncu_exe, "--version"],
                capture_output=True,
                text=True,
                timeout=30,
                shell=False,
                env=run_env,
            )
            ver_out = (ver.stdout or "").strip() or "<empty>"
            ver_err = (ver.stderr or "").strip() or "<empty>"
            stdout_analyze += f"\nNCU version returncode: {ver.returncode}"
            stdout_analyze += "\n[NCU VERSION STDOUT]\n" + ver_out
            stdout_analyze += "\n[NCU VERSION STDERR]\n" + ver_err
        except Exception as _e:
            stdout_analyze += f"\nNCU version: <error> {type(_e).__name__}: {_e}"

        combined = ""
        if stdout_profile:
            combined += "\n[NCU STDOUT]\n" + stdout_profile

        # Always include stderr section to avoid ambiguity (empty stderr is still useful information).
        combined += "\n[NCU STDERR]\n" + (stderr_profile if stderr_profile else "<empty>")

        # Always capture the raw script's stdout/stderr as ground truth when diagnosing profiling.
        # This helps distinguish:
        # - script never reached GPU kernel launch (likely)
        # - script launched kernels but ncu failed to capture them (tool/config/environment)
        if True:
            try:
                diag_timeout = int(os.environ.get("NCU_DIAG_TIMEOUT", "120"))
            except Exception:
                diag_timeout = 120

            try:
                diag_env = dict(os.environ)
                diag_env["GEAK_PROFILE_DIAG"] = "1"
                diag_env["GEAK_NCU_SELFTEST"] = diag_env.get("GEAK_NCU_SELFTEST", "1")
                if gpu_id is not None:
                    diag_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                diag = subprocess.run(
                    [sys.executable, gen_file],
                    capture_output=True,
                    text=True,
                    timeout=min(timeout, diag_timeout),
                    shell=False,
                    env=diag_env,
                )
                diag_out = diag.stdout if diag.stdout else "<empty>"
                diag_err = diag.stderr if diag.stderr else "<empty>"
                combined += "\n[SCRIPT DIAG]\n"
                combined += f"ReturnCode: {diag.returncode}\n"
                combined += "[SCRIPT STDOUT]\n" + diag_out
                combined += "\n[SCRIPT STDERR]\n" + diag_err
            except subprocess.TimeoutExpired:
                combined += "\n[SCRIPT DIAG]\nReturnCode: <timeout>\n"
            except Exception as e:
                combined += f"\n[SCRIPT DIAG]\nReturnCode: <error>\nError: {type(e).__name__}: {e}\n"

        # If we produced a report, try to import it to get a readable summary for prompt injection.
        # This mirrors the user's manual workflow: `ncu --import xxx.ncu-rep`.
        try:
            report_exists = os.path.exists(report_rep)
            combined += f"\n[NCU REPORT EXISTS]\n{report_exists}"
            if not report_exists:
                try:
                    nearby = []
                    base_dir = os.path.dirname(report_base)
                    if base_dir and os.path.isdir(base_dir):
                        for fn in os.listdir(base_dir):
                            if fn.startswith(f"{fname_split}_ncu_report"):
                                nearby.append(fn)
                    if nearby:
                        combined += "\n[NCU REPORT NEARBY FILES]\n" + "\n".join(sorted(nearby))
                except Exception:
                    pass

            if report_exists:
                import_args = [ncu_exe, "--import", report_rep]
                import_cmd_str = " ".join(f'"{a}"' if (" " in a or "\t" in a) else a for a in import_args)
                imported = subprocess.run(
                    import_args,
                    capture_output=True,
                    text=True,
                    timeout=min(timeout, 180),
                    shell=False,
                    env=run_env,
                )
                combined += "\n[NCU IMPORT CMD]\n" + import_cmd_str
                combined += f"\n[NCU IMPORT RETURN]\n{imported.returncode}"
                combined += "\n[NCU IMPORT STDOUT]\n" + (imported.stdout if imported.stdout else "<empty>")
                combined += "\n[NCU IMPORT STDERR]\n" + (imported.stderr if imported.stderr else "<empty>")
        except subprocess.TimeoutExpired:
            combined += "\n[NCU IMPORT]\n<timeout>\n"
        except Exception as e:
            combined += f"\n[NCU IMPORT]\n<error> {type(e).__name__}: {e}\n"

        if combined and len(combined) > max_chars:
            combined = combined[:max_chars] + "\n...<truncated>...\n"

        stdout_analyze += combined
    except Exception as e:
        return None, None, str(e), None
    return profile_status, stdout_profile, stderr_profile, stdout_analyze

def extract_code_from_llm_output(response):
    # Extract code blocks from the LLM response
    code = ""
    if "```" not in response:
        return response
    code_blocks = extract_code_blocks(response)
    for _code in code_blocks.code_dict_list:
        code += _code['context'] + "\n"
    return code

def get_fname_difficulty_from_label(label):
    triton_root = "dataloaders/TB_eval/TritonBench/data/TritonBench_G_comp_alpac_v1_fixed_with_difficulty.json"
    with open(triton_root, 'r') as f:
        data = json.load(f)
        for item in data:
            if item['output'] == label:
                return item['file'], item['difficulty']
    return None, None

def process_code(code: str):
    if "```python" in code:
        code = code.split("```python")[-1].strip().replace("```", "").replace("<|EOT|>", "")
    
    try:
        tree = ast.parse(code)
        imports = []
        function_definitions = []

        # Traverse the AST to find import statements and function definitions
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                # Collect the import statements
                imports.append(ast.unparse(node))  # Convert the AST node back to code
            elif isinstance(node, ast.FunctionDef):
                # Collect function definitions
                function_code = ast.unparse(node)  # Get the Python code for the function
                function_definitions.append(function_code)

        return "\n".join(imports) + "\n\n" + "\n".join(function_definitions)

    except:
        return code


def code_call_exec_success_allclose(code, fname, py_folder, temp_root="tmp2", atol=1e-3, rtol=1e-1, timeout=2*60, verbose=False, gpu_id=0):
    tmp_gen_folder = os.path.join(temp_root, "gen")
    os.makedirs(tmp_gen_folder, exist_ok=True)
    match = re.match(r"^([a-zA-Z0-9_]+?)(?:_\d+)?\.py$", fname)
    if match:
        op = match.group(1)
    filename = op + '.py'
    triton_root = py_folder
    triton_file = os.path.join(triton_root, filename)

    gen_file = get_temp_file(prefix=f'{fname}_gen_triton_code')
    gen_file = os.path.join(tmp_gen_folder, gen_file)

    hash_line = "#"*146

    with open(triton_file, 'r') as f:
        lines = f.readlines()
        for iL, line in enumerate(lines):
            if line.strip() == hash_line:
                break
        test_code_lines = lines[iL+1:]
        test_code_lines_procs = test_code_lines

    # code = process_code(code)

    code =  code + '\n\n' + hash_line + '\n' + '\n' + '\n'.join(test_code_lines_procs)

    with open(gen_file, 'w') as f:
        f.write(code)

    try:
        ## Just to a simple call to the generated code
        result_call = subprocess.run([f'CUDA_VISIBLE_DEVICES={gpu_id} "{sys.executable}" "{gen_file}"'], capture_output=True, text=True, timeout=timeout, shell=True)
        call_status = result_call.returncode == 0

        # Check for correctness
        result_corr = subprocess.run([f'CUDA_VISIBLE_DEVICES={gpu_id} "{sys.executable}" dataloaders/TB_eval/correctness.py --gen_file "{gen_file}" --ref_file "{triton_file}" --atol {atol} --rtol {rtol}'], capture_output=True, text=True, timeout=timeout, shell=True)
        stdout_corr = result_corr.stdout
        stderr_corr = result_corr.stderr

    except Exception as e:
        if verbose:
            print(f"File: {fname}, Execution error: {e}")
        return None, None, None, str(e), None, None
    
    # Clean up the temporary file
    except subprocess.TimeoutExpired:
        if verbose:
            print(f"File: {fname} timed out!")
        return None, None, None, "Time out", None, None
    finally:
        pass

    with open(gen_file+".stdout", 'w') as f:
        f.write(stdout_corr)

    with open(gen_file+".stderr", 'w') as f:
        f.write(stderr_corr)

    # Check if the generated code executed successfully
    if result_corr.returncode != 0:
        if verbose:
            print(f"Error in generated code: {stderr_corr}")
        return call_status, None, result_call.stdout, result_call.stderr, stdout_corr, stderr_corr
    else:
        if verbose:
            print(f"Success in generated code: {stdout_corr}")
        _, exec_status, gen_stdout, gen_stderr = stdout_corr.split("*#*#")
        return call_status, exec_status, result_call.stdout, result_call.stderr, gen_stdout, gen_stderr

    

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def green_or_red(status):
    if status:
        return bcolors.OKGREEN
    else:
        return bcolors.FAIL

def color_end():
    return bcolors.ENDC

def bool_colorize(status):
    if status:
        return bcolors.OKGREEN + str(status) + bcolors.ENDC
    else:
        return bcolors.FAIL + str(status) + bcolors.ENDC