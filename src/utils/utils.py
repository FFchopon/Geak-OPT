import re
import json
import ast

def extract_function_signatures(code):
    function_defs = []
    pattern = r'def\s+([a-zA-Z0-9_]+)\s*\(([^)]*)\)'
    matches = re.finditer(pattern, code)
    
    for match in matches:
        func_name = match.group(1)
        params = match.group(2)
        function_defs.append(f"def {func_name}({params})")
    
    return function_defs

def clear_code(code):
    if  "```python" in code:
        code = code.split("```python")[-1].replace("", "").replace("<|EOT|>", "")
    if "```" in code:
        code = code.split("```")[0]
    return code

def extract_function_calls(code):
    calls = []
    pattern = r'([a-zA-Z0-9_]+)\s*\(([^)]*)\)'
    matches = re.finditer(pattern, code)
    
    for match in matches:
        func_name = match.group(1)
        args = match.group(2)
        calls.append(f"{func_name}({args})")
    
    return calls

def infer_function_signatures_from_test_code(test_code):
    if not test_code:
        return []

    reserved = {
        "if",
        "for",
        "while",
        "return",
        "print",
        "range",
        "len",
        "int",
        "float",
        "str",
        "list",
        "dict",
        "set",
        "tuple",
        "min",
        "max",
        "sum",
        "abs",
    }

    blocked_prefixes = ("test_", "torch.", "triton.", "tl.")

    sigs = []
    seen = set()
    for call in extract_function_calls(test_code):
        name = call.split("(", 1)[0].strip()
        if not name:
            continue
        if name in reserved:
            continue
        if any(name.startswith(p) for p in blocked_prefixes):
            continue
        if "." in name:
            continue
        if name in seen:
            continue

        args_raw = call.split("(", 1)[1].rsplit(")", 1)[0]
        args = []
        for a in [x.strip() for x in args_raw.split(",") if x.strip()]:
            if a.startswith("*"):
                continue
            if "=" in a:
                a = a.split("=", 1)[0].strip()
            if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", a):
                continue
            args.append(a)

        if not args:
            continue

        sigs.append(f"def {name}({', '.join(args)})")
        seen.add(name)

    return sigs

def clear_json(response):
    if type(response) is dict:
        return response
    elif type(response) is not str:
        response = str(response)
    try:
        response = response.replace("\n", " ")
        response = re.search('({.+})', response).group(0)
        response = re.sub(r"(\w)'(\w|\s)", r"\1\\'\2", response)
        result = ast.literal_eval(response)
    except (SyntaxError, NameError, AttributeError):
        return "ERR_SYNTAX"
    return result