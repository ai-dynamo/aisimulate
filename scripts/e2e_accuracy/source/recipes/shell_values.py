# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Non-executing evaluator for recipe scalar/array assignments and known branches.

Unknown conditions taint their assignments. Commands are never executed, and
unknown variables used by a serving command fail closed.
"""

from __future__ import annotations

import ast
import fnmatch
import operator
import re
import shlex

from scripts.e2e_accuracy.source.recipes.inferencex_recipe import InferenceXRecipeError

OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.FloorDiv: operator.floordiv,
    ast.Div: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
}


def arithmetic(text: str, values: dict) -> int:
    text = text.strip().replace("$", "")
    ternary = re.fullmatch(r"(.+?)\?(.+?):(.+)", text)
    if ternary:
        return arithmetic(ternary[2] if arithmetic(ternary[1], values) else ternary[3], values)

    def evaluate(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, bool)):
            return node.value
        if isinstance(node, ast.Name) and node.id in values:
            return int(values[node.id])
        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            return OPS[type(node.op)](evaluate(node.left), evaluate(node.right))
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in OPS:
            return OPS[type(node.ops[0])](evaluate(node.left), evaluate(node.comparators[0]))
        raise InferenceXRecipeError(f"unresolved arithmetic: {text}")

    try:
        return int(evaluate(ast.parse(text, mode="eval").body))
    except (SyntaxError, ValueError, TypeError, ZeroDivisionError) as error:
        raise InferenceXRecipeError(f"unresolved arithmetic: {text}") from error


def expand(text: str, values: dict) -> str:
    text = re.sub(r'"(\$\{[A-Za-z_]+\[@\]\})"', r"\1", text)
    text = re.sub(r"\$\(\((.*?)\)\)", lambda m: str(arithmetic(m[1], values)), text)

    def conditional_echo(match):
        selected = condition(match[1], values)
        if selected is None:
            raise InferenceXRecipeError("unresolved conditional echo")
        return match[2] if selected else match[3]

    text = re.sub(
        r"\$\(\s*(\[\[.*?\]\])\s*&&\s*echo\s+([\w.-]+)\s*\|\|\s*echo\s+([\w.-]+)\s*\)", conditional_echo, text
    )

    def variable(m):
        expr = m[1] or m[2]
        name = re.split(r"\[|:|%", expr)[0]
        if name not in values or (values[name] == "" and ":-" in expr):
            if ":-" in expr:
                return expr.split(":-", 1)[1]
            raise InferenceXRecipeError(f"unresolved shell variable {name}")
        value = values[name]
        value = " ".join(map(str, value)) if isinstance(value, list) else str(value)
        if "%, " in expr:
            value = value.removesuffix(", ")
        return value

    text = re.sub(r"\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z_0-9]*)", variable, text)
    if "$" in text or "`" in text:
        raise InferenceXRecipeError("unsupported command substitution")
    return text


def condition(text: str, values: dict) -> bool | None:
    text = text.strip().removesuffix(";").strip()
    text = re.sub(r"^\[\[?\s*|\s*\]\]?$", "", text)
    if "&&" in text:
        parts = [condition(p, values) for p in text.split("&&")]
        return False if False in parts else None if None in parts else True
    if "||" in text:
        parts = [condition(p, values) for p in text.split("||")]
        return True if True in parts else None if None in parts else False
    try:
        tokens = shlex.split(expand(text, values))
        if len(tokens) == 2 and tokens[0] in ("-n", "-z"):
            return bool(tokens[1]) == (tokens[0] == "-n")
        if len(tokens) != 3:
            return None
        a, op, b = tokens
        if op in ("=", "==", "!="):
            result = fnmatch.fnmatchcase(a, b)
            return not result if op == "!=" else result
        compare = {
            "-eq": operator.eq,
            "-ne": operator.ne,
            "-gt": operator.gt,
            "-ge": operator.ge,
            "-lt": operator.lt,
            "-le": operator.le,
        }.get(op)
        return compare(int(a), int(b)) if compare else None
    except (InferenceXRecipeError, ValueError):
        return None


def resolve_lines(text: str, initial: dict) -> tuple[list[str], dict, dict[str, str]]:
    values = dict(initial)
    text = re.sub(r"(?ms)^sanitize_slurm_mpi_env_for_trtllm\(\) \{.*?^\}", "", text)
    # Reviewed bounded graph-capture loop. No shell loop is executed.
    capture = (
        r"CAPTURE_SIZE=4\s+while \(\( CAPTURE_SIZE < CONC \)\); "
        r"do CAPTURE_SIZE=\$\(\(CAPTURE_SIZE \* 2\)\); done\s+"
        r"\(\( CAPTURE_SIZE > 2048 \)\) && CAPTURE_SIZE=2048"
    )
    if re.search(capture, text):
        text = re.sub(capture, f"CAPTURE_SIZE={min(2048, max(4, 1 << (int(values['CONC']) - 1).bit_length()))}", text)
    text = text.replace("\\\n", " ")
    # Fold multiline shell arrays into one assignment.
    text = re.sub(
        r"(?m)^(\s*[A-Za-z_]+\+?=\()\s*\n(.*?)\n\s*\)",
        lambda m: m[1] + " ".join(m[2].splitlines()) + ")",
        text,
        flags=re.S,
    )
    # Preserve multiline quoted YAML fragments as one scalar assignment.
    text = re.sub(
        r'(?m)^(\s*(?:export )?[A-Za-z_]+=")([^"]*\n[^"]*)"',
        lambda m: m[1] + m[2].replace("\n", "\x00") + '"',
        text,
        flags=re.S,
    )
    lines = text.splitlines()
    stack = []
    active: bool | None = True
    selected = []
    files = {}
    index = 0
    while index < len(lines):
        line = lines[index].strip().replace("\x00", "\n")
        index += 1
        if not line or line.startswith("#"):
            continue
        # Single-line download/logging guards contain no serving configuration.
        if line.startswith("if ") and "; fi" in line and not re.search(r"\b(?:serve|launch_server|[A-Za-z_]+=)", line):
            continue
        if line.startswith(("if ", "elif ")) and line.endswith("then"):
            test = condition(re.sub(r"^(?:if|elif)\s+|;?\s*then$", "", line), values)
            if line.startswith("if "):
                stack.append([active, test])
            else:
                parent, prior = stack[-1]
                test = False if prior is True else None if prior is None else test
                stack[-1][1] = (
                    True if prior is True or test is True else None if prior is None or test is None else False
                )
            parent = stack[-1][0]
            active = False if parent is False or test is False else None if parent is None or test is None else True
            continue
        if line == "else":
            parent, prior = stack[-1]
            active = False if parent is False or prior is True else None if parent is None or prior is None else True
            continue
        if line == "fi":
            if not stack:
                raise InferenceXRecipeError("unbalanced recipe condition")
            active = stack.pop()[0]
            continue
        case = re.fullmatch(r"case\s+(.+?)\s+in", line)
        if case:
            body = []
            while index < len(lines) and lines[index].strip() != "esac":
                body.append(lines[index])
                index += 1
            index += 1
            if active is False:
                continue
            try:
                selector = shlex.split(expand(case[1], values))[0]
            except (InferenceXRecipeError, IndexError):
                raise InferenceXRecipeError("unresolved case selector") from None
            selected_case = ""
            for pattern, commands in re.findall(r"(?:^|\n)\s*([^()\n]+)\)(.*?)\;\;", "\n".join(body), re.S):
                if any(fnmatch.fnmatchcase(selector, p.strip().strip("\"'")) for p in pattern.split("|")):
                    selected_case = commands
                    break
            lines[index:index] = selected_case.splitlines()
            continue
        here = re.search(r'cat\s+(?:>[^<]+\s+)?<<\s*[\'"]?(\w+)[\'"]?\s*(>>?)?\s*(.*)', line)
        if here:
            body = []
            while index < len(lines) and lines[index].strip() != here[1]:
                body.append(lines[index])
                index += 1
            index += 1
            if active is True:
                target = here[3] or re.search(r">\s*(\S+)\s*<<", line)[1]
                target = expand(target, values).strip("\"'")
                rendered = expand("\n".join(body), values)
                files[target] = files.get(target, "") + "\n" + rendered if here[2] == ">>" else rendered
            continue
        serving_line = line
        if "${SERVE_CMD[@]}" in line and not line.startswith("SERVE_CMD="):
            serving_line = expand(re.sub(r'\s+(?:>>?|2>>?)\s*["$A-Za-z/].*$', "", line), values)
        if not re.match(r"[A-Za-z_]+=[(\"]", line) and re.search(
            r"\b(?:vllm serve|sglang serve|sglang.launch_server|trtllm-serve)\b", serving_line
        ):
            if active is None:
                raise InferenceXRecipeError("serving command depends on an unresolved condition")
            if active:
                selected.append(expand(re.sub(r'\s+(?:>>?|2>>?)\s*["$A-Za-z/].*$', "", serving_line), values))
            continue
        assignment = re.fullmatch(r"(?:export\s+)?([A-Za-z_][A-Za-z_0-9]*)(\+?=)(.*)", line, re.S)
        if assignment:
            name, op, value = assignment.groups()
            if active is False:
                continue
            if active is None:
                values.pop(name, None)
                continue
            try:
                seq = re.compile(r"\$\(\s*seq\s+([^()]+)\)")

                def sequence(m):
                    nums = [int(x) for x in shlex.split(expand(m[1], values))]
                    start, step, stop = (nums[0], 1, nums[1]) if len(nums) == 2 else nums
                    if step <= 0 or abs(stop - start) > 100000:
                        raise InferenceXRecipeError("unbounded capture sequence")
                    return " ".join(map(str, range(start, stop + 1, step)))

                value = seq.sub(sequence, value)
                printf = re.fullmatch(r'\$\(printf "%s, " "\$\{([A-Za-z_]+)\[@\]\}"\)', value)
                if printf:
                    value = ", ".join(map(str, values[printf[1]])) + ", "
                elif value.startswith("(") and value.endswith(")"):
                    value = shlex.split(expand(value[1:-1], values))
                else:
                    tokens = shlex.split(expand(value, values), comments=True)
                    value = " ".join(tokens)
                if op == "+=":
                    value = values[name] + value
                values[name] = value
            except (InferenceXRecipeError, ValueError, KeyError, TypeError):
                values.pop(name, None)
            continue
    if stack:
        raise InferenceXRecipeError("unclosed recipe condition")
    return selected, values, files
