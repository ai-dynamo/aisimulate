# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure reductions and Slack rendering for the weekly accuracy report."""

from __future__ import annotations

import base64
import hashlib
import html
import json
import sys
import zlib
from array import array
from collections import defaultdict
from statistics import mean

METHODS = ("warmup", "nowarmup", "regression")


def encode_points(values):
    """Little-endian float64 percentage errors; -1 denotes an unsuccessful prediction."""
    packed = array("d", (-1 if value is None else value for value in values))
    if sys.byteorder != "little":
        packed.byteswap()
    return {"count": len(packed), "data": base64.b64encode(zlib.compress(packed.tobytes())).decode()}


def decode_points(packed):
    count = packed["count"]
    if type(count) is not int or not 0 <= count <= 10_000_000:
        raise ValueError("invalid point sequence length")
    decoder = zlib.decompressobj()
    raw = decoder.decompress(base64.b64decode(packed["data"], validate=True), count * 8 + 1)
    if len(raw) != count * 8 or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
        raise ValueError("invalid compressed point sequence")
    result = array("d")
    result.frombytes(raw)
    if sys.byteorder != "little":
        result.byteswap()
    return result


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def e2e_snapshot(summary, rules, url):
    campaign = summary["snapshot"]["campaign"]
    groups = {}
    for model in summary["models"]:
        for workload in model["workloads"]:
            for gpu in workload["gpus"]:
                for topology in gpu["topologies"]:
                    key = identity(
                        [
                            model["model"],
                            workload["identity"],
                            gpu["gpu"],
                            topology["id"],
                        ]
                    )
                    framework = topology["framework"].removeprefix("dynamo-")
                    framework = "trtllm" if framework == "trt" else framework
                    points = {}
                    for point in topology["points"]:
                        point_key = str(point["concurrency"])
                        if point_key in points:
                            raise ValueError("duplicate E2E point identity")
                        points[point_key] = (
                            [abs(point["aisimulate"][metric + "_error_pct"]) for metric in ("tpot", "ttft")]
                            if point["status"] == "success"
                            else None
                        )
                    if key in groups:
                        raise ValueError("duplicate E2E topology identity")
                    groups[key] = {
                        "label": f"{model['model']} / {gpu['gpu']} / {framework} / {topology['id']}",
                        "model": model["model"],
                        "gpu": gpu["gpu"],
                        "framework": framework,
                        "points": points,
                    }
    return {
        "kind": "e2e",
        "branch": campaign["branch"],
        "commit": campaign["commit_sha"],
        "dataset": campaign["measurement_sha256"],
        "rules": rules,
        "url": url,
        "groups": groups,
    }


def fpm_snapshot(summary, rules, url, evidence=None):
    groups = {}
    for row in summary["rows"]:
        row_key = row["configuration_id"] + "/" + row["snapshot_id"]
        for method, result in row["results"].items():
            key = row_key + "/" + method
            samples = evidence[row_key]["methods"].get(method) if evidence is not None else None
            groups[key] = {
                "label": f"{row['model']} / {row['gpu']} / {row['configuration_id']} / {method}",
                "model": row["model"],
                "gpu": row["gpu"],
                "method": method,
                "membership": row["membership_sha256"],
                "protocol": [
                    row["protocol_id"],
                    row["parser_policy_id"],
                    row["ordering"],
                ],
                "points": samples,
                "point_order": evidence[row_key]["order_sha256"] if evidence is not None else None,
                "metric": result["metrics"]["all"],
            }
    snapshot = summary["snapshot"]
    return {
        "kind": "fpm",
        "branch": snapshot["branch"],
        "commit": snapshot["commit_sha"],
        "dataset": snapshot["hf_revision"],
        "rules": rules,
        "url": url,
        "groups": groups,
    }


def compare(current, previous):
    """Compare successful common points; lost successes alert independently of MAPE."""
    if previous is None:
        return [], ["Initial baseline; no earlier qualified comparison."]
    if current["dataset"] != previous["dataset"] or current["rules"] != previous["rules"]:
        return [], ["Dataset or evaluation rules changed; baseline reset, no regression conclusion."]
    alerts, notes = [], []
    fpm = current["kind"] == "fpm"
    metrics = ("forward-pass",) if fpm else ("TPOT", "TTFT")
    for key, old in previous["groups"].items():
        new = current["groups"].get(key)
        if old["points"] is None or (new is not None and new["points"] is None):
            notes.append("Point evidence unavailable; coverage and common-point regression comparison unavailable.")
            continue
        before_points = decode_points(old["points"]) if fpm else old["points"]
        successful = (
            {i for i, v in enumerate(before_points) if v >= 0}
            if fpm
            else {k for k, v in before_points.items() if v is not None}
        )
        if new is None:
            if successful:
                alerts.append(f"{old['label']}: {len(successful)} previously predicted points missing.")
            continue
        if old.get("protocol") != new.get("protocol") or (
            fpm and (old.get("membership"), old.get("point_order")) != (new.get("membership"), new.get("point_order"))
        ):
            notes.append(f"{new['label']}: measurement membership/order or protocol changed; not comparable.")
            continue
        after_points = decode_points(new["points"]) if fpm else new["points"]
        after_success = (
            {i for i, v in enumerate(after_points) if v >= 0}
            if fpm
            else {k for k, v in after_points.items() if v is not None}
        )
        lost = successful - after_success
        if lost:
            alerts.append(f"{new['label']}: {len(lost)} previously predicted points now missing/unavailable/failed.")
        common = successful & after_success
        if not common:
            continue
        for index, metric in enumerate(metrics):
            before = mean(before_points[k] if fpm else before_points[k][index] for k in common)
            after = mean(after_points[k] if fpm else after_points[k][index] for k in common)
            delta = after - before
            if delta >= 2 - 1e-9 and delta >= before * 0.10 - 1e-9:
                relative = f"{delta / before * 100:.1f}%" if before else "from zero"
                alerts.append(
                    f"{new['label']}: {metric} MAPE {before:.2f}% -> {after:.2f}% "
                    f"(+{delta:.2f} pp, {relative}; {len(common)} common points)."
                )
    return alerts, sorted(set(notes))


def reduce_e2e(groups):
    points = [point for group in groups for point in group["points"].values()]
    success = [point for point in points if point is not None]
    pair = "/".join(f"{mean(p[i] for p in success):.2f}%" for i in (0, 1)) if success else "N/A"
    return pair, f"{len(success)}/{len(points)}"


def reduce_fpm(groups, *, coverage=True):
    metrics = [group["metric"] for group in groups]
    count = sum(m["predicted_count"] for m in metrics)
    measured = sum(m["measured_count"] for m in metrics)
    mape = sum((m["mape_pct"] or 0) * m["predicted_count"] for m in metrics) / count if count else None
    value = f"{mape:.2f}%" if mape is not None else "N/A"
    return f"{value} ({count}/{measured})" if coverage else value


def table(headers, rows):
    # Keep arbitrary artifact strings from breaking code fences or generating mentions.
    def clean(value):
        return str(value).replace("`", "'").replace("\n", " ").replace("\r", " ")

    rows = [[clean(c) for c in row] for row in [headers, *rows]]
    widths = [max(len(row[i]) for row in rows) for i in range(len(headers))]
    return (
        "```\n"
        + "\n".join("  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)) for row in rows)
        + "\n```"
    )


def escape(text):
    return html.escape(str(text), quote=False)


def messages(day, pipelines, snapshots, alerts, notes, recovered=()):
    run_links = []
    for kind, pipeline in pipelines.items():
        status = pipeline["status"]
        detail = status.removeprefix("success").strip() if status.startswith("success") else f"({status})"
        run_links.append(f"<{pipeline['url']}|{kind.upper()} run>{escape(detail)}")
    links = " · ".join(run_links)
    summary = f"> *:alert: {len(alerts)} alert(s)*"
    if notes:
        summary += f" · {len(notes)} comparison note(s) in thread"
    lines = [
        f":rainbow: *Accuracy Weekly · {day}*",
        "> " + links + " · <https://ai-dynamo.org/aisimulate/e2e-accuracy/|E2E overview>"
        " · <https://ai-dynamo.org/aisimulate/fpm-accuracy/|FPM overview>",
    ]
    lines.extend("> " + escape("• " + (alert if len(alert) <= 240 else alert[:237] + "...")) for alert in alerts[:3])
    if len(alerts) > 3:
        lines.append(f"> {len(alerts) - 3} additional alerts in thread.")
    lines.extend("> " + escape("• Recovered: " + item) for item in recovered)
    rows = []
    for branch, snapshot in sorted(snapshots.get("e2e", {}).items()):
        groups = list(snapshot["groups"].values())
        overall, coverage = reduce_e2e(groups)
        row = [branch, overall]
        row.extend(reduce_e2e([g for g in groups if g["framework"] == f])[0] for f in ("vllm", "sglang", "trtllm"))
        rows.append([*row, coverage])
    lines += [
        "*E2E · TPOT/TTFT MAPE*",
        escape(table(["Branch", "Overall", "vLLM", "SGLang", "TRT-LLM", "Coverage"], rows))
        if rows
        else "No qualified E2E results.",
    ]
    for branch, snapshot in sorted(snapshots.get("e2e", {}).items()):
        for dimension in ("model", "gpu"):
            buckets = defaultdict(list)
            for group in snapshot["groups"].values():
                buckets[group[dimension]].append(group)
            entries = [[label, *reduce_e2e(groups)] for label, groups in sorted(buckets.items())]
            lines.extend(
                [
                    f"*E2E · {escape(branch)} · per {dimension}*",
                    escape(table([dimension.title(), "TPOT/TTFT MAPE", "Coverage"], entries)),
                ]
            )
    rows = []
    for branch, snapshot in sorted(snapshots.get("fpm", {}).items()):
        rows.append(
            [
                branch,
                *[
                    reduce_fpm([g for g in snapshot["groups"].values() if g["method"] == m], coverage=False)
                    for m in METHODS
                ],
            ]
        )
    lines += [
        "*FPM · MAPE*",
        escape(table(["Branch", "KV on", "KV off", "Regression"], rows)) if rows else "No qualified FPM results.",
    ]
    lines.append(summary)
    replies = []
    coverage_rows = []
    for branch, snapshot in sorted(snapshots.get("fpm", {}).items()):
        cells = []
        for method in METHODS:
            metrics = [g["metric"] for g in snapshot["groups"].values() if g["method"] == method]
            cells.append(f"{sum(m['predicted_count'] for m in metrics)}/{sum(m['measured_count'] for m in metrics)}")
        coverage_rows.append([branch, *cells])
    if coverage_rows:
        replies.append(
            "*FPM coverage · predicted/eligible*\n"
            + links
            + "\n"
            + escape(table(["Branch", "KV on", "KV off", "Regression"], coverage_rows))
            + "\nCoverage counts eligible points. Run artifacts are authoritative; Pages may lag."
        )
    for offset in range(0, len(alerts) + len(notes), 8):
        replies.append(
            "*Comparison details*\n"
            + links
            + "\n"
            + "\n".join(escape("• " + line) for line in (alerts + notes)[offset : offset + 8])
        )
    root = "\n".join(lines)
    # Slack hard limit is 40k; refuse instead of silently losing branches or evidence.
    if any(len(message) > 35000 for message in [root, *replies]):
        raise ValueError("Slack message too large; review report size before delivery")
    return root, replies
