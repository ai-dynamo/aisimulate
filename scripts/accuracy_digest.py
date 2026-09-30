# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure reductions and Slack rendering for the daily accuracy report."""

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
    pair = " / ".join(f"{mean(p[i] for p in success):.2f}" for i in (0, 1)) if success else "N/A"
    return pair, f"{len(success)}/{len(points)}"


def reduce_fpm(groups):
    metrics = [group["metric"] for group in groups]
    count = sum(m["predicted_count"] for m in metrics)
    measured = sum(m["measured_count"] for m in metrics)
    mape = sum((m["mape_pct"] or 0) * m["predicted_count"] for m in metrics) / count if count else None
    return f"{mape:.2f}% ({count}/{measured})" if mape is not None else f"N/A (0/{measured})"


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
    links = " · ".join(f"<{p['url']}|{kind.upper()} run>" for kind, p in pipelines.items())
    statuses = " · ".join(f"{kind.upper()}: {p['status']}" for kind, p in pipelines.items())
    lines = [f"*AISimulate Accuracy Daily · {day} (Los Angeles)*", statuses, links]
    rows = []
    for branch, snapshot in sorted(snapshots.get("e2e", {}).items()):
        groups = list(snapshot["groups"].values())
        overall, coverage = reduce_e2e(groups)
        row = [branch, overall]
        row.extend(reduce_e2e([g for g in groups if g["framework"] == f])[0] for f in ("vllm", "sglang", "trtllm"))
        rows.append([*row, coverage])
    lines += [
        "*E2E · TPOT / TTFT MAPE (%)*",
        escape(table(["Branch", "Overall", "vLLM", "SGLang", "TRT-LLM", "Coverage"], rows))
        if rows
        else "No qualified E2E results.",
    ]
    rows = []
    for branch, snapshot in sorted(snapshots.get("fpm", {}).items()):
        rows.append(
            [
                branch,
                *[reduce_fpm([g for g in snapshot["groups"].values() if g["method"] == m]) for m in METHODS],
            ]
        )
    lines += [
        "*FPM · MAPE % (predicted/measured)*",
        escape(table(["Branch", "KV warmup on", "KV warmup off", "Online regression"], rows))
        if rows
        else "No qualified FPM results.",
    ]
    lines.append(f"*Attention: {len(alerts)} alert(s)*")
    lines.extend(escape("• " + alert) for alert in alerts[:8])
    if len(alerts) > 8:
        lines.append(f"{len(alerts) - 8} additional alerts in thread.")
    lines.extend(escape("• " + note) for note in notes[:8])
    lines.extend(escape("• Recovered: " + item) for item in recovered)
    lines += [
        "<https://ai-dynamo.org/aisimulate/e2e-accuracy/|E2E Overview> · "
        "<https://ai-dynamo.org/aisimulate/fpm-accuracy/|FPM Overview>",
        "Coverage counts eligible points. Pages may still show an older snapshot; run artifacts are authoritative.",
    ]
    replies = []
    for branch, snapshot in sorted(snapshots.get("e2e", {}).items()):
        for dimension in ("model", "gpu"):
            buckets = defaultdict(list)
            for group in snapshot["groups"].values():
                buckets[group[dimension]].append(group)
            # Split at rows, preserving a complete code fence in every reply.
            entries = [[label, *reduce_e2e(groups)] for label, groups in sorted(buckets.items())]
            for offset in range(0, len(entries), 12):
                replies.append(
                    f"*E2E · {escape(branch)} · per {dimension}*\n<{snapshot['url']}|Pipeline>\n"
                    + escape(
                        table(
                            [dimension.title(), "TPOT / TTFT MAPE (%)", "Coverage"],
                            entries[offset : offset + 12],
                        )
                    )
                )
    for offset in range(0, len(alerts) + len(notes), 8):
        replies.append(
            "*Comparison details*\n"
            + links
            + "\n"
            + "\n".join(escape("• " + line) for line in (alerts + notes)[offset : offset + 8])
        )
    root = "\n\n".join(lines)
    # Slack hard limit is 40k; refuse instead of silently losing branches or evidence.
    if any(len(message) > 35000 for message in [root, *replies]):
        raise ValueError("Slack message too large; reduce the table chunk size")
    return root, replies
