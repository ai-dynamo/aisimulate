# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Configuration schema and field mapping adapted from ai-dynamo/dynamo,
# components/src/dynamo/router/simulation/{config.py,provider.py}, revision
# 6d6d36d7bfa63d80a0302df637c08f85523aa5b6 (Apache-2.0):
# https://github.com/ai-dynamo/dynamo/tree/6d6d36d7bfa63d80a0302df637c08f85523aa5b6/components/src/dynamo/router/simulation
# Modified for AISimulate execution using an externally installed Dynamo policy API.

"""Configuration and transport for externally installed Dynamo routing policy."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .config_adapter import PredictionAdapterContext, RecommendationAdapterContext
from .sweeper.provider import AdapterReplaySpec, AdapterSearchPlan, CandidateContext, JSONValue, RuntimeHookSpec
from .sweeper.replay import ReplaySpec

_NATIVE_HOOK = RuntimeHookSpec(provider="engine.router", kind="placement_policy", api_version=1)
_FiniteNonnegative = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]


class ConversationAffinityConfig(BaseModel):
    """Native conversation affinity whose lifetime follows simulation time."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["session", "sibling_group"]
    ttl_seconds: float = Field(default=3600, strict=True, ge=1, le=31_536_000, allow_inf_nan=False)


class _PrefillLoadModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Custom AIS load estimators belong to the optional integration stack.
    type: Literal["none"] = "none"


class RouterConfig(BaseModel):
    """Concrete native policy; parameter search and custom load models are unsupported."""

    model_config = ConfigDict(extra="forbid")

    policy: Literal["round_robin", "kv_router"] = "round_robin"
    prefill_load_model: _PrefillLoadModel = Field(default_factory=_PrefillLoadModel)
    overlap_score_credit: Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)] | None = None
    prefill_load_scale: _FiniteNonnegative | None = None
    temperature: _FiniteNonnegative | None = None
    affinity: ConversationAffinityConfig | None = None

    @field_validator("overlap_score_credit", "prefill_load_scale", "temperature", mode="before")
    @classmethod
    def _reject_explicit_null(cls, value: Any) -> Any:
        if value is None:
            raise ValueError("Router numeric knobs do not accept null")
        return value

    @model_validator(mode="after")
    def _validate_policy(self) -> RouterConfig:
        if self.policy == "round_robin":
            if self.affinity is not None:
                raise ValueError("router.affinity requires policy='kv_router'")
            knobs = {"overlap_score_credit", "prefill_load_scale", "temperature"} & self.model_fields_set
            if knobs:
                raise ValueError(f"round_robin rejects KV-router knobs {sorted(knobs)}")
        elif self.affinity is not None:
            for name, default in (("overlap_score_credit", 1.0), ("prefill_load_scale", 1.0), ("temperature", 0.0)):
                if getattr(self, name) not in (None, default):
                    raise ValueError("router.affinity requires default KV-router scoring parameters")
        return self


def _compile(config: Mapping[str, JSONValue]) -> AdapterReplaySpec:
    public = RouterConfig.model_validate(config)
    concrete = public.model_dump(mode="json", exclude_none=True)
    if public.policy == "round_robin":
        return AdapterReplaySpec(config=concrete)
    # Omit unspecified fields so the installed production policy owns its defaults.
    router_config = {
        native: getattr(public, name)
        for name, native in (
            ("overlap_score_credit", "overlap_score_credit"),
            ("prefill_load_scale", "prefill_load_scale"),
            ("temperature", "router_temperature"),
        )
        if getattr(public, name) is not None
    }
    native: dict[str, JSONValue] = {"router_config": router_config}
    if public.affinity is not None:
        native["affinity"] = public.affinity.model_dump(mode="json")
    hook = RuntimeHookSpec(
        provider=_NATIVE_HOOK.provider,
        kind=_NATIVE_HOOK.kind,
        api_version=_NATIVE_HOOK.api_version,
        config={"provider": "dynamo_kv_router", "config": native},
    )
    return AdapterReplaySpec(config=concrete, runtime_hooks=(hook,))


class EngineRouterConfigAdapter:
    """Compile a fixed routing policy without importing any Dynamo Python package."""

    name = "engine.router"
    section = "router"
    config_adapter_api_version = 3

    def compile_prediction(
        self, config: Mapping[str, JSONValue], context: PredictionAdapterContext
    ) -> AdapterReplaySpec:
        del context
        return _compile(config)

    def compile_recommendation(
        self, config: Mapping[str, JSONValue], context: RecommendationAdapterContext
    ) -> AdapterSearchPlan:
        del context
        compiled = _compile(config)
        return AdapterSearchPlan(state=compiled.config, potential_runtime_hooks=compiled.runtime_hooks)

    def materialize_candidate(
        self, plan: AdapterSearchPlan, selection: Mapping[str, JSONValue], context: CandidateContext
    ) -> AdapterReplaySpec:
        del context
        if selection:
            raise ValueError("engine.router supports a fixed policy, not router parameter search")
        if not isinstance(plan.state, Mapping):
            raise ValueError("engine.router search plan requires a concrete configuration")
        return _compile(plan.state)


def native_placement(spec: ReplaySpec) -> dict[str, JSONValue]:
    """Validate the built-in adapter and lower it to the core placement descriptor."""

    unknown = set(spec.adapters) - {"engine.router"}
    if unknown:
        raise ValueError(f"EngineReplayRunner does not support component adapters {sorted(unknown)}")
    adapter = spec.adapters.get("engine.router")
    if adapter is None:
        return {"provider": "round_robin", "config": None}
    expected = _compile(adapter.config)
    if adapter.runtime_hooks != expected.runtime_hooks:
        raise ValueError("engine.router runtime hooks must match the configured routing policy")
    deployment = spec.backend_deployment
    if deployment.deployment_mode not in {"agg", "disagg"} or deployment.encoder is not None:
        raise ValueError("engine.router requires aggregated or disaggregated text replay")
    if not expected.runtime_hooks:
        return {"provider": "round_robin", "config": None}
    if spec.execution_mode != "offline":
        raise ValueError("engine.router requires offline execution")
    if adapter.config.get("affinity") is not None:
        _require_affinity_scope(spec)
    return expected.runtime_hooks[0].config


def _require_affinity_scope(spec: ReplaySpec) -> None:
    deployment = spec.backend_deployment
    if deployment.backend not in {"vllm", "sglang"}:
        raise ValueError("router.affinity requires backend=vllm or sglang")
    if any(hook.kind == "scaling_policy" for hook in spec.runtime_hooks):
        raise ValueError("router.affinity requires static worker pools")
    for args in (deployment.agg_engine_args, deployment.prefill_engine_args, deployment.decode_engine_args):
        if not args:
            continue
        rank = args.get("rank", args)
        if not isinstance(rank, Mapping):
            continue  # The engine descriptor validator reports malformed ranks.
        if any(rank.get(key) is not None for key in ("native_host_offload", "g3_offload")):
            raise ValueError("router.affinity requires device-only KV cache without host or G3 offload")
        if rank.get("kv_cache_groups"):
            raise ValueError("router.affinity does not support grouped KV caches")
        timing = rank.get("timing_model")
        identity = timing.get("config", {}) if isinstance(timing, Mapping) else {}
        if not isinstance(identity, Mapping):
            identity = {}
        if any(
            source.get(key) not in (None, 0)
            for source in (args, rank, identity)
            for key in ("aic_nextn", "nextn", "speculation")
        ):
            raise ValueError("router.affinity requires speculative decoding disabled")
        if any(source.get("startup_time") not in (None, 0) for source in (args, rank)):
            raise ValueError("router.affinity requires static worker pools without startup delay")


class _DynamoPolicy:
    """Translate AISim lifecycle events into existing Dynamo policy operations."""

    def __init__(self, context_json: str):
        from dynamo import _core as native

        context = json.loads(context_json)
        if context["schema_version"] != 1:
            raise ValueError("unsupported external placement protocol version")
        self.role = context["role"]
        self.dp_size = context["dp_size"]
        self.workers = set(range(len(context["topology"])))
        for worker_id, worker in enumerate(context["topology"]):
            if worker["worker_id"] != worker_id or worker["scheduler_ids"] != list(
                range(worker_id * self.dp_size, (worker_id + 1) * self.dp_size)
            ):
                raise ValueError("Dynamo policy requires contiguous worker and rank IDs")
        rank = context["engine"]["rank"]
        self.block_size = rank["block_size"]
        config = context["placement"]["config"]
        router_config = dict(config.get("router_config", {}))
        # Each pool reports its own load: prefill has no decode occupancy, and
        # decode allocates independent blocks without prefill-cache credit.
        if self.role == "prefill":
            router_config["router_track_active_blocks"] = False
        elif self.role == "decode":
            router_config.update(
                overlap_score_credit=0.0,
                router_assume_kv_reuse=False,
                router_track_prefill_tokens=False,
                router_prefill_load_model="none",
            )
        self.native = native
        self.router = native.OfflineRouter(
            self.block_size,
            len(self.workers),
            self.dp_size,
            rank["num_gpu_blocks"],
            rank["max_num_batched_tokens"],
            json.dumps(router_config),
            context["selector_seed"],
        )
        self.tree = native.RadixTree()
        self.affinity_config = config.get("affinity")
        self.affinity = native.SessionAffinity(self.affinity_config["ttl_seconds"]) if self.affinity_config else None
        self.waiting = {}
        self.pending = {}
        self.staged = {}
        self.active = {}
        self.now_ms = 0.0
        self.ready = False
        self.evidence = dict(
            native_policy="dynamo.DefaultWorkerSelector",
            dynamo_version=native.__version__,
            physical_kv_events=0,
            decision_count=0,
            decisions_captured=context["capture_decisions"],
            decisions=[],
            post_dispatch_checks=0,
            dispatch_aborts=0,
        )

    def _group(self, command):
        if self.affinity is None:
            return None
        context = command["request"].get("replay_context") or {}
        agentic = context.get("agentic") or {}
        session = agentic.get("conversation_id") or command.get("session_id") or context.get("session_id")
        if not session:
            raise ValueError("conversation affinity requires nonempty session identity")
        play = agentic.get("play_id")
        identity = ["session", play, session]
        if self.affinity_config["mode"] == "sibling_group":
            lineage = agentic.get("lineage") or {}
            if lineage.get("schema") != "aisimulate.agentic.conversation-lineage.v1" or not lineage.get(
                "root_conversation_id"
            ):
                raise ValueError("sibling_group affinity requires unambiguous conversation lineage")
            parent = lineage.get("parent_conversation_id")
            identity = ["siblings" if parent else "root", play, lineage["root_conversation_id"], parent or session]
        return self.native.subagent_group_affinity_id(json.dumps(identity))

    def _submit(self, command):
        request = command["request"]
        request_id = request["uuid"]
        group = self._group(command)
        hold = self.affinity.try_acquire(group) if self.affinity else None
        if self.affinity and hold is None:
            self.waiting[request_id] = command
            return
        length = command["input_length"]
        hashes = command.get("hashes")
        tokens = request.get("tokens", [])
        if hashes is None:
            source = (request.get("replay_context") or {}).get("prompt_token_source")
            if source not in (None, "materialized") or len(tokens) != length:
                raise ValueError("Dynamo routing requires authored prompt tokens or complete block hashes")
            local = self.native.compute_block_hash_for_seq(tokens, self.block_size)
            sequence = None
        else:
            local, sequence = hashes["local_block_hashes"], hashes["sequence_hashes"]
            if len(local) != length // self.block_size or len(sequence) != len(local):
                raise ValueError("request hashes must cover exactly the complete prompt blocks")
        limit = (
            len(request["output_token_ids"])
            if request.get("output_token_ids") is not None
            else request["max_output_tokens"]
        )
        if command.get("max_output_tokens_override") is not None:
            limit = min(limit, command["max_output_tokens_override"])
        target = hold.target if hold else None
        self.router.submit(
            request_id,
            length,
            limit,
            self.tree.find_matches(local),
            local,
            sequence,
            tokens if len(tokens) == length else None,
            target,
            max(0, request.get("priority", 0)),
            request.get("strict_priority", 0),
            request.get("policy_class"),
            command.get("session_id"),
        )
        self.pending[request_id] = (length, group, hold)

    def _drain(self):
        for request_id in list(self.waiting):
            self._submit(self.waiting.pop(request_id))
        result = []
        for request_id, worker, rank, overlap, best in self.router.poll():
            length, group, hold = self.pending.pop(request_id)
            self.staged[request_id] = (hold, (worker, rank))
            self.evidence["decision_count"] += 1
            if self.evidence["decisions_captured"]:
                self.evidence["decisions"].append(
                    dict(
                        request_id=request_id,
                        role=self.role,
                        native_policy=self.evidence["native_policy"],
                        worker_id=worker,
                        dp_rank=rank,
                        group_key=group,
                        binding_reused=hold is not None and hold.target is not None,
                        overlap_blocks=overlap,
                        best_available_overlap_blocks=best,
                    )
                )
            result.append(
                dict(
                    request_id=request_id,
                    scheduler_id=worker * self.dp_size + rank,
                    reported_overlap_tokens=overlap * self.block_size,
                    cache_sample=dict(
                        overlap_blocks=overlap,
                        best_available_overlap_blocks=best,
                        isl_blocks=(length + self.block_size - 1) // self.block_size,
                    ),
                )
            )
        self.ready = False
        return result

    def _cancel(self, request_id):
        removed = self.waiting.pop(request_id, None) is not None
        pending = self.pending.pop(request_id, None)
        if pending is not None:
            self.router.cancel(request_id)
            if pending[2] is not None:
                pending[2].release()
            removed = True
        self.ready |= removed
        return removed

    def call_json(self, command_json: str) -> str:
        command = json.loads(command_json)
        op, request_id = command["op"], command.get("request_id")
        if "now_ms" in command:
            self.router.advance_clock(command["now_ms"])
            if self.affinity:
                self.affinity.advance_clock(command["now_ms"])
            self.now_ms = command["now_ms"]
        result = None
        if op == "place":
            request = command["request"]
            request_id = request["uuid"]
            if any(request_id in state for state in (self.waiting, self.pending, self.staged, self.active)):
                raise ValueError("duplicate routing request")
            if request.get("preferred_dp_rank") is not None or request.get("preferred_prefill_dp_rank") is not None:
                raise ValueError("Dynamo policy does not support authored DP-rank pins")
            if request.get("kv_hints") is not None:
                raise ValueError("Dynamo policy does not support authored KV-fetch hints")
            self._submit(command)
            ready = self._drain()
            result = dict(
                placement=next((row for row in ready if row["request_id"] == request_id), None),
                released=[row for row in ready if row["request_id"] != request_id],
            )
        elif op == "dispatch_committed":
            hold, target = self.staged.pop(request_id)
            self.active[request_id] = self.affinity.commit(hold, *target) if hold else None
            self.evidence["post_dispatch_checks"] += 1
            self.ready = True
        elif op in ("request_terminal", "dispatch_aborted"):
            self._cancel(request_id)
            hold, _ = self.staged.pop(request_id, (None, None))
            lease = self.active.pop(request_id, None)
            for owned in (hold, lease):
                if owned is not None:
                    owned.release()
            self.router.free(request_id)
            self.ready = True
            if op == "dispatch_aborted":
                self.evidence["dispatch_aborts"] += 1
            else:
                result = self._drain()
        elif op == "cancel_pending":
            result = self._cancel(request_id)
        elif op == "prefill_completed":
            self.router.prefill_complete(request_id)
            result = self._drain()
        elif op == "observe":
            for batch in command["events"]:
                if (
                    batch["stage"] != self.role
                    or batch["worker_id"] not in self.workers
                    or not 0 <= batch["dp_rank"] < self.dp_size
                ):
                    raise ValueError("cache observation has an invalid role, worker or rank")
                for event in batch["events"]:
                    if event.get("tier", "device") != "device" or event["dp_rank"] != batch["dp_rank"]:
                        raise ValueError("routing requires device KV events with matching rank")
                    if self.role != "decode":
                        self.tree.apply_event(batch["worker_id"], json.dumps(event).encode())
                        self.evidence["physical_kv_events"] += 1
            result = []
        elif op in ("advance_clock", "topology_settled"):
            result = self._drain()
        elif op in ("worker_ready", "worker_draining", "worker_removed"):
            worker = command["worker"]["worker_id"]
            if op == "worker_ready":
                self.router.add_worker(worker)
                self.workers.add(worker)
            elif op == "worker_draining":
                self.router.remove_worker(worker)
            else:
                self.router.unregister_worker(worker)
                self.tree.remove_worker(worker)
                self.workers.remove(worker)
            result = []
        elif op not in ("status", "snapshot"):
            raise ValueError(f"unknown policy operation {op!r}")
        pending = len(self.waiting) + len(self.pending)
        response = dict(
            result=result, pending_count=pending, next_wakeup_ms=self.now_ms if self.ready and pending else None
        )
        if op == "snapshot":
            response["evidence"] = self.evidence
        return json.dumps(response, separators=(",", ":"))


def run_dynamo_policy(runtime: Any, payload: str) -> str:
    """Run AISimulate's executor using independently installed native policy APIs."""
    try:
        from dynamo._core import OfflineRouter, SessionAffinity
    except ImportError as error:
        raise ValueError(
            "router.policy=kv_router requires ai-dynamo-runtime with OfflineRouter and SessionAffinity"
        ) from error
    if not hasattr(OfflineRouter, "submit") or not hasattr(SessionAffinity, "try_acquire"):
        raise ValueError("installed Dynamo runtime predates the standalone policy bindings")
    report = json.loads(runtime.run_replay_with_policy_json(payload, _DynamoPolicy))
    roles = report.pop("external_policy")
    first = next(iter(roles.values()))
    report["routing_policy"] = dict(
        api_version=1, native_policy=first["native_policy"], dynamo_version=first["dynamo_version"], roles=roles
    )
    return json.dumps(report, separators=(",", ":"))
