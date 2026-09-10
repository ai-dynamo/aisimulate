<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# System network reference configurations

The shipped system YAMLs select a reference network, not a network guaranteed by
the GPU SKU. NIC generation, active port speed, fabric (InfiniBand or Ethernet),
and NIC-to-GPU allocation depend on the deployment. InferenceX publishes useful
static hardware specifications; these do not establish the hardware of every
historical benchmark run or of the collectors that populated our database.

## Public evidence and defaults

Sources checked on **2026-09-09**. Bandwidth below is nominal **decimal GB/s per
GPU in one direction**, not measured application throughput. For the HGX-style
profiles we model the listed NIC capacity as available per GPU; shared NICs,
inactive ports, or oversubscribed fabrics require a deployment-specific override.

| Profile / InferenceX source | Published scale-out reference | NVLink domain / one-direction BW per GPU | Scale-out default in this change |
| --- | --- | --- | --- |
| [B200](https://inferencex.semianalysis.com/chips/b200) | CX7, 400GbE, gIB RoCEv2 | 8 GPUs / 900 GB/s | `inter_node_bw`: **100 → 50 GB/s** |
| [B300](https://inferencex.semianalysis.com/chips/b300) | CX8, 2 × 400GbE, RoCEv2 | 8 GPUs / 900 GB/s | `inter_node_bw`: **100 GB/s**, unchanged |
| [H100](https://inferencex.semianalysis.com/chips/h100) | CX7, 2 × 200GbE, RoCEv2 | 8 GPUs / 450 GB/s | `inter_node_bw`: **50 GB/s**, unchanged |
| [H200](https://inferencex.semianalysis.com/chips/h200) | CX7, 400 Gb/s, NDR InfiniBand | 8 GPUs / 450 GB/s | `inter_node_bw`: **50 GB/s**, unchanged |
| [GB200 NVL72](https://inferencex.semianalysis.com/chips/gb200-nvl72) | Cross-rack NIC and network not specified | 72 GPUs / 900 GB/s | `inter_rack_bw`: **100 GB/s**, retained assumption, not validated by this source |
| [GB300 NVL72](https://inferencex.semianalysis.com/chips/gb300-nvl72) | Cross-rack NIC and network not specified | 72 GPUs / 900 GB/s | `inter_rack_bw`: **100 GB/s**, retained assumption, not validated by this source |

GB200/GB300 use 900 GB/s for both `intra_node_bw` and `inter_node_bw`: leaving a
four-GPU node does **not** leave the 72-GPU NVLink domain. `inter_rack_bw` applies
only after leaving that domain. InferenceX's missing scale-out entry does not
mean these systems have no NICs or that NVLink extends across racks.

## What was incomplete in AIConfigurator #477

[AIConfigurator PR #477](https://github.com/ai-dynamo/aiconfigurator/pull/477)
raised six scale-out defaults on the premise that the previous NIC bandwidths
were one generation behind. That premise is not sufficient to choose a network
from a GPU name:

- B200's 100 GB/s / CX8 XDR assumption does not describe the public InferenceX
  CX7 400GbE reference. This change selects the documented 50 GB/s reference;
  it does not claim that an 800 Gb/s B200 deployment is impossible.
- H100's 50 GB/s value agrees with the aggregate reference capacity, but the
  unconditional NDR InfiniBand comment does not describe its RoCE reference.
- H200's 50 GB/s / NDR reference is consistent and is retained.
- GB200/GB300's cross-rack CX8 XDR assertion is not established by the cited
  NVL72 pages. We retain the numeric default for compatibility, label it as an
  assumption, and leave verification against the target fabric as a **TODO**.
- B300 was **not changed by #477**. Its current 100 GB/s is consistent with the
  InferenceX reference, but its XDR comment is corrected to dual-port RoCE.
  L40S was changed by #477 but is outside this evidence set and is untouched.

This is a scoped correction, not a blanket rollback of #477 or a claim that
every H100/B200 cluster uses Ethernet. No collector data, fitted parameters,
PCIe specifications, GPU compute specifications, or runtime schema are changed.

## Units and runtime impact

Convert network rates using `GB/s = Gb/s / 8`: 400 Gb/s gives 50 GB/s, two
200 Gb/s ports give 50 GB/s, and two 400 Gb/s ports give 100 GB/s. Summing ports
assumes both can serve the GPU concurrently; it is **not** summing TX and RX.
Full-duplex modeling uses `max(TX / BW, RX / BW)` for an endpoint, not
`(TX + RX) / BW`. NVLink values here already use the one-direction convention;
do not divide them by two again.

`SystemSpec.get_p2p_bandwidth` chooses the node/rack tier, and Rust consumers
also read these YAML bandwidths. Changing B200's nominal scale-out bandwidth
affects estimates that use that field; for example, a 1 GB one-way payload's
ideal serialization term changes from 10 ms to 20 ms. This is not a prediction
that end-to-end latency doubles. The transport descriptions are documentation,
not new protocol-specific runtime models.

The nominal `beta_spec` is distinct from DeepEP-LL's OLS-derived `beta_fit` and
from measured end-to-end throughput. Exact LL calibration does not gain a new
spec floor; donor extrapolation can change when the scale-out spec is the
bottleneck. See [DeepEP-LL modeling](DEEPEP_LL_MODELING.md).

For another deployment, use the existing custom systems-path mechanism with a
system YAML carrying its verified bandwidth and topology. TODO: collect active
link rates, NIC sharing, and effective communication bandwidth per deployment,
especially for cross-rack GB200/GB300, before treating these defaults as calibrated.
