# Task 3: AISimulate buffer-lease implementation

## Delivered

- Added the compatible V1 compact hash-buffer lease tail to the AISimulate
  Steppable plugin: lease-aware create, host-buffer registration, and compact
  range submission.
- Preserved the existing copied compact submission path unchanged for hosts
  without the lease capability.
- Added `CompactHashIdsLease`, a safe provider-facing core trait. A leased
  `CompactDirectRequest` carries an `Arc<dyn CompactHashIdsLease>`; the core
  retains that owner for an accepted request until terminal completion,
  cancellation, or replay destruction. `CompactDirectRequest::owned` remains
  the copied route.
- Kept FFI raw pointers inside the plugin's `HostHashBufferLease`. Its `Drop`
  invokes the validated host callback once, and only after core marks the
  submission accepted. Rejected submissions therefore do not emit a release.

## Evidence

- `cargo test -p aisimulate-steppable-plugin --test abi` — 7 passed, covering
  copied compact submission and lease release on terminal, cancellation, and
  destruction.
- `cargo fmt --check` — passed.
- `cargo clippy -p aisimulate-steppable-plugin --test abi -- -D warnings` —
  blocked by 94 existing unrelated `aisimulate-core` lint errors (mostly
  perfmodel and replay files outside this change); no change was made to hide
  or alter those baseline failures.

## Review remediation

- An accepted range now consumes its registered buffer ID immediately. The ID
  is removed before its release callback can let the host recycle the backing
  storage, so a later submission with that stale ID fails closed rather than
  creating a second lease or aliasing reused memory.
- Registration and all ABI `u32` slice borrowing reject misaligned host
  pointers before any `from_raw_parts` call.
- The focused cancellation regression covers both safeguards: a deliberately
  misaligned registration is rejected, and a released buffer ID cannot be
  resubmitted or trigger a second callback.
