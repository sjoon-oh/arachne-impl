# Exposed CostAwareReplacementConfig's tuning knobs as CLI flags

Test-infra-only change (`cpp/test/index/hnsw_workload_compare.cpp`), no
core library (`arachne_core`) touched. Follow-up to the CostAware-policy
refinement discussion this entry's own conversation covers (options
B/minimum_residency, C/admission_hysteresis, E/minimum_observations,
A/pass-aware batch admission -- deprioritized, see that discussion): before
any of B/C/E could be swept and measured, the harness needed a way to pass
non-default values at all -- previously every run got
`CostAwareReplacementConfig{}`'s hardcoded defaults, no override path
existed anywhere.

## What changed

- `Args` gained six `std::optional<...>` fields, one per
  `CostAwareReplacementConfig` member (`minimum_observations`,
  `heat_half_life`, `minimum_residency`, `admission_hysteresis`,
  `potential_writeback_weight`, `maximum_incremental_bytes`) -- each stays
  `std::nullopt` unless its own flag is passed, so an invocation that never
  mentions any of them is unaffected.
- New flags: `--cost-aware-min-observations`, `--cost-aware-heat-half-life-ms`,
  `--cost-aware-min-residency-ms`, `--cost-aware-admission-hysteresis`,
  `--cost-aware-writeback-weight`, `--cost-aware-max-incremental-bytes`. Only
  meaningful with `--replacement-policy cost_aware` (the default);
  silently ignored otherwise, matching how every other cost_aware-specific
  thing already behaves in this harness.
- `MakeReplacementPolicy()`: signature changed from `(const std::string&)`
  to `(const Args&)` (needs more than the policy name now). Behavior
  preserved exactly when no `--cost-aware-*` flag is given -- still returns
  `nullptr` for `cost_aware` (Controller/RegionManager's own
  default-construction path, byte-for-byte the same as before this entry).
  The moment any one flag is set, builds a `CostAwareReplacementConfig`
  starting from its own defaults, overrides only the fields that were
  actually passed, and returns an explicitly-constructed
  `CostAwareReplacementPolicy` instead.
- Startup banner gained a `cost_aware: min_observations=... heat_half_life_ms=...
  min_residency_ms=... admission_hysteresis=... writeback_weight=...
  max_incremental_bytes=...` line, printed only when
  `--replacement-policy cost_aware`, showing *resolved* values (default or
  overridden, either way) -- so a sweep script's own logs are
  self-describing about which config every run actually used, without
  needing to cross-reference the invoking command line.

## Verification

- Both build dirs (`cpp/build/`, `cpp/build-trace/`) compile clean.
- Full unit suite: 365/366 (same single pre-existing, unrelated failure as
  every entry in this investigation) -- expected, since nothing in
  `arachne_core` changed.
- Smoke-tested at small scale (`--limit-base 2000 --limit-steps 2`): (1) an
  override (`--cost-aware-admission-hysteresis 1.5 --cost-aware-min-observations 2`)
  shows up correctly in the resolved-config banner; (2) no flags given shows
  `CostAwareReplacementConfig{}`'s own defaults (`min_observations=1
  admission_hysteresis=1 ...`); (3) `--replacement-policy fifo` with no
  cost-aware flags runs fine and the `cost_aware:` banner line is correctly
  suppressed.

Not yet done: an actual B/C/E sweep using these flags -- this entry only
unblocks that, doesn't run it.
