# Archived code

This directory preserves code removed from the active validation package during
the publication cleanup. Nothing under `old_code` is imported by `core`,
`modules`, `experiments`, or `tests`.

- `coordinator_config.py` contains configuration dataclasses that were drafted
  for a coordinator refactor but never wired into `EnergyCoordinator`.
- `coordinator_legacy.py` contains unreachable helpers and the untested
  coordinate-descent path removed from `core/coordinator.py`.
- `observability_extensions.py` preserves telemetry collectors for sensitivity,
  escape, confidence, homotopy, and polynomial providers absent from this repo.
- `benchmark_presets.py` records benchmark options that never matched
  `EnergyCoordinator` constructor fields.
- `supports_redundancy.py` preserves an unused protocol from the information
  metrics module.

The archive is retained for recovery and comparison. Moving a feature back into
the active package requires an explicit API contract and focused tests.
