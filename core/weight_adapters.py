"""Weight adapter implementations for coordinating term balances.

Contains reactive adapters that satisfy the WeightAdapter protocol and can be
plugged into EnergyCoordinator to keep term contributions well-behaved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, TYPE_CHECKING

try:  # Optional dependency: torch is only required for GSPO-token adapter
    import torch
    from torch import nn
    import torch.optim as optim
except Exception:  # pragma: no cover - torch not installed
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    optim = None  # type: ignore[assignment]

from .interfaces import WeightAdapter
from .agm_metrics import compute_agm_phase_metrics


@dataclass
class GradNormWeightAdapter:
    """GradNorm-style balancing for energy terms.

    Args:
        target_norm: Desired L2 norm per term (defaults to 1.0).
        alpha: Restoring-force strength; larger => faster corrections.
        update_rate: Fraction of the adjustment applied each step (0-1].
        floor: Minimum allowed weight returned by the adapter.
        ceiling: Maximum allowed weight; set None to disable.
        eps: Numerical guard to avoid division by zero.
    """

    target_norm: float = 1.0
    alpha: float = 1.5
    update_rate: float = 0.1
    floor: float = 0.1
    ceiling: float | None = 2.0
    eps: float = 1e-9

    def __post_init__(self) -> None:
        assert self.target_norm > 0.0, "target_norm must be positive"
        assert 0.0 < self.update_rate <= 1.0, "update_rate must be in (0, 1]"
        assert self.floor >= 0.0, "floor must be non-negative"
        if self.ceiling is not None:
            assert self.ceiling >= self.floor, "ceiling must be >= floor"

    def step(
        self,
        term_grad_norms: Mapping[str, float],
        energy: float,  # noqa: ARG002 - required by protocol
        current: Mapping[str, float],
    ) -> Mapping[str, float]:
        assert term_grad_norms, "GradNormWeightAdapter requires gradient norms"
        updated: Dict[str, float] = {str(k): float(v) for k, v in current.items()}
        target = max(self.target_norm, self.eps)
        finite_norms = [float(n) for n in term_grad_norms.values() if self._is_valid(n)]
        average_norm = float(sum(finite_norms) / len(finite_norms)) if finite_norms else target
        for key, raw_norm in term_grad_norms.items():
            norm = float(raw_norm) if self._is_valid(raw_norm) else average_norm
            ratio = norm / target
            adjustment = self.alpha * (1.0 - ratio)
            weight = float(updated.get(key, 1.0))
            weight *= 1.0 + self.update_rate * adjustment
            weight = self._clamp(weight)
            updated[key] = weight
        return updated

    def _is_valid(self, value: float) -> bool:
        return value >= 0.0 and value == value  # excludes NaNs without importing math

    def _clamp(self, value: float) -> float:
        v = max(value, self.floor)
        if self.ceiling is not None:
            v = min(v, self.ceiling)
        return v


__all__ = ["GradNormWeightAdapter"]


@dataclass
class AGMPhaseWeightAdapter:
    """Phase-adaptive weighting using AGM-style metrics on energy history.

    Policy (simple, conservative):
      - If rate is high and trend positive: slightly increase coupling weights,
        slightly reduce gate local energy weight (exploitation).
      - If rate is low or oscillation high: slightly reduce coupling weights,
        slightly increase gate local energy weight (exploration/regularization).

    Adjustments are multiplicative and gentle to avoid violent swings.
    """

    increase_factor: float = 1.05
    decrease_factor: float = 0.97
    gate_local_key: str = "local:EnergyGatingModule"
    energy_history: List[float] = field(default_factory=list)

    def step(
        self,
        term_grad_norms: Mapping[str, float],  # noqa: ARG002 - protocol compatibility
        energy: float,
        current: Mapping[str, float],
    ) -> Mapping[str, float]:
        # Append energy for phase assessment
        self.energy_history.append(float(energy))
        metrics = compute_agm_phase_metrics(self.energy_history)
        updated: Dict[str, float] = {str(k): float(v) for k, v in current.items()}

        # Identify coupling keys (simple heuristic: prefix "coup:")
        coupling_keys = [k for k in updated.keys() if k.startswith("coup:")]

        # Decide a regime
        rate = metrics["rate"]
        trend = metrics["trend"]
        oscillation = metrics["oscillation"]

        if rate > 0.7 and trend > 0.0 and oscillation < 0.05:
            # Exploitation: favor couplings, soften gate local slightly
            for k in coupling_keys:
                updated[k] = updated.get(k, 1.0) * self.increase_factor
            updated[self.gate_local_key] = updated.get(self.gate_local_key, 1.0) * self.decrease_factor
        elif rate < 0.3 or oscillation > 0.1:
            # Exploration or unstable: favor regularization via gate local, tame couplings
            for k in coupling_keys:
                updated[k] = updated.get(k, 1.0) * self.decrease_factor
            updated[self.gate_local_key] = updated.get(self.gate_local_key, 1.0) * self.increase_factor
        # Else keep weights unchanged

        return updated


__all__.extend(["AGMPhaseWeightAdapter"])





@dataclass
class SmallGainWeightAdapter:
    """Per-edge stability-margin allocator with row-aware greedy budgeting.

    Production intent: conservative, monotone-compatible allocator that spends a fraction
    of the available contractivity budget (global and per-row) to boost coupling families
    with the highest expected ΔF per ΔL ratio.

    Coordinator integration:
      - Before calling step(...), the coordinator should populate:
          self.edge_costs: Dict[str, float]    # ΔL per coupling key (e.g., 'coup:ClassName')
          self.row_margins: Dict[int, float]   # per-row margins m_r
          self.global_margin: float            # global margin
      - These are treated as snapshots for the current step only.

    Bounded updates and smoothing:
      - Per-step weight change is limited to ±max_step_change.
      - Scores (value/cost) use EMA to reduce noise.

    Observability:
      - last_allocations: Dict[str, float] of Δweight applied per key (for logging)
      - last_spent_global: float
      - last_spent_row: Dict[int, float]
    """

    # Budgeting
    budget_fraction: float = 0.7
    max_step_change: float = 0.10

    # Bounds and smoothing
    floor: float = 0.1
    ceiling: float = 3.0
    ema_alpha: float = 0.3
    eps: float = 1e-9

    # Snapshots injected by coordinator per step
    edge_costs: Dict[str, float] = field(default_factory=dict)      # ΔL per coupling key
    row_margins: Dict[int, float] = field(default_factory=dict)     # per-row margins
    global_margin: float = 0.0

    # Adapter state
    scores: Dict[str, float] = field(default_factory=dict)          # EMA(value/cost)

    # Observability
    last_allocations: Dict[str, float] = field(default_factory=dict)
    last_spent_global: float = 0.0
    last_spent_row: Dict[int, float] = field(default_factory=dict)

    def step(
        self,
        term_grad_norms: Mapping[str, float],
        energy: float,  # noqa: ARG002 - reserved for future use / compatibility
        current: Mapping[str, float],
    ) -> Mapping[str, float]:
        # Reset observability
        self.last_allocations = {}
        self.last_spent_global = 0.0
        self.last_spent_row = {r: 0.0 for r in self.row_margins.keys()}

        # Build values for coupling families that exist in term_grad_norms
        values: Dict[str, float] = {}
        for k, v in term_grad_norms.items():
            if isinstance(k, str) and k.startswith("coup:"):
                try:
                    values[k] = float(v) * float(v)  # grad_norm^2
                except Exception:
                    continue

        if not values:
            # Nothing to do
            return dict(current)

        # Costs per family (ΔL per unit relative scaling)
        costs: Dict[str, float] = {}
        for k in values.keys():
            c = self.edge_costs.get(k, 0.0)
            costs[k] = float(c) if c == c else 0.0  # guard NaN

        # Update EMA scores (value/cost)
        for k in values.keys():
            denom = costs[k] + self.eps
            raw = values[k] / denom if denom > 0.0 else values[k]
            old = self.scores.get(k, raw)
            self.scores[k] = self.ema_alpha * raw + (1.0 - self.ema_alpha) * old

        # Rank by score
        ranked = sorted(values.keys(), key=lambda kk: self.scores.get(kk, 0.0), reverse=True)

        # Budgets
        row_budget = {r: max(0.0, float(m)) * self.budget_fraction for r, m in self.row_margins.items()}
        global_budget = max(0.0, float(self.global_margin)) * self.budget_fraction

        # Prepare output mapping
        updated: Dict[str, float] = {str(k): float(v) for k, v in current.items()}

        # Greedy allocation
        for k in ranked:
            # Stop if global budget exhausted
            if global_budget - self.last_spent_global <= 0.0:
                break

            fam_key = k
            w_old = float(updated.get(fam_key, 1.0))
            w_new = w_old

            # Propose an increase within bounds
            proposed = min(self.ceiling, max(self.floor, w_old * (1.0 + self.max_step_change)))
            delta_w = proposed - w_old
            if delta_w <= 0.0:
                continue

            # Linearized ΔL spend for this increment
            cost = costs.get(k, 0.0)
            # Approximate proportionality of ΔL with relative scaling
            denom_scale = max(w_old, self.eps)
            delta_L = cost * (delta_w / denom_scale)

            # Check global budget
            if self.last_spent_global + delta_L > global_budget:
                continue

            # For a first implementation, we do not enforce per-row incidence booking here,
            # as row attribution requires edge->row mapping. Keep global cap conservative.
            # Future: accept an injected edge->rows map and update self.last_spent_row accordingly.

            # Commit
            w_new = proposed
            updated[fam_key] = w_new
            self.last_allocations[fam_key] = delta_w
            self.last_spent_global += delta_L

        return updated


__all__.extend(["SmallGainWeightAdapter"])


def _maybe_import_gspo_backend() -> tuple[Any, Callable] | tuple[None, None]:
    """Lazy import for GSPO trainer utilities."""

    try:
        from .gspo_token_vectorized import (  # type: ignore
            GSPOTokenConfig,
            run_gspo_token_vectorized_step,
        )
    except Exception:  # pragma: no cover - torch or torch deps missing
        return None, None
    return GSPOTokenConfig, run_gspo_token_vectorized_step


@dataclass
class GSPOTokenWeightAdapter:
    """GSPO-token-driven adapter that learns term weights via sequence policy optimization.

    This adapter treats the per-step term gradient norms as a prompt and trains a tiny
    sequence policy (GRU + linear head) using the in-repo GSPO-token trainer. Each step:

      1. Encode gradient ratios into discrete tokens (prompt).
      2. Sample multiple candidate weight sequences via GSPO-token (token-level advantages).
      3. Reward = negative absolute error compared to GradNorm-style target weights.
      4. Apply the greedy decoded weights (smoothed by `apply_rate`) back to the coordinator.

    **Usage caveats (read before deploying):**
      - **Outer-loop only**: This adapter runs a mini RL training step (sampling, forward passes,
        backward passes) at each coordinator relaxation step. Intended for meta-training / weight
        search (e.g., 30-100 relaxation steps). NOT suitable for tight inner loops with 1000s of
        steps unless you throttle update frequency (e.g., skip_steps=10).
      - **Architecture capacity**: The default `hidden_size=64` GRU is sufficient for 2-10 term
        families. For 10-20+ terms, consider increasing `hidden_size` to 128-256 or switching to
        a transformer-based policy head (future work; see P4 in docs/fixes_and__related_todos.md).
      - **Compute cost**: Expect 5-20x slowdown vs GradNormWeightAdapter per relaxation step,
        depending on `group_size` and `batch_size`. Profile with small `batch_size=1, group_size=2`
        before scaling up.

    Requirements:
        * PyTorch must be installed (see `pip install .[torch]`).
        * `core/gspo_token_vectorized.py` is lazily imported; failures bubble with context.
    """

    target_norm: float = 1.0
    floor: float = 0.05
    ceiling: float = 4.0
    num_buckets: int = 16
    max_ratio: float = 4.0
    group_size: int = 4
    batch_size: int = 2
    learning_rate: float = 5e-4
    hidden_size: int = 64
    apply_rate: float = 0.35
    use_token_level: bool = True
    device: Optional[str] = None
    reference_sync_interval: int = 10
    update_every_n_steps: int = 1  # Throttle RL updates; decode-only on skipped steps
    ema_reference_alpha: Optional[float] = None  # If set, EMA-sync reference each update (0<α<1)
    logging_callback: Optional[Callable[[Dict[str, float]], None]] = None  # Hook for dashboards
    enable_throttling: bool = True  # If False, always train every step regardless of update_every_n_steps

    _policy_model: Any = field(default=None, init=False, repr=False)
    _reference_model: Any = field(default=None, init=False, repr=False)
    _optimizer: Any = field(default=None, init=False, repr=False)
    _config: Any = field(default=None, init=False, repr=False)
    _prompt_len: int = field(default=0, init=False, repr=False)
    _response_len: int = field(default=0, init=False, repr=False)
    _device_str: str = field(default="cpu", init=False, repr=False)
    _step_counter: int = field(default=0, init=False, repr=False)
    _last_weights: Dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _gspo_config_cls: Any = field(default=None, init=False, repr=False)
    _gspo_step_fn: Optional[Callable[..., Dict[str, float]]] = field(default=None, init=False, repr=False)
    _last_metrics: Optional[Dict[str, float]] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.floor <= 0.0:
            raise ValueError("floor must be positive for GSPO-token adapter")
        if self.ceiling <= self.floor:
            raise ValueError("ceiling must exceed floor")
        if self.num_buckets < 4:
            raise ValueError("num_buckets must be >= 4")
        if self.batch_size <= 0 or self.group_size <= 0:
            raise ValueError("batch_size and group_size must be positive")
        self._device_str = self.device or self._default_device()

    # ------------------------------------------------------------------
    def step(
        self,
        term_grad_norms: Mapping[str, float],
        energy: float,  # noqa: ARG002 - compatibility
        current: Mapping[str, float],
    ) -> Mapping[str, float]:
        if not term_grad_norms:
            return dict(current)
        keys = sorted(term_grad_norms.keys())
        self._ensure_backend(len(keys))
        prompt_tokens = self._encode_prompt(keys, term_grad_norms)
        prompts = prompt_tokens.repeat(self.batch_size, 1)
        desired_weights = torch.tensor(
            [self._desired_weight(term_grad_norms[k]) for k in keys],
            dtype=torch.float32,
            device=self._device_str,
        )
        self._step_counter += 1

        did_train = False
        metrics: Dict[str, float] = {}
        # Throttling: only train every N steps (unless disabled)
        should_train = (not self.enable_throttling) or (self.update_every_n_steps <= 1) or (self._step_counter % self.update_every_n_steps == 0)
        if should_train:
            reward_fn = self._build_reward_fn(desired_weights)
            metrics = self._gspo_step_fn(  # type: ignore[call-arg]
                policy_model=self._policy_model,
                reference_model=self._reference_model,
                optimizer=self._optimizer,
                prompts=prompts,
                reward_fn=reward_fn,
                config=self._config,
            )
            did_train = True
            # EMA reference sync if configured
            if isinstance(self.ema_reference_alpha, (float, int)) and 0.0 < float(self.ema_reference_alpha) < 1.0:
                self._ema_update_reference(float(self.ema_reference_alpha))
            # Periodic full sync as a stabilizer
            if self._step_counter % self.reference_sync_interval == 0:
                self._reference_model.load_state_dict(self._policy_model.state_dict())
        else:
            # Skipped training: mark metrics
            metrics = {"skipped": 1.0}

        self._last_metrics = metrics

        # Decode greedy weights for actual application
        decoded_tokens = self._greedy_decode(prompt_tokens[:1])
        decoded_weights = self._tokens_to_weights(decoded_tokens).squeeze(0).tolist()

        updated: Dict[str, float] = dict(current)
        for key, weight in zip(keys, decoded_weights):
            prev = float(updated.get(key, 1.0))
            blended = (1.0 - self.apply_rate) * prev + self.apply_rate * weight
            updated[key] = float(self._clamp(blended))
        self._last_weights = updated
        # Optional logging hook
        if self.logging_callback is not None:
            try:
                self.logging_callback({"did_train": 1.0 if did_train else 0.0, **metrics})
            except Exception:
                pass
        return updated

    # ------------------------------------------------------------------
    def _default_device(self) -> str:
        if torch is not None and torch.cuda.is_available():  # pragma: no cover - GPU optional
            return "cuda"
        return "cpu"

    def _ema_update_reference(self, alpha: float) -> None:
        """EMA update of reference parameters toward policy parameters."""
        with torch.no_grad():
            for ref_p, pol_p in zip(self._reference_model.parameters(), self._policy_model.parameters()):
                ref_p.data.mul_(alpha).add_(pol_p.data, alpha=1.0 - alpha)

    def _ensure_backend(self, num_terms: int) -> None:
        if torch is None or nn is None or optim is None:
            raise ImportError(
                "GSPOTokenWeightAdapter requires PyTorch. "
                "Install with `pip install .[torch]` or disable the adapter."
            )
        if self._gspo_config_cls is None or self._gspo_step_fn is None:
            config_cls, step_fn = _maybe_import_gspo_backend()
            if config_cls is None or step_fn is None:
                raise ImportError(
                    "Failed to import GSPO trainer utilities. Ensure torch is installed and "
                    "core/gspo_token_vectorized.py is available."
                )
            self._gspo_config_cls = config_cls
            self._gspo_step_fn = step_fn
        if (
            self._policy_model is None
            or self._response_len != num_terms
            or self._prompt_len != num_terms
        ):
            self._build_models(num_terms)

    def _build_models(self, num_terms: int) -> None:
        vocab_size = self.num_buckets + 2  # buckets + EOS + PAD
        prompt_len = num_terms
        response_len = num_terms
        config = self._gspo_config_cls(
            group_size=self.group_size,
            epsilon=0.2,
            max_length=prompt_len + response_len + 1,
            eos_token=self.num_buckets,
            pad_token=self.num_buckets + 1,
            use_token_level=self.use_token_level,
        )

        class _WeightPolicy(nn.Module):
            def __init__(self, vocab: int, hidden: int) -> None:
                super().__init__()
                self.embed = nn.Embedding(vocab, hidden)
                self.gru = nn.GRU(hidden, hidden, batch_first=True)
                self.head = nn.Linear(hidden, vocab)

            def forward(self, tokens: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
                emb = self.embed(tokens)
                out, _ = self.gru(emb)
                return self.head(out)

        policy = _WeightPolicy(vocab_size, self.hidden_size).to(self._device_str)
        reference = _WeightPolicy(vocab_size, self.hidden_size).to(self._device_str)
        reference.load_state_dict(policy.state_dict())
        optimizer = optim.Adam(policy.parameters(), lr=self.learning_rate)

        self._policy_model = policy
        self._reference_model = reference
        self._optimizer = optimizer
        self._config = config
        self._prompt_len = prompt_len
        self._response_len = response_len

    def _encode_prompt(self, keys: List[str], norms: Mapping[str, float]) -> torch.Tensor:
        encoded = [self._bucketize_ratio(float(norms[k])) for k in keys]
        tensor = torch.tensor([encoded], dtype=torch.long, device=self._device_str)
        return tensor

    def _bucketize_ratio(self, value: float) -> int:
        target = max(self.target_norm, 1e-8)
        ratio = min(max(value / target, 0.0), self.max_ratio)
        bucket = round((ratio / self.max_ratio) * (self.num_buckets - 1))
        return int(max(0, min(self.num_buckets - 1, bucket)))

    def _desired_weight(self, grad_norm: float) -> float:
        denom = max(grad_norm, 1e-8)
        raw = self.target_norm / denom
        return self._clamp(raw)

    def _clamp(self, value: float) -> float:
        return float(max(self.floor, min(self.ceiling, value)))

    def _build_reward_fn(self, desired: torch.Tensor) -> Callable[[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
        prompt_len = self._prompt_len
        response_len = self._response_len
        device = self._device_str

        def reward_fn(_: torch.Tensor, responses: torch.Tensor) -> Dict[str, torch.Tensor]:
            resp_tokens = responses[:, -response_len:]
            decoded = self._tokens_to_weights(resp_tokens)
            diff = torch.abs(decoded - desired.unsqueeze(0))
            rewards = -diff.mean(dim=1)
            if self.use_token_level:
                token_adv = torch.zeros_like(responses, dtype=torch.float32, device=device)
                token_adv[:, -response_len:] = -diff
                return {"rewards": rewards, "token_advantages": token_adv}
            return {"rewards": rewards}

        return reward_fn

    def _tokens_to_weights(self, token_tensor: torch.Tensor) -> torch.Tensor:
        clamped = token_tensor.clamp(0, self.num_buckets - 1).float()
        frac = clamped / max(1, self.num_buckets - 1)
        return self.floor + frac * (self.ceiling - self.floor)

    def _greedy_decode(self, prompt: torch.Tensor) -> torch.Tensor:
        seq = prompt.clone()
        for _ in range(self._response_len):
            logits = self._policy_model(seq)[:, -1, :]
            token = torch.argmax(logits, dim=-1)
            seq = torch.cat([seq, token.unsqueeze(1)], dim=1)
        return seq[:, -self._response_len:]


__all__.extend(["GSPOTokenWeightAdapter"])
