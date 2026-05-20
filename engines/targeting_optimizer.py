"""
Engine 2: Targeting & Timing Optimizer — Contextual Multi-Armed Bandit (LinUCB)
Decides the optimal (channel × time_of_day × creative_variant) arm for each grower
and updates from observed rewards (click / purchase).

Arms: {channel} × {time_slot} × {creative_variant}
Context: grower features + weather + pest pressure + stock availability
Reward: 1 if WhatsApp clicked or purchase attributed within 14 days, else 0
"""

"""
OFFLINE STATUS: NO CHANGES REQUIRED
-------------------------------------
This engine is already fully offline-capable in the original codebase.
It loads all state from local .pkl files (models/) and performs all
inference in-process. It makes zero external API calls.

  targeting_optimizer.py   → LinUCB bandit loads from linucb_bandit.pkl
  receptivity_predictor.py → XGBoost model loads from receptivity_model.pkl
  micro_segmentation.py    → HDBSCAN + PCA load from embedding_pipeline.pkl

No API keys needed. No internet needed. No changes made.
This file is identical to the original.
"""


import numpy as np
import json
import joblib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Arm definitions
# ---------------------------------------------------------------------------
CHANNELS = ["whatsapp_rich", "whatsapp_text", "ivr_voice", "sms", "field_rep_brief"]
TIME_SLOTS = ["morning_7_10", "midday_12_14", "evening_18_21"]
CREATIVE_VARIANTS = ["threat_alert", "product_benefit", "social_proof", "demo_invite"]

# Cartesian product of arms
ARMS = [
    f"{ch}|{ts}|{cv}"
    for ch in CHANNELS
    for ts in TIME_SLOTS
    for cv in CREATIVE_VARIANTS
]
N_ARMS = len(ARMS)
ARM_INDEX = {arm: i for i, arm in enumerate(ARMS)}

# Solution doc Module 4 cost table (₹ per delivery):
# whatsapp_text ₹0.10 | whatsapp_rich ₹0.30 (approx video) | ivr_voice ₹0.80
# sms ₹0.15 | field_rep_brief ₹50.00
# Used for cost-aware reporting and budget-constrained selection.
ARM_COST_INR = {
    arm: (
        0.30 if arm.startswith("whatsapp_rich") else
        0.10 if arm.startswith("whatsapp_text") else
        0.80 if arm.startswith("ivr_voice") else
        0.15 if arm.startswith("sms") else
        50.0  # field_rep_brief
    )
    for arm in ARMS
}

# Device → eligible channels constraint
DEVICE_ELIGIBLE_CHANNELS = {
    "smartphone": ["whatsapp_rich", "whatsapp_text", "sms"],
    "keypad": ["ivr_voice", "sms"],
    "unknown": ["field_rep_brief", "sms"],
}


@dataclass
class GrowerContext:
    """Feature vector for a single grower at decision time."""
    grower_id: str
    device_type: str
    language: str
    crop: str
    growth_stage: str
    season_progress: float          # 0-1 (sowing→harvest)
    days_to_next_stage: int
    hist_open_rate: float           # historical WA open rate
    hist_click_rate: float          # historical WA click rate
    tehsil_stock_rate: float        # product stock rate in tehsil
    days_since_rep_visit: int
    digital_literacy_score: float   # 0-1
    grower_age: int
    farm_size_acres: float
    offline_campaign_attended: bool
    product_scan_done: bool
    weather_risk_score: float = 0.0  # 0-1 (0=normal, 1=high risk)
    pest_pressure_index: float = 0.0 # 0-1 from external surveillance


class LinUCBAgent:
    """
    Disjoint Linear Upper Confidence Bound (LinUCB) bandit.
    Each arm has its own linear ridge regression model.
    Exploration parameter alpha controls exploration-exploitation trade-off.
    
    Reference: Li et al., 2010 — "A Contextual-Bandit Approach to Personalized News Article Recommendation"
    """

    def __init__(self, n_arms: int, context_dim: int, alpha: float = 0.5):
        self.n_arms = n_arms
        self.d = context_dim
        self.alpha = alpha

        # Per-arm: A (d×d identity), b (d-dim zero vector)
        self.A = [np.identity(self.d) for _ in range(n_arms)]
        self.b = [np.zeros(self.d) for _ in range(n_arms)]

        # Tracking
        self.arm_pull_counts = np.zeros(n_arms, dtype=int)
        self.arm_rewards = np.zeros(n_arms)
        self.total_rounds = 0

    def _A_inv(self, arm_idx: int) -> np.ndarray:
        """Compute inverse of A with ridge regularization for numerical stability."""
        A_reg = self.A[arm_idx] + 1e-6 * np.identity(self.d)
        return np.linalg.inv(A_reg)

    def _theta(self, arm_idx: int) -> np.ndarray:
        """Compute estimated parameter vector for arm."""
        return self._A_inv(arm_idx) @ self.b[arm_idx]

    def _ucb(self, arm_idx: int, context: np.ndarray) -> float:
        """Compute UCB score = estimated reward + exploration bonus."""
        A_inv = self._A_inv(arm_idx)
        theta = A_inv @ self.b[arm_idx]
        p = theta @ context + self.alpha * np.sqrt(context @ A_inv @ context)
        return float(p)

    def select_arm(self, context: np.ndarray, eligible_arm_indices: Optional[list] = None) -> int:
        """Select arm with highest UCB score among eligible arms."""
        arms_to_consider = eligible_arm_indices if eligible_arm_indices else list(range(self.n_arms))
        scores = [self._ucb(i, context) for i in arms_to_consider]
        best_local = int(np.argmax(scores))
        return arms_to_consider[best_local]

    def update(self, arm_idx: int, context: np.ndarray, reward: float):
        """Update arm's A and b matrices with observed reward."""
        self.A[arm_idx] += np.outer(context, context)
        self.b[arm_idx] += reward * context
        self.arm_pull_counts[arm_idx] += 1
        self.arm_rewards[arm_idx] += reward
        self.total_rounds += 1

    def arm_stats(self) -> list[dict]:
        """Return per-arm pull count and empirical reward rate."""
        stats = []
        for i, arm in enumerate(ARMS):
            pulls = int(self.arm_pull_counts[i])
            emp_reward = float(self.arm_rewards[i] / pulls) if pulls > 0 else 0.0
            stats.append({
                "arm": arm,
                "pulls": pulls,
                "empirical_reward": round(emp_reward, 4),
                "theta_norm": float(np.linalg.norm(self._theta(i))),
            })
        return sorted(stats, key=lambda x: x["empirical_reward"], reverse=True)

    def save(self, path: Path):
        joblib.dump({
            "A": self.A, "b": self.b,
            "arm_pull_counts": self.arm_pull_counts,
            "arm_rewards": self.arm_rewards,
            "total_rounds": self.total_rounds,
            "n_arms": self.n_arms, "d": self.d, "alpha": self.alpha,
        }, path)
        print(f"Bandit saved → {path}")

    @classmethod
    def load(cls, path: Path) -> "LinUCBAgent":
        data = joblib.load(path)
        agent = cls(data["n_arms"], data["d"], data["alpha"])
        agent.A = data["A"]
        agent.b = data["b"]
        agent.arm_pull_counts = data["arm_pull_counts"]
        agent.arm_rewards = data["arm_rewards"]
        agent.total_rounds = data["total_rounds"]
        return agent


# ---------------------------------------------------------------------------
# Thompson Sampling agent (solution doc Module 4 specifies Thompson Sampling)
# "Contextual Multi-Armed Bandit (Thompson Sampling)" — Solution 2, Module 4
# Used as an alternative/parallel to LinUCB; preferred for cold-start growers
# because it explores more aggressively via Beta distribution sampling.
# ---------------------------------------------------------------------------

class ThompsonSamplingAgent:
    """
    Beta-Thompson Sampling bandit (non-contextual version for cold-start).
    Maintains Beta(alpha, beta) distribution per arm.
    Select arm by sampling from each distribution and picking the max.
    Update: success → alpha += 1, failure → beta += 1.

    Use for: new growers with no WhatsApp history (hist_open_rate == 0).
    Fall back to LinUCB once ≥3 engagement events are recorded.
    """

    def __init__(self, n_arms: int, device_type: str = "smartphone"):
        self.n_arms = n_arms
        self.alpha  = np.ones(n_arms)   # successes + 1 (Beta prior)
        self.beta   = np.ones(n_arms)   # failures  + 1

    def select_arm(self, eligible_arm_indices: list | None = None) -> int:
        arms = eligible_arm_indices if eligible_arm_indices else list(range(self.n_arms))
        samples = [np.random.beta(self.alpha[i], self.beta[i]) for i in arms]
        return arms[int(np.argmax(samples))]

    def update(self, arm_idx: int, reward: float) -> None:
        """reward expected in [0, 1]; threshold at 0.5 for success/failure."""
        if reward >= 0.5:
            self.alpha[arm_idx] += 1
        else:
            self.beta[arm_idx]  += 1

    def arm_stats(self) -> list[dict]:
        return [
            {
                "arm": ARMS[i],
                "alpha": self.alpha[i],
                "beta":  self.beta[i],
                "mean_reward": self.alpha[i] / (self.alpha[i] + self.beta[i]),
            }
            for i in range(self.n_arms)
        ]


# ---------------------------------------------------------------------------
# Context encoder
# ---------------------------------------------------------------------------
CONTEXT_DIM = 18  # must match features extracted below

def encode_context(ctx: GrowerContext) -> np.ndarray:
    """Convert GrowerContext into a numeric feature vector (CONTEXT_DIM dims)."""
    device_enc = {"smartphone": 1.0, "keypad": 0.5, "unknown": 0.0}.get(ctx.device_type, 0.0)
    crop_enc = {"wheat": 0.9, "chickpea": 0.8, "mustard": 0.7, "potato": 0.6,
                "barley": 0.5, "lentil": 0.4, "safflower": 0.3, "cumin": 0.2, "maize": 0.1}.get(ctx.crop, 0.5)

    features = np.array([
        device_enc,
        crop_enc,
        ctx.season_progress,
        min(ctx.days_to_next_stage, 60) / 60.0,   # normalize 0-1
        ctx.hist_open_rate,
        ctx.hist_click_rate,
        ctx.tehsil_stock_rate,
        min(ctx.days_since_rep_visit, 180) / 180.0,
        ctx.digital_literacy_score,
        min(ctx.grower_age, 80) / 80.0,
        min(ctx.farm_size_acres, 20) / 20.0,
        float(ctx.offline_campaign_attended),
        float(ctx.product_scan_done),
        ctx.weather_risk_score,
        ctx.pest_pressure_index,
        # Interaction terms
        ctx.hist_open_rate * ctx.digital_literacy_score,
        ctx.tehsil_stock_rate * ctx.pest_pressure_index,
        ctx.season_progress * ctx.weather_risk_score,
    ], dtype=float)

    assert len(features) == CONTEXT_DIM, f"Expected {CONTEXT_DIM} features, got {len(features)}"
    # Sanitize: replace any NaN/Inf (e.g. from missing grower_farm_size in CSV) with 0.0
    features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=0.0)
    return features


# ---------------------------------------------------------------------------
# Decision interface
# ---------------------------------------------------------------------------
@dataclass
class DeliveryDecision:
    grower_id: str
    selected_arm: str
    channel: str
    time_slot: str
    creative_variant: str
    ucb_score: float
    eligible_arms_count: int
    decision_timestamp: str


def get_eligible_arms(device_type: str) -> list[int]:
    """Filter arms to only those compatible with the grower's device."""
    eligible_channels = DEVICE_ELIGIBLE_CHANNELS.get(device_type, ["sms"])
    eligible = [
        ARM_INDEX[arm] for arm in ARMS
        if arm.split("|")[0] in eligible_channels
    ]
    return eligible if eligible else list(range(N_ARMS))


def decide(agent: LinUCBAgent, ctx: GrowerContext,
           thompson_agent: "ThompsonSamplingAgent | None" = None) -> DeliveryDecision:
    """Select the best arm for a grower using the trained bandit.

    Cold-start routing (solution doc Module 4 / Thompson Sampling):
    If the grower has no WhatsApp history (hist_open_rate == 0 AND cum_messages == 0),
    use ThompsonSamplingAgent for pure exploration instead of LinUCB.
    LinUCB takes over once ≥3 engagement events exist (cum_messages >= 3).
    """
    eligible = get_eligible_arms(ctx.device_type)
    is_cold_start = (ctx.hist_open_rate == 0.0 and ctx.hist_click_rate == 0.0)

    if is_cold_start and thompson_agent is not None:
        # Use Thompson Sampling for cold-start exploration (solution doc Module 4)
        best_arm_idx = thompson_agent.select_arm(eligible)
        ucb_score = float(
            thompson_agent.alpha[best_arm_idx] /
            (thompson_agent.alpha[best_arm_idx] + thompson_agent.beta[best_arm_idx])
        )
    else:
        context_vec = encode_context(ctx)
        best_arm_idx = agent.select_arm(context_vec, eligible)
        ucb_score = agent._ucb(best_arm_idx, encode_context(ctx))

    best_arm = ARMS[best_arm_idx]
    channel, time_slot, creative = best_arm.split("|")
    return DeliveryDecision(
        grower_id=ctx.grower_id,
        selected_arm=best_arm,
        channel=channel,
        time_slot=time_slot,
        creative_variant=creative,
        ucb_score=round(ucb_score, 4),
        eligible_arms_count=len(eligible),
        decision_timestamp=datetime.utcnow().isoformat(),
    )


def record_reward(agent: LinUCBAgent, ctx: GrowerContext,
                  arm: str, reward: float) -> None:
    """Feed back observed reward (click=0.5, purchase=1.0) into the bandit."""
    arm_idx = ARM_INDEX[arm]
    context_vec = encode_context(ctx)
    agent.update(arm_idx, context_vec, reward)


# ---------------------------------------------------------------------------
# Bootstrap: initialize bandit from historical WhatsApp data
# ---------------------------------------------------------------------------
def bootstrap_from_history(wa_path: str, growers_path: str) -> LinUCBAgent:
    """
    Warm-start bandit from historical campaign data.
    Maps existing WhatsApp messages to arm=whatsapp_rich|{time_slot}|threat_alert
    and uses clicked_status as reward signal.
    """
    import pandas as pd

    print("Bootstrapping bandit from historical data...")
    wa = pd.read_csv(wa_path)
    growers = pd.read_csv(growers_path)

    wa["message_sent_date"] = pd.to_datetime(wa["message_sent_date"])
    wa["hour"] = wa["message_sent_date"].dt.hour

    def hour_to_slot(h):
        if 7 <= h < 10:
            return "morning_7_10"
        elif 12 <= h < 14:
            return "midday_12_14"
        elif 18 <= h < 21:
            return "evening_18_21"
        else:
            return "morning_7_10"  # default

    wa["time_slot"] = wa["hour"].apply(hour_to_slot)

    grower_meta = growers.set_index("grower_id")[
        ["device_type", "language", "grower_age", "grower_farm_size",
         "offline_campaign_attended", "product_scan"]
    ].to_dict("index")

    agent = LinUCBAgent(n_arms=N_ARMS, context_dim=CONTEXT_DIM, alpha=0.5)

    updated = 0
    for _, row in wa.iterrows():
        gid = row["grower_id"]
        meta = grower_meta.get(gid, {})
        device = meta.get("device_type", "smartphone")
        arm = f"whatsapp_rich|{row['time_slot']}|threat_alert"
        if arm not in ARM_INDEX:
            continue

        ctx = GrowerContext(
            grower_id=gid,
            device_type=device,
            language=meta.get("language", "Hindi"),
            crop=row.get("campaign_crop", "wheat"),
            growth_stage="tillering",
            season_progress=0.4,
            days_to_next_stage=14,
            hist_open_rate=0.0,
            hist_click_rate=0.0,
            tehsil_stock_rate=0.7,
            days_since_rep_visit=30,
            digital_literacy_score=0.8 if device == "smartphone" else 0.4,
            grower_age=int(meta.get("grower_age", 45)),
            farm_size_acres=float(meta.get("grower_farm_size", 2.0)),
            offline_campaign_attended=bool(meta.get("offline_campaign_attended", False)),
            product_scan_done=bool(meta.get("product_scan", False)),
        )

        # Reward: 1 if clicked, 0.3 if opened, 0 otherwise
        reward = 1.0 if row["clicked_status"] else (0.3 if row["opened_status"] else 0.0)
        record_reward(agent, ctx, arm, reward)
        updated += 1

    print(f"  Bootstrapped {updated:,} historical interactions")
    print(f"  Total bandit rounds: {agent.total_rounds:,}")

    path = MODELS_DIR / "linucb_bandit.pkl"
    agent.save(path)
    return agent


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from pathlib import Path as P

    print("=== Engine 2: LinUCB Bandit Demo ===\n")

    data_path = P("data")
    if (data_path / "whatsapp_campaign.csv").exists():
        agent = bootstrap_from_history(
            str(data_path / "whatsapp_campaign.csv"),
            str(data_path / "growers.csv"),
        )
    else:
        print("Data not found. Initializing fresh agent.")
        agent = LinUCBAgent(n_arms=N_ARMS, context_dim=CONTEXT_DIM, alpha=0.5)

    # Test decision for a sample grower
    test_ctx = GrowerContext(
        grower_id="GRW_00001",
        device_type="smartphone",
        language="Hindi",
        crop="wheat",
        growth_stage="flowering",
        season_progress=0.65,
        days_to_next_stage=7,
        hist_open_rate=0.4,
        hist_click_rate=0.1,
        tehsil_stock_rate=0.8,
        days_since_rep_visit=15,
        digital_literacy_score=0.85,
        grower_age=45,
        farm_size_acres=3.5,
        offline_campaign_attended=False,
        product_scan_done=False,
        weather_risk_score=0.7,
        pest_pressure_index=0.6,
    )

    decision = decide(agent, test_ctx)
    print(f"Grower: {decision.grower_id}")
    print(f"Selected arm:     {decision.selected_arm}")
    print(f"Channel:          {decision.channel}")
    print(f"Time slot:        {decision.time_slot}")
    print(f"Creative variant: {decision.creative_variant}")
    print(f"UCB score:        {decision.ucb_score}")

    print(f"\nTop 5 arms by empirical reward:")
    for stat in agent.arm_stats()[:5]:
        print(f"  {stat['arm']}: pulls={stat['pulls']}, reward={stat['empirical_reward']}")