
import math
from dataclasses import dataclass
from typing import Dict, Tuple, List
import numpy as np  # for Monte Carlo Poisson sampling


# ================================================================
#  Basic math utilities (Poisson)
# ================================================================

def poisson_pmf(k: int, lam: float) -> float:
    """Poisson probability P(X = k) with mean lam."""
    return math.exp(-lam) * lam**k / math.factorial(k)


def poisson_distribution(
    lam: float,
    max_k: int | None = None,
    cutoff: float = 0.999
) -> List[Tuple[int, float]]:
    """
    Return [(k, p_k)] for a Poisson(lam) until:
      - k reaches max_k, OR
      - cumulative probability >= cutoff.
    """
    dist: List[Tuple[int, float]] = []
    cum = 0.0
    k = 0
    while True:
        p = poisson_pmf(k, lam)
        dist.append((k, p))
        cum += p
        k += 1
        if (max_k is not None and k > max_k) or (max_k is None and cum >= cutoff):
            break
    return dist


def prob_at_least(k_min: int, lam: float) -> float:
    """P(X >= k_min) for X ~ Poisson(lam)."""
    return 1.0 - sum(poisson_pmf(k, lam) for k in range(k_min))


# ================================================================
#  Config structures
# ================================================================

@dataclass
class SetConfig:
    """
    Config for a Lorcana set.

    rarity_card_counts:
        number of distinct cards of each rarity (N_R).
        e.g. {"common": 72, "uncommon": 54, ...}

    expected_per_pack:
        expected number of *cards* of each rarity in ONE booster pack (E_R).
        This is per *any* card of that rarity, not per unique.
    """
    set_number: int
    name: str
    rarity_card_counts: Dict[str, int]
    expected_per_pack: Dict[str, float]


# ================================================================
#  Global seeding assumptions
# ================================================================

# Rare+ slot seeding (for every set with standard layout):
RARE_PLUS_SLOTS_PER_PACK = 2
P_RARE_PER_SLOT       = 8.0 / 12.0
P_SUPER_RARE_PER_SLOT = 3.0 / 12.0
P_LEGENDARY_PER_SLOT  = 1.0 / 12.0

E_RARE_FROM_SLOTS       = RARE_PLUS_SLOTS_PER_PACK * P_RARE_PER_SLOT       # ≈ 1.3333
E_SUPER_RARE_FROM_SLOTS = RARE_PLUS_SLOTS_PER_PACK * P_SUPER_RARE_PER_SLOT # = 0.5
E_LEGENDARY_FROM_SLOTS  = RARE_PLUS_SLOTS_PER_PACK * P_LEGENDARY_PER_SLOT  # ≈ 0.1667

# Foil-slot seeded rules (per PACK, "any card of that rarity"):
ENCHANTED_PER_PACK_ANY = 1.0 / 96.0          # any Enchanted
ICONIC_PER_PACK_ANY    = 1.0 / 960.0         # any Iconic (sets 9–10 only)
EPIC_PER_PACK_ANY      = 1.0 / 16.0          # any Epic (foil slot only, sets 9–10)


# ================================================================
#  Set rarity counts (1–10)
# ================================================================

SET_RARITY_COUNTS: Dict[int, Dict] = {
    1: {
        "name": "The First Chapter",
        "rarity_card_counts": {
            "common": 72,
            "uncommon": 54,
            "rare": 48,
            "super_rare": 18,
            "legendary": 12,
            "enchanted": 12,
        },
    },
    2: {
        "name": "Rise of the Floodborn",
        "rarity_card_counts": {
            "common": 72,
            "uncommon": 54,
            "rare": 48,
            "super_rare": 18,
            "legendary": 12,
            "enchanted": 12,
        },
    },
    3: {
        "name": "Into the Inklands",
        "rarity_card_counts": {
            "common": 72,
            "uncommon": 54,
            "rare": 48,
            "super_rare": 18,
            "legendary": 12,
            "enchanted": 18,
        },
    },
    4: {
        "name": "Ursula's Return",
        "rarity_card_counts": {
            "common": 72,
            "uncommon": 54,
            "rare": 48,
            "super_rare": 18,
            "legendary": 12,
            "enchanted": 18,
        },
    },
    5: {
        "name": "Shimmering Skies",
        "rarity_card_counts": {
            "common": 72,
            "uncommon": 54,
            "rare": 48,
            "super_rare": 18,
            "legendary": 12,
            "enchanted": 18,
        },
    },
    6: {
        "name": "Azurite Sea",
        "rarity_card_counts": {
            "common": 72,
            "uncommon": 54,
            "rare": 48,
            "super_rare": 18,
            "legendary": 12,
            "enchanted": 18,
        },
    },
    7: {
        "name": "Archazia's Island",
        "rarity_card_counts": {
            "common": 72,
            "uncommon": 54,
            "rare": 48,
            "super_rare": 18,
            "legendary": 12,
            "enchanted": 18,
        },
    },
    8: {
        "name": "Reign of Jafar",
        "rarity_card_counts": {
            "common": 72,
            "uncommon": 54,
            "rare": 48,
            "super_rare": 18,
            "legendary": 12,
            "enchanted": 18,
        },
    },
    9: {
        "name": "Fabled",
        "rarity_card_counts": {
            "common": 72,
            "uncommon": 54,
            "rare": 48,
            "super_rare": 18,
            "legendary": 12,
            "epic": 18,
            "enchanted": 18,
            "iconic": 2,
        },
    },
    10: {
        "name": "Whispers in the Well",
        "rarity_card_counts": {
            "common": 72,
            "uncommon": 54,
            "rare": 48,
            "super_rare": 18,
            "legendary": 12,
            "epic": 18,
            "enchanted": 18,
            "iconic": 2,
        },
    },
}


# ================================================================
#  Foil slot: seeded vs non-seeded
# ================================================================

def seeded_foil_probs(rarity_counts: Dict[str, int]) -> Dict[str, float]:
    """Seeded foil probabilities per pack (any card of that rarity)."""
    seeded: Dict[str, float] = {}

    if "enchanted" in rarity_counts and rarity_counts["enchanted"] > 0:
        seeded["enchanted"] = ENCHANTED_PER_PACK_ANY

    if "iconic" in rarity_counts and rarity_counts["iconic"] > 0:
        seeded["iconic"] = ICONIC_PER_PACK_ANY

    if "epic" in rarity_counts and rarity_counts["epic"] > 0:
        seeded["epic"] = EPIC_PER_PACK_ANY

    return seeded


def non_seeded_foil_distribution(
    rarity_counts: Dict[str, int],
) -> Tuple[float, Dict[str, float]]:
    """
    Compute:
      - non_seeded_foil_prob: foil slot probability that is *not* Epic/Enchanted/Iconic.
      - foil_per_rarity: contribution per rarity among the non-seeded pool,
        assuming uniform-by-card.
    """
    seeded = seeded_foil_probs(rarity_counts)
    p_seeded_total = sum(seeded.values())
    non_seeded_foil_prob = max(0.0, 1.0 - p_seeded_total)

    seeded_rarities = set(seeded.keys())
    non_seeded_total_cards = sum(
        count for r, count in rarity_counts.items() if r not in seeded_rarities
    )

    foil_per_rarity: Dict[str, float] = {r: 0.0 for r in rarity_counts.keys()}

    if non_seeded_total_cards > 0 and non_seeded_foil_prob > 0.0:
        for r, count in rarity_counts.items():
            if r in seeded_rarities:
                continue
            foil_per_rarity[r] = non_seeded_foil_prob * (count / non_seeded_total_cards)

    return non_seeded_foil_prob, foil_per_rarity


# ================================================================
#  Building SetConfig (expected_per_pack)
# ================================================================

def make_set_config(set_number: int) -> SetConfig:
    """
    Build a SetConfig for a given set number.
    """
    meta = SET_RARITY_COUNTS[set_number]
    name = meta["name"]
    counts = meta["rarity_card_counts"]

    expected_per_pack: Dict[str, float] = {r: 0.0 for r in counts.keys()}

    if "common" in counts:
        expected_per_pack["common"] += 6.0
    if "uncommon" in counts:
        expected_per_pack["uncommon"] += 3.0

    if "rare" in counts:
        expected_per_pack["rare"] += E_RARE_FROM_SLOTS
    if "super_rare" in counts:
        expected_per_pack["super_rare"] += E_SUPER_RARE_FROM_SLOTS
    if "legendary" in counts:
        expected_per_pack["legendary"] += E_LEGENDARY_FROM_SLOTS

    seeded = seeded_foil_probs(counts)
    for r, p_any in seeded.items():
        expected_per_pack[r] += p_any

    _, foil_per_rarity = non_seeded_foil_distribution(counts)
    for r, p_any in foil_per_rarity.items():
        expected_per_pack[r] += p_any

    return SetConfig(
        set_number=set_number,
        name=name,
        rarity_card_counts=counts,
        expected_per_pack=expected_per_pack,
    )


SET_CONFIGS: Dict[int, SetConfig] = {
    n: make_set_config(n) for n in range(1, 11)
}


# ================================================================
#  Per-card lambda & Poisson distributions
# ================================================================

def lambda_for_card(cfg: SetConfig, rarity: str, num_packs: int) -> float:
    if rarity not in cfg.rarity_card_counts:
        raise ValueError(f"Rarity {rarity!r} not in set config.")
    if rarity not in cfg.expected_per_pack:
        raise ValueError(f"No expected_per_pack entry for rarity {rarity!r}.")
    N_R = cfg.rarity_card_counts[rarity]
    E_R = cfg.expected_per_pack[rarity]
    return num_packs * (E_R / N_R)


def lambda_for_rarity_total(cfg: SetConfig, rarity: str, num_packs: int) -> float:
    if rarity not in cfg.expected_per_pack:
        raise ValueError(f"No expected_per_pack entry for rarity {rarity!r}.")
    E_R = cfg.expected_per_pack[rarity]
    return num_packs * E_R


def card_distribution_poisson(
    cfg: SetConfig,
    rarity: str,
    num_packs: int,
    max_k: int | None = None,
    cutoff: float = 0.999,
) -> Tuple[float, List[Tuple[int, float]]]:
    lam = lambda_for_card(cfg, rarity, num_packs)
    return lam, poisson_distribution(lam, max_k=max_k, cutoff=cutoff)


def prob_at_least_k_copies(
    cfg: SetConfig,
    rarity: str,
    num_packs: int,
    k_min: int,
) -> float:
    lam = lambda_for_card(cfg, rarity, num_packs)
    return prob_at_least(k_min, lam)


# ================================================================
#  Monte Carlo simulations
# ================================================================


def monte_carlo_histogram(
    lam: float,
    num_trials: int,
    max_k: int = 10,
) -> dict:
    samples = np.random.poisson(lam, size=num_trials)
    counts = [int((samples == k).sum()) for k in range(max_k + 1)]
    overflow = int((samples > max_k).sum())
    return {
        "lambda": float(lam),
        "num_trials": int(num_trials),
        "max_k": int(max_k),
        "counts": counts,
        "overflow": overflow,
    }


def monte_carlo_histogram_for_card(
    cfg: SetConfig,
    rarity: str,
    num_packs: int,
    num_trials: int,
    max_k: int = 10,
) -> dict:
    lam = lambda_for_card(cfg, rarity, num_packs)
    return monte_carlo_histogram(lam, num_trials, max_k=max_k)


def inspect_rarity(set_number: int, rarity: str, num_packs: int, k_values=(1, 4)) -> dict:
    cfg = SET_CONFIGS[set_number]
    lam_card = lambda_for_card(cfg, rarity, num_packs)
    lam_any = lambda_for_rarity_total(cfg, rarity, num_packs)
    p0_card = math.exp(-lam_card)
    p0_any = math.exp(-lam_any)
    pk_card = {k: prob_at_least(k, lam_card) for k in k_values}
    pk_any = {k: prob_at_least(k, lam_any) for k in k_values}
    return {
        "set_number": set_number,
        "set_name": cfg.name,
        "rarity": rarity,
        "packs": num_packs,
        "lambda_card": lam_card,
        "lambda_any": lam_any,
        "p0_card": p0_card,
        "p0_any": p0_any,
        "pk_card": pk_card,
        "pk_any": pk_any,
    }
