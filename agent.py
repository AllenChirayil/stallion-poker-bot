from __future__ import annotations

"""
agent.py - YOUR poker bot. This one file is a complete web server.

Run it:
    python agent.py                  # listens on port 5008
    PORT=5010 python agent.py        # or pick a port

Play a match against a baseline (three windows):
    python agent.py                                 # your bot on port 5008
    python baseline-bots/baseline_random.py         # opponent on port 5009
    python judge_engine.py                          # judge: both halves (uses those two ports)

Check that the tournament can run it:
    python -m pytest -q              # tests in test_agent.py
    ./tools/check_build.sh           # builds and smoke-tests it like the tournament

TODO:
    1. Assign your agent a unique PARTICIPANT and AGENT_NAME below.
    2. Implement the three functions between "YOUR CODE HERE" and "END OF YOUR CODE".

Everything under "SERVER PLUMBING" talks to the judge. You normally never
need to change it, and if you break it the judge can't talk to your bot.
"""

import os
import json
import traceback

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from evaluator import (
    card_rank, card_suit, cards_to_str, omaha_score, warmup,
    NUM_CARDS, ACE, FLUSH, STRAIGHT, THREE_OF_A_KIND,
    FULL_HOUSE, FOUR_OF_A_KIND, STRAIGHT_FLUSH, TWO_PAIR, ONE_PAIR,
    HIGH_CARD, category_of,
)
from game import Action, AgentConfig, HandResult, Observation, Street

# Build omaha_score's lookup table now, while the server starts, instead of
# during your first decision (where it would eat into your time bank).
# Do any other slow setup (loading a model, ...) here too: the server must
# answer within 15 seconds of starting.
warmup()

# rename as your team/bot's name
PARTICIPANT = "Team Mustangs"
AGENT_NAME = "Stallion"


# ===== YOUR CODE HERE =====

import random
from itertools import combinations

# --------------------------------------------------------------------------
# Pre-computed constants
# --------------------------------------------------------------------------

# The 10 ways to pick 2 of 5 hole cards
_HOLE_PAIRS = list(combinations(range(5), 2))
# The 10 ways to pick 3 of 5 board cards
_BOARD_TRIPLES = list(combinations(range(5), 3))

# Card encoding helpers
def _rank(c: int) -> int:
    return c >> 2

def _suit(c: int) -> int:
    return c & 3


# --------------------------------------------------------------------------
# Preflop Neural Model Loader & Predictor
# --------------------------------------------------------------------------

def _extract_preflop_features(hole: tuple[int, ...]) -> list[float]:
    """Extract 22 numerical features for a 5-card preflop hand."""
    from collections import Counter
    ranks = sorted([c >> 2 for c in hole], reverse=True)
    suits = [c & 3 for c in hole]
    
    rank_counts = Counter(ranks)
    pairs = sum(1 for c in rank_counts.values() if c == 2)
    trips = sum(1 for c in rank_counts.values() if c == 3)
    quads = sum(1 for c in rank_counts.values() if c >= 4)
    
    suit_counts = sorted(Counter(suits).values(), reverse=True)
    while len(suit_counts) < 4:
        suit_counts.append(0)
    
    suits_with_ge_2 = sum(1 for c in suit_counts if c >= 2)
    is_double_suited = 1.0 if suits_with_ge_2 >= 2 else 0.0
    
    suit_to_cards = {}
    for c in hole:
        s = c & 3
        suit_to_cards.setdefault(s, []).append(c >> 2)
    
    has_nut_flush_draw = 0.0
    has_second_nut_flush_draw = 0.0
    for s, s_ranks in suit_to_cards.items():
        if len(s_ranks) >= 2:
            if 8 in s_ranks:
                has_nut_flush_draw = 1.0
            elif 7 in s_ranks:
                has_second_nut_flush_draw = 1.0
                
    distinct_ranks = sorted(set(ranks))
    conn_adjacent = 0
    conn_onegap = 0
    for i in range(len(distinct_ranks) - 1):
        gap = distinct_ranks[i + 1] - distinct_ranks[i]
        if gap == 1:
            conn_adjacent += 1
        elif gap == 2:
            conn_onegap += 1
    wheel_wrap = 1.0 if (8 in distinct_ranks and 0 in distinct_ranks) else 0.0
    
    aces = ranks.count(8)
    kings = ranks.count(7)
    queens = ranks.count(6)
    jacks = ranks.count(5)
    
    avg_rank = sum(ranks) / 5.0
    
    return [
        avg_rank / 8.0,
        aces / 4.0, kings / 4.0, queens / 4.0, jacks / 4.0,
        pairs / 2.0, float(trips), float(quads),
        suit_counts[0] / 5.0, suit_counts[1] / 5.0,
        is_double_suited,
        has_nut_flush_draw,
        has_second_nut_flush_draw,
        conn_adjacent / 4.0,
        conn_onegap / 4.0,
        wheel_wrap,
        len(distinct_ranks) / 5.0,
        ranks[0] / 8.0, ranks[1] / 8.0, ranks[2] / 8.0, ranks[3] / 8.0, ranks[4] / 8.0
    ]

_PREFLOP_MODEL = None

def _load_preflop_model():
    global _PREFLOP_MODEL
    try:
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "preflop_model.json")
        if os.path.exists(model_path):
            with open(model_path, "r") as f:
                d = json.load(f)
            try:
                import numpy as np
                _PREFLOP_MODEL = {
                    "W1": np.array(d["W1"], dtype=np.float32),
                    "b1": np.array(d["b1"], dtype=np.float32),
                    "W2": np.array(d["W2"], dtype=np.float32).reshape(-1, 1),
                    "b2": np.float32(d["b2"]),
                    "use_numpy": True,
                }
            except Exception:
                _PREFLOP_MODEL = {
                    "W1": d["W1"],
                    "b1": d["b1"],
                    "W2": d["W2"],
                    "b2": d["b2"],
                    "use_numpy": False,
                }
    except Exception:
        _PREFLOP_MODEL = None

_load_preflop_model()

def _predict_preflop_equity(hole_cards: tuple) -> float:
    """Predict preflop equity using the trained 2-layer neural network."""
    if _PREFLOP_MODEL is None:
        return _preflop_strength(hole_cards)
    try:
        feat = _extract_preflop_features(hole_cards)
        if _PREFLOP_MODEL["use_numpy"]:
            import numpy as np
            x = np.array(feat, dtype=np.float32)
            h1 = np.maximum(0.0, x.dot(_PREFLOP_MODEL["W1"]) + _PREFLOP_MODEL["b1"])
            out = float((h1.dot(_PREFLOP_MODEL["W2"]) + _PREFLOP_MODEL["b2"])[0])
        else:
            W1 = _PREFLOP_MODEL["W1"]
            b1 = _PREFLOP_MODEL["b1"]
            W2 = _PREFLOP_MODEL["W2"]
            b2 = _PREFLOP_MODEL["b2"]
            hidden = [0.0] * len(b1)
            for j in range(len(b1)):
                val = b1[j]
                for i in range(len(feat)):
                    val += feat[i] * W1[i][j]
                hidden[j] = val if val > 0.0 else 0.0
            out = b2
            for j in range(len(hidden)):
                out += hidden[j] * W2[j]
        return max(0.15, min(0.85, out))
    except Exception:
        return _preflop_strength(hole_cards)


# --------------------------------------------------------------------------
# Preflop hand strength evaluation (fast heuristic fallback)
# --------------------------------------------------------------------------

def _preflop_strength(hole_cards: tuple) -> float:
    """Evaluate preflop hand strength on a 0-1 scale based on hand features.

    Features considered:
    - High cards (aces, kings, queens)
    - Pairs / trips in hole cards
    - Suited cards (flush potential)
    - Connected ranks (straight potential)
    - Double-suited hands

    Returns a score from 0.0 (trash) to 1.0 (premium).
    """
    ranks = [_rank(c) for c in hole_cards]
    suits = [_suit(c) for c in hole_cards]

    score = 0.0

    # --- High card value ---
    # Ace=8, King=7, Queen=6, Jack=5 in this encoding
    high_card_values = {8: 0.12, 7: 0.08, 6: 0.05, 5: 0.03}
    for r in ranks:
        score += high_card_values.get(r, 0.0)

    # --- Pairs and trips ---
    from collections import Counter
    rank_counts = Counter(ranks)
    for r, cnt in rank_counts.items():
        if cnt == 2:
            # Pair bonus, scaled by rank
            score += 0.08 + 0.02 * (r / 8.0)
        elif cnt == 3:
            # Trips in hand (strong)
            score += 0.15 + 0.03 * (r / 8.0)
        elif cnt == 4:
            # Quads in hand is actually bad in Omaha (can only use 2)
            score -= 0.05

    # --- Suit groups (flush potential) ---
    suit_counts = Counter(suits)
    num_suited_groups = 0
    for s, cnt in suit_counts.items():
        if cnt >= 2:
            num_suited_groups += 1
            if cnt == 2:
                score += 0.06
            elif cnt == 3:
                score += 0.08  # 3 suited is decent
            elif cnt >= 4:
                score += 0.04  # 4+ suited is bad in Omaha (uses exactly 2)

    # Double-suited bonus (two different suits each with 2+ cards)
    if num_suited_groups >= 2:
        score += 0.05

    # --- Connectivity (straight potential) ---
    unique_ranks = sorted(set(ranks))
    # Count how many connected pairs exist
    connected = 0
    for i in range(len(unique_ranks) - 1):
        gap = unique_ranks[i + 1] - unique_ranks[i]
        if gap == 1:
            connected += 1
        elif gap == 2:
            connected += 0.5  # one-gapper

    # Ace-low connectivity (A-6-7-8-9 wrap)
    if 8 in unique_ranks and 0 in unique_ranks:  # Ace and 6
        connected += 0.5
    if 8 in unique_ranks and 1 in unique_ranks:  # Ace and 7
        connected += 0.3

    score += connected * 0.04

    # --- Rundown bonus (4+ cards in a run, great for straights) ---
    if len(unique_ranks) >= 4:
        # Check for 4-card runs
        for i in range(len(unique_ranks) - 3):
            if unique_ranks[i + 3] - unique_ranks[i] <= 4:
                score += 0.06
                break

    # Clamp to [0, 1]
    return max(0.0, min(1.0, score))


# --------------------------------------------------------------------------
# Blocker and Nut Potential Awareness
# --------------------------------------------------------------------------

def _has_nut_blocker(hole_cards: tuple, board: tuple, category: int) -> bool:
    """Check if we hold key nut blockers (e.g. Ace of flush suit or top straight cards)."""
    if category == FLUSH:
        from collections import Counter
        board_suits = [c & 3 for c in board]
        suit_counts = Counter(board_suits)
        if not suit_counts:
            return True
        flush_suit = max(suit_counts, key=suit_counts.get)
        ace_flush_card = (ACE << 2) | flush_suit
        return ace_flush_card in hole_cards
    elif category == STRAIGHT:
        ranks = [c >> 2 for c in hole_cards]
        return (ACE in ranks) or ((ACE - 1) in ranks)
    return True


# --------------------------------------------------------------------------
# Monte Carlo equity estimation (post-flop)
# --------------------------------------------------------------------------

def _monte_carlo_equity(hole_cards: tuple, board: tuple, num_sims: int,
                        rng: random.Random, filter_opp_range: bool = False) -> float:
    """Estimate win probability by dealing random opponent hands and
    completing the board, then comparing omaha_score.

    If filter_opp_range is True (e.g. facing bets), filters out pure uncoordinated
    trash without high cards or pairs so we don't over-estimate our equity.
    """
    used = set(hole_cards) | set(board)
    remaining = [c for c in range(NUM_CARDS) if c not in used]
    cards_to_deal_board = 5 - len(board)
    board_list = list(board)

    wins = 0
    ties = 0
    total = 0

    for _ in range(num_sims):
        needed = cards_to_deal_board + 5
        if needed > len(remaining):
            break
        sampled = rng.sample(remaining, needed)
        sim_board = board_list + sampled[:cards_to_deal_board]
        opp_hole = sampled[cards_to_deal_board:]

        # Range filter: if opponent is betting, eliminate uncoordinated rags
        if filter_opp_range:
            opp_ranks = [c >> 2 for c in opp_hole]
            if max(opp_ranks) < 6 and len(set(opp_ranks)) == 5:
                # Resample once to draw from a stronger distribution
                sampled = rng.sample(remaining, needed)
                sim_board = board_list + sampled[:cards_to_deal_board]
                opp_hole = sampled[cards_to_deal_board:]

        my_score = omaha_score(hole_cards, tuple(sim_board))
        opp_score = omaha_score(tuple(opp_hole), tuple(sim_board))

        if my_score > opp_score:
            wins += 1
        elif my_score == opp_score:
            ties += 1
        total += 1

    if total == 0:
        return 0.5
    return (wins + ties * 0.5) / total


# --------------------------------------------------------------------------
# Pot odds calculation
# --------------------------------------------------------------------------

def _pot_odds(to_call: int, pot: int) -> float:
    """Fraction of the time we need to win to justify calling.
    pot_odds = to_call / (pot + to_call)
    """
    if to_call <= 0:
        return 0.0
    return to_call / (pot + to_call)


# --------------------------------------------------------------------------
# Opponent modeling with Bayesian smoothing
# --------------------------------------------------------------------------

def _get_opp_stats(memory: dict) -> dict:
    """Return aggregated opponent statistics with Laplace Bayesian smoothing."""
    hands = memory.get("opp_hands_seen", 0)

    # Informative Bayesian priors (equivalent to observing ~6 standard hands)
    prior_vpip = 0.60 * 6
    prior_pfr = 0.30 * 6
    prior_raises = 4.0
    prior_calls = 3.0
    prior_fold_to_raise = 1.8
    prior_faced_raise = 4.5

    obs_vpip = memory.get("opp_vpip_count", 0)
    obs_pfr = memory.get("opp_pfr_count", 0)
    obs_raises = memory.get("opp_raise_count", 0)
    obs_calls = memory.get("opp_call_count", 0)
    obs_folds = memory.get("opp_fold_to_raise", 0)
    obs_faced = memory.get("opp_faced_raise", 0)

    vpip = (obs_vpip + prior_vpip) / (hands + 6)
    pfr = (obs_pfr + prior_pfr) / (hands + 6)
    agg_factor = (obs_raises + prior_raises) / max(obs_calls + prior_calls, 1.0)
    fold_to_raise = (obs_folds + prior_fold_to_raise) / max(obs_faced + prior_faced_raise, 1.0)

    return {
        "vpip": vpip,
        "pfr": pfr,
        "agg_factor": agg_factor,
        "fold_to_raise": fold_to_raise,
        "showdown_rate": memory.get("opp_showdown_count", 0) / max(hands, 1),
        "hands_seen": hands,
    }


def _update_opponent_history(obs: Observation, memory: dict) -> None:
    """Scan action history to record opponent betting patterns in real time."""
    current_hand = obs.hand_index
    last_processed = memory.get("last_processed_action_idx", -1)
    last_hand = memory.get("last_processed_hand", -1)

    if current_hand != last_hand:
        last_processed = -1
        memory["last_processed_hand"] = current_hand

    for i in range(last_processed + 1, len(obs.action_history)):
        act = obs.action_history[i]
        if not act.is_me:
            if act.action in (Action.RAISE_HALF, Action.RAISE_POT):
                memory["opp_raise_count"] = memory.get("opp_raise_count", 0) + 1
                if act.street == Street.PREFLOP:
                    memory["opp_pfr_count"] = memory.get("opp_pfr_count", 0) + 1
                    memory["opp_vpip_count"] = memory.get("opp_vpip_count", 0) + 1
            elif act.action == Action.CHECK_CALL:
                memory["opp_call_count"] = memory.get("opp_call_count", 0) + 1
                if act.street == Street.PREFLOP and act.amount > 0:
                    memory["opp_vpip_count"] = memory.get("opp_vpip_count", 0) + 1
        else:
            if act.action in (Action.RAISE_HALF, Action.RAISE_POT):
                memory["opp_faced_raise"] = memory.get("opp_faced_raise", 0) + 1

    memory["last_processed_action_idx"] = len(obs.action_history) - 1


# --------------------------------------------------------------------------
# Main strategy functions
# --------------------------------------------------------------------------

def new_half(config: AgentConfig, memory: dict) -> None:
    """Called at the start of each half-match (1000 hands).

    `memory` is a fresh, empty dict every half. Store anything you want to
    remember between hands in it, e.g. memory["opp_raises"] = 0.
    `config` holds the match settings (starting_stack, big_blind, ...).

    In the tournament, each half runs in a brand-new copy of your server,
    so nothing you learn in one half carries over to the other half.
    """
    memory["hands_played"] = 0
    memory["rng"] = random.Random(42)

    # Opponent tracking stats
    memory["opp_hands_seen"] = 0
    memory["opp_vpip_count"] = 0       # Voluntarily put in pot (not just posting blind)
    memory["opp_pfr_count"] = 0        # Preflop raise
    memory["opp_raise_count"] = 0      # Total raises
    memory["opp_call_count"] = 0       # Total calls
    memory["opp_fold_count"] = 0       # Total folds
    memory["opp_fold_to_raise"] = 0    # Folded when facing a raise
    memory["opp_faced_raise"] = 0      # Times opponent faced a raise
    memory["opp_showdown_count"] = 0   # Hands that went to showdown
    memory["opp_bluff_count"] = 0      # Raised with losing hand at showdown
    memory["opp_showdown_raise_count"] = 0  # Raised and went to showdown

    # Track our bluffing frequency to stay balanced
    memory["our_bluff_count"] = 0
    memory["our_value_bet_count"] = 0


def choose_action(obs: Observation, memory: dict) -> Action:
    """Called every time it is your turn. Return one of obs.legal_actions.

    Multi-layered strategy:
    1. Preflop: Use hand feature evaluation
    2. Post-flop: Monte Carlo equity + pot odds
    3. Opponent modeling: Adjust based on opponent tendencies
    4. Position awareness: Leverage button advantage
    """
    try:
        return _choose_action_impl(obs, memory)
    except Exception:
        # Safety net: never crash, always return a legal action
        if Action.CHECK_CALL in obs.legal_actions:
            return Action.CHECK_CALL
        return obs.legal_actions[0]


def _choose_action_impl(obs: Observation, memory: dict) -> Action:
    """Internal implementation of choose_action with full strategy."""
    _update_opponent_history(obs, memory)
    rng = memory["rng"]
    opp_stats = _get_opp_stats(memory)
    pot_odds = _pot_odds(obs.to_call, obs.pot)

    # ====== PREFLOP ======
    if obs.street == Street.PREFLOP:
        return _preflop_action(obs, memory, rng, opp_stats, pot_odds)

    # ====== POST-FLOP (Flop, Turn, River) ======
    return _postflop_action(obs, memory, rng, opp_stats, pot_odds)


def _preflop_action(obs: Observation, memory: dict, rng: random.Random,
                    opp_stats: dict, pot_odds: float) -> Action:
    """Preflop strategy using neural network predicted showdown equity."""
    equity = _predict_preflop_equity(obs.hole_cards)

    # Position bonus: button acts last post-flop
    if obs.is_button:
        equity += 0.04

    fold_to_raise = opp_stats["fold_to_raise"]

    # --- Premium hands (top ~15%, equity >= 0.58): raise pot ---
    if equity >= 0.58:
        if Action.RAISE_POT in obs.legal_actions:
            return Action.RAISE_POT
        if Action.RAISE_HALF in obs.legal_actions:
            return Action.RAISE_HALF
        return Action.CHECK_CALL

    # --- Strong hands (~15-35%, equity >= 0.52): raise half or call ---
    if equity >= 0.52:
        if fold_to_raise > 0.40 and Action.RAISE_HALF in obs.legal_actions:
            return Action.RAISE_HALF
        if Action.RAISE_HALF in obs.legal_actions and rng.random() < 0.4:
            return Action.RAISE_HALF
        return Action.CHECK_CALL

    # --- Playable hands (~35-60%, equity >= 0.45): call cheap bets ---
    if equity >= 0.45:
        if obs.to_call <= obs.big_blind * 2:
            return Action.CHECK_CALL
        # Facing larger raises: call if we have odds and decent equity
        if obs.to_call <= obs.big_blind * 4 and equity >= 0.48:
            return Action.CHECK_CALL
        if Action.FOLD in obs.legal_actions:
            return Action.FOLD
        return Action.CHECK_CALL

    # --- Weak hands (equity < 0.45): fold if bet into, check if free ---
    if obs.to_call == 0:
        return Action.CHECK_CALL
    # Button discount: small blind only needs 1 chip to complete
    if obs.is_button and obs.to_call <= (obs.big_blind // 2) and equity >= 0.38:
        return Action.CHECK_CALL
    if Action.FOLD in obs.legal_actions:
        return Action.FOLD
    return Action.CHECK_CALL


def _postflop_action(obs: Observation, memory: dict, rng: random.Random,
                     opp_stats: dict, pot_odds: float) -> Action:
    """Post-flop strategy using equity estimation and opponent modeling."""
    facing_bet = obs.to_call > 0

    if obs.street == Street.RIVER:
        my_score = omaha_score(obs.hole_cards, obs.board)
        category = category_of(my_score)
        return _river_action(obs, memory, rng, opp_stats, pot_odds,
                             my_score, category)

    if obs.street == Street.FLOP:
        num_sims = 60
    else:  # TURN
        num_sims = 80

    # If facing an aggressive bet, filter opponent range to realistic cards
    equity = _monte_carlo_equity(obs.hole_cards, obs.board, num_sims, rng,
                                 filter_opp_range=facing_bet)

    return _equity_based_action(obs, memory, rng, opp_stats, pot_odds, equity)


def _river_action(obs: Observation, memory: dict, rng: random.Random,
                  opp_stats: dict, pot_odds: float,
                  my_score: int, category: int) -> Action:
    """River strategy: complete board with blocker awareness and polarized bluffs."""
    facing_bet = obs.to_call > 0
    equity = _monte_carlo_equity(obs.hole_cards, obs.board, 120, rng,
                                 filter_opp_range=facing_bet)

    # Cooler / reverse-implied-odds protection:
    # If holding non-nut flush or non-nut straight against a large bet, discount equity
    if category in (FLUSH, STRAIGHT) and not _has_nut_blocker(obs.hole_cards, obs.board, category):
        if obs.to_call > obs.big_blind * 3:
            equity -= 0.08

    # === Monster hands (equity >= 0.78) ===
    if equity >= 0.78:
        memory["our_value_bet_count"] = memory.get("our_value_bet_count", 0) + 1
        if not facing_bet:
            # Trap play: 10% slow-play check to protect checking range
            if rng.random() < 0.10:
                return Action.CHECK_CALL
            # Mixed sizing: 75% pot, 25% half-pot (unpredictable)
            if rng.random() < 0.75 and Action.RAISE_POT in obs.legal_actions:
                return Action.RAISE_POT
            if Action.RAISE_HALF in obs.legal_actions:
                return Action.RAISE_HALF
            return Action.CHECK_CALL
        else:
            if Action.RAISE_POT in obs.legal_actions and equity >= 0.85:
                return Action.RAISE_POT
            if Action.RAISE_HALF in obs.legal_actions:
                return Action.RAISE_HALF
            return Action.CHECK_CALL

    # === Good hands (equity 0.55 - 0.78) ===
    if equity >= 0.55:
        if not facing_bet:
            if Action.RAISE_HALF in obs.legal_actions and rng.random() < 0.65:
                return Action.RAISE_HALF
            return Action.CHECK_CALL
        else:
            if equity >= pot_odds + 0.04:
                return Action.CHECK_CALL
            if Action.FOLD in obs.legal_actions:
                return Action.FOLD
            return Action.CHECK_CALL

    # === Marginal hands (equity 0.35 - 0.55) ===
    if equity >= 0.35:
        if not facing_bet:
            return Action.CHECK_CALL
        if pot_odds < 0.30 and equity >= pot_odds:
            return Action.CHECK_CALL
        if Action.FOLD in obs.legal_actions:
            return Action.FOLD
        return Action.CHECK_CALL

    # === Weak hands (equity < 0.35): Polarized bluffs ===
    if not facing_bet:
        if opp_stats["fold_to_raise"] > 0.35:
            bluffs = memory.get("our_bluff_count", 0)
            value_bets = memory.get("our_value_bet_count", 0)
            if value_bets > 0 and (bluffs / (bluffs + value_bets + 1)) < 0.30:
                if rng.random() < 0.22:
                    memory["our_bluff_count"] = bluffs + 1
                    # 40% pot bluff, 60% half-pot bluff
                    if rng.random() < 0.40 and Action.RAISE_POT in obs.legal_actions:
                        return Action.RAISE_POT
                    if Action.RAISE_HALF in obs.legal_actions:
                        return Action.RAISE_HALF
        return Action.CHECK_CALL

    # Facing bet with weak hand: occasional check-raise bluff on cheap bets against chronic folders
    if (obs.to_call <= obs.big_blind * 2 and
            opp_stats["fold_to_raise"] > 0.45 and
            Action.RAISE_HALF in obs.legal_actions and
            rng.random() < 0.12):
        memory["our_bluff_count"] = memory.get("our_bluff_count", 0) + 1
        return Action.RAISE_HALF

    if Action.FOLD in obs.legal_actions:
        return Action.FOLD
    return Action.CHECK_CALL


def _equity_based_action(obs: Observation, memory: dict, rng: random.Random,
                         opp_stats: dict, pot_odds: float,
                         equity: float) -> Action:
    """Flop/Turn strategy with mixed bet sizing and deceptive checking."""
    facing_bet = obs.to_call > 0

    # === Monster hands (equity >= 0.78) ===
    if equity >= 0.78:
        memory["our_value_bet_count"] = memory.get("our_value_bet_count", 0) + 1
        if not facing_bet:
            # Trap play: 15% slow-play check to protect checking range and trap aggressive bots
            if rng.random() < 0.15:
                return Action.CHECK_CALL
            # Mixed sizing: 70% pot, 30% half-pot
            if rng.random() < 0.70 and Action.RAISE_POT in obs.legal_actions:
                return Action.RAISE_POT
            if Action.RAISE_HALF in obs.legal_actions:
                return Action.RAISE_HALF
            return Action.CHECK_CALL
        else:
            if rng.random() < 0.75 and Action.RAISE_POT in obs.legal_actions:
                return Action.RAISE_POT
            if Action.RAISE_HALF in obs.legal_actions:
                return Action.RAISE_HALF
            return Action.CHECK_CALL

    # === Strong hands (equity 0.58 - 0.78) ===
    if equity >= 0.58:
        if not facing_bet:
            # Value bet: mostly half-pot, 20% pot
            if rng.random() < 0.20 and Action.RAISE_POT in obs.legal_actions:
                return Action.RAISE_POT
            if Action.RAISE_HALF in obs.legal_actions and rng.random() < 0.75:
                return Action.RAISE_HALF
            return Action.CHECK_CALL
        else:
            if equity >= 0.68:
                if Action.RAISE_POT in obs.legal_actions and rng.random() < 0.50:
                    return Action.RAISE_POT
                if Action.RAISE_HALF in obs.legal_actions:
                    return Action.RAISE_HALF
            elif Action.RAISE_HALF in obs.legal_actions and rng.random() < 0.30:
                return Action.RAISE_HALF
            return Action.CHECK_CALL

    # === Drawing / Medium hands (equity 0.40 - 0.58) ===
    if equity >= 0.40:
        if not facing_bet:
            if obs.is_button and Action.RAISE_HALF in obs.legal_actions and rng.random() < 0.30:
                return Action.RAISE_HALF
            return Action.CHECK_CALL
        else:
            implied_needed = pot_odds * 0.90
            if equity >= implied_needed and obs.to_call <= obs.pot * 0.6:
                return Action.CHECK_CALL
            if Action.FOLD in obs.legal_actions:
                return Action.FOLD
            return Action.CHECK_CALL

    # === Weak hands (equity < 0.40) ===
    if not facing_bet:
        if opp_stats["fold_to_raise"] > 0.40 and Action.RAISE_HALF in obs.legal_actions and rng.random() < 0.15:
            memory["our_bluff_count"] = memory.get("our_bluff_count", 0) + 1
            return Action.RAISE_HALF
        return Action.CHECK_CALL

    if equity >= 0.25 and pot_odds < 0.20 and obs.to_call <= obs.big_blind * 2:
        return Action.CHECK_CALL

    if Action.FOLD in obs.legal_actions:
        return Action.FOLD
    return Action.CHECK_CALL


def on_hand_end(result: HandResult, memory: dict) -> None:
    """Called once for every finished hand. Track opponent tendencies."""
    try:
        memory["hands_played"] = memory.get("hands_played", 0) + 1
        memory["opp_hands_seen"] = memory.get("opp_hands_seen", 0) + 1

        if result.went_to_showdown:
            memory["opp_showdown_count"] = memory.get("opp_showdown_count", 0) + 1
        elif result.winner == "me":
            # Opponent folded
            memory["opp_fold_count"] = memory.get("opp_fold_count", 0) + 1
            memory["opp_fold_to_raise"] = memory.get("opp_fold_to_raise", 0) + 1

    except Exception:
        pass  # Never crash in on_hand_end


# ===== END OF YOUR CODE =====


# ===== SERVER PLUMBING (you normally don't need to edit below) =====
# The judge calls these routes:
#   GET  /           -> who you are (also used as the health check)
#   POST /new-half   -> start a fresh half: memory is wiped
#   POST /act        -> the judge sends any finished hands plus an Observation;
#                       on_hand_end() runs for each hand, then you answer an action
#   POST /hand-end   -> the judge sends the last finished hands of a half
#
# Why "async def"? The routes run on one thread and call your functions
# directly, so your functions never run twice at once. Your own functions
# stay normal "def" (don't make them async).
#
# Keep your per-half state in `memory`. In the tournament every half runs in
# a fresh server process, so everything starts over. When you test locally,
# though, one process can play both halves and only `memory` is wiped, NOT
# other global variables. Using `memory` makes your bot behave the same way
# in both places.

app = FastAPI()
MEMORY: dict = {}                                   # your memory for this half
STATUS = {"started": False}                         # has /new-half been called?
TOKEN = os.environ.get("AGENT_TOKEN") or None       # set by the tournament


async def _read_request(request: Request, path: str) -> tuple[dict, JSONResponse | None]:
    """Check the token (and that the half has started) BEFORE reading the body.

    Checking first means a request without the right token costs your bot
    almost nothing, so nobody else can slow it down by sending junk.
    Returns (body, None) when the request is fine, or ({}, error response).
    """
    if TOKEN and request.headers.get("x-agent-token") != TOKEN:
        return {}, JSONResponse({"error": "bad or missing X-Agent-Token"}, status_code=401)
    if not STATUS["started"] and path != "/new-half":
        return {}, JSONResponse({"error": "call /new-half first"}, status_code=409)
    try:
        body = await request.json()
    except ValueError:
        body = None
    if not isinstance(body, dict):
        return {}, JSONResponse({"error": "the request body must be a JSON object"}, status_code=400)
    return body, None


def _error_info(exc: Exception) -> dict:
    """Describe an exception from your code (call this inside `except`)."""
    return {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}


def _crash(exc: Exception, hand_end_errors: list | None = None) -> JSONResponse:
    """Report an exception from your code to the judge (with traceback)."""
    return JSONResponse({**_error_info(exc), "hand_end_errors": hand_end_errors or []},
                        status_code=500)


def _deliver_results(raw_results) -> list:
    """Call on_hand_end() for each finished hand the judge sent.
    Returns a list of error reports (one per hand that crashed)."""
    errors = []
    for raw in raw_results or []:
        try:
            on_hand_end(HandResult.from_dict(raw), MEMORY)
        except Exception as exc:
            errors.append(_error_info(exc))
    return errors


@app.get("/")
async def info():
    """Health check: tells the judge who this bot is."""
    return {"participant": PARTICIPANT, "agent_name": AGENT_NAME}


@app.post("/new-half")
async def route_new_half(request: Request):
    """Wipe memory and call your new_half(). The body is {"config": {...}}."""
    body, refused = await _read_request(request, "/new-half")
    if refused:
        return refused
    MEMORY.clear()
    STATUS["started"] = True
    try:
        new_half(AgentConfig.from_dict(body.get("config", {})), MEMORY)
    except Exception as exc:
        return _crash(exc)
    return {"ok": True}


@app.post("/act")
async def route_act(request: Request):
    """Deliver finished hands, then turn the JSON into an Observation, call
    choose_action(), and send the answer back."""
    body, refused = await _read_request(request, "/act")
    if refused:
        return refused
    hand_end_errors = _deliver_results(body.get("hand_results"))
    try:
        obs = Observation.from_dict(body["observation"])
        action = choose_action(obs, MEMORY)
    except Exception as exc:
        return _crash(exc, hand_end_errors)
    # Send the action's name; anything strange is passed on for the judge to reject.
    if isinstance(action, Action):
        answer = action.name
    elif action is None or isinstance(action, (str, int, float, bool)):
        answer = action
    else:
        answer = repr(action)
    return {"action": answer, "hand_end_errors": hand_end_errors}


@app.post("/hand-end")
async def route_hand_end(request: Request):
    """Call on_hand_end() for the last finished hands of a half."""
    body, refused = await _read_request(request, "/hand-end")
    if refused:
        return refused
    return {"ok": True, "hand_end_errors": _deliver_results(body.get("results"))}


if __name__ == "__main__":
    # The tournament sets PORT as an environment variable, so each bot is compliant to our evaluation platform
    # Locally it defaults to 5008 for your own testing convenience
    port = int(os.environ.get("PORT", "5008"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info", access_log=False)
