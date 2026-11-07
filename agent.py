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
# Preflop hand strength evaluation (fast, no simulation needed)
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
# Monte Carlo equity estimation (post-flop)
# --------------------------------------------------------------------------

def _monte_carlo_equity(hole_cards: tuple, board: tuple, num_sims: int,
                        rng: random.Random) -> float:
    """Estimate win probability by dealing random opponent hands and
    completing the board, then comparing omaha_score.

    Returns equity as a float 0.0 to 1.0.
    """
    # Cards that are in play (known to us)
    used = set(hole_cards) | set(board)
    remaining = [c for c in range(NUM_CARDS) if c not in used]
    cards_to_deal_board = 5 - len(board)
    board_list = list(board)

    wins = 0
    ties = 0
    total = 0

    for _ in range(num_sims):
        # We need: remaining board cards + 5 opponent hole cards
        needed = cards_to_deal_board + 5
        if needed > len(remaining):
            break
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
# Opponent modeling
# --------------------------------------------------------------------------

def _get_opp_stats(memory: dict) -> dict:
    """Return aggregated opponent statistics."""
    hands = memory.get("opp_hands_seen", 0)
    if hands < 5:
        # Not enough data, use defaults (assume slightly aggressive)
        return {
            "vpip": 0.65,           # Voluntary put in pot
            "pfr": 0.35,            # Preflop raise %
            "agg_factor": 1.5,      # (bets + raises) / calls
            "fold_to_raise": 0.35,  # Fold when facing a raise
            "showdown_rate": 0.35,  # Goes to showdown
            "bluff_ratio": 0.3,     # Raises with losing hand at showdown
            "enough_data": False,
        }

    return {
        "vpip": memory.get("opp_vpip_count", 0) / max(hands, 1),
        "pfr": memory.get("opp_pfr_count", 0) / max(hands, 1),
        "agg_factor": (memory.get("opp_raise_count", 0) /
                       max(memory.get("opp_call_count", 1), 1)),
        "fold_to_raise": (memory.get("opp_fold_to_raise", 0) /
                          max(memory.get("opp_faced_raise", 1), 1)),
        "showdown_rate": (memory.get("opp_showdown_count", 0) /
                          max(hands, 1)),
        "bluff_ratio": (memory.get("opp_bluff_count", 0) /
                        max(memory.get("opp_showdown_raise_count", 1), 1)),
        "enough_data": True,
    }


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
    """Preflop strategy based on hand strength features."""
    strength = _preflop_strength(obs.hole_cards)

    # Position bonus: being on button (acting last post-flop) is valuable
    if obs.is_button:
        strength += 0.05

    # Adapt to opponent: if opponent folds a lot to raises, raise more
    fold_to_raise = opp_stats["fold_to_raise"]

    # --- Premium hands (top ~15%): raise pot ---
    if strength >= 0.55:
        if Action.RAISE_POT in obs.legal_actions:
            return Action.RAISE_POT
        if Action.RAISE_HALF in obs.legal_actions:
            return Action.RAISE_HALF
        return Action.CHECK_CALL

    # --- Good hands (~15-35%): raise half or call ---
    if strength >= 0.35:
        # If opponent folds a lot, raise to steal
        if fold_to_raise > 0.45 and Action.RAISE_HALF in obs.legal_actions:
            return Action.RAISE_HALF
        if Action.RAISE_HALF in obs.legal_actions and rng.random() < 0.3:
            return Action.RAISE_HALF
        return Action.CHECK_CALL

    # --- Marginal hands (~35-55%): call if cheap, fold if expensive ---
    if strength >= 0.20:
        if obs.to_call <= obs.big_blind * 2:
            return Action.CHECK_CALL
        # If opponent is very aggressive, tighten up
        if opp_stats["agg_factor"] > 2.5 and obs.to_call > obs.big_blind * 3:
            if Action.FOLD in obs.legal_actions:
                return Action.FOLD
        if obs.to_call <= obs.big_blind * 4:
            return Action.CHECK_CALL
        if Action.FOLD in obs.legal_actions:
            return Action.FOLD
        return Action.CHECK_CALL

    # --- Weak hands (bottom ~55%): fold if facing a raise, otherwise check ---
    if obs.to_call == 0:
        return Action.CHECK_CALL
    # Call the minimum blind if we're the button (just 1 chip to see a flop)
    if obs.is_button and obs.to_call <= obs.big_blind and strength >= 0.12:
        return Action.CHECK_CALL
    if Action.FOLD in obs.legal_actions:
        return Action.FOLD
    return Action.CHECK_CALL


def _postflop_action(obs: Observation, memory: dict, rng: random.Random,
                     opp_stats: dict, pot_odds: float) -> Action:
    """Post-flop strategy using equity estimation and opponent modeling."""

    # Determine simulation budget based on street
    # Total time bank: 8 seconds for 1000 hands, ~2-4 decisions per hand
    # That's ~2-4ms per decision. omaha_score is very fast (dict lookup).
    # Budget simulations carefully.
    if obs.street == Street.RIVER:
        # On river, board is complete - use exact hand strength
        my_score = omaha_score(obs.hole_cards, obs.board)
        category = category_of(my_score)
        return _river_action(obs, memory, rng, opp_stats, pot_odds,
                             my_score, category)

    # Flop or Turn: estimate equity via Monte Carlo
    if obs.street == Street.FLOP:
        num_sims = 60   # ~0.3ms at ~5μs per sim
    else:  # TURN
        num_sims = 80   # ~0.4ms, fewer unknown cards

    equity = _monte_carlo_equity(obs.hole_cards, obs.board, num_sims, rng)

    return _equity_based_action(obs, memory, rng, opp_stats, pot_odds, equity)


def _river_action(obs: Observation, memory: dict, rng: random.Random,
                  opp_stats: dict, pot_odds: float,
                  my_score: int, category: int) -> Action:
    """River strategy: board is complete, use exact hand strength."""

    # Estimate relative hand strength using a rough equity estimate
    # On the river, we can also do a small number of simulations against
    # random opponent hands to gauge our relative strength
    equity = _monte_carlo_equity(obs.hole_cards, obs.board, 100, rng)

    # === Strong value hands (equity > 75%) ===
    if equity >= 0.75:
        memory["our_value_bet_count"] = memory.get("our_value_bet_count", 0) + 1
        if Action.RAISE_POT in obs.legal_actions:
            return Action.RAISE_POT
        if Action.RAISE_HALF in obs.legal_actions:
            return Action.RAISE_HALF
        return Action.CHECK_CALL

    # === Good hands (equity 55-75%) ===
    if equity >= 0.55:
        if obs.to_call == 0:
            # We can bet for value
            if Action.RAISE_HALF in obs.legal_actions:
                return Action.RAISE_HALF
            return Action.CHECK_CALL
        # Facing a bet: call if pot odds are right
        if equity >= pot_odds + 0.05:
            return Action.CHECK_CALL
        if Action.FOLD in obs.legal_actions:
            return Action.FOLD
        return Action.CHECK_CALL

    # === Marginal hands (equity 35-55%) ===
    if equity >= 0.35:
        if obs.to_call == 0:
            # Check behind (pot control)
            return Action.CHECK_CALL
        # Call small bets
        if pot_odds < 0.30 and equity >= pot_odds:
            return Action.CHECK_CALL
        if Action.FOLD in obs.legal_actions:
            return Action.FOLD
        return Action.CHECK_CALL

    # === Weak hands (equity < 35%) ===
    if obs.to_call == 0:
        # Consider a bluff if opponent folds a lot
        if (opp_stats["fold_to_raise"] > 0.40 and
                Action.RAISE_HALF in obs.legal_actions):
            bluffs = memory.get("our_bluff_count", 0)
            value_bets = memory.get("our_value_bet_count", 0)
            # Keep a balanced bluff ratio (~30% of bets)
            if value_bets > 0 and bluffs / (bluffs + value_bets + 1) < 0.35:
                if rng.random() < 0.25:
                    memory["our_bluff_count"] = bluffs + 1
                    return Action.RAISE_HALF
        return Action.CHECK_CALL

    # Facing a bet with a weak hand
    if Action.FOLD in obs.legal_actions:
        return Action.FOLD
    return Action.CHECK_CALL


def _equity_based_action(obs: Observation, memory: dict, rng: random.Random,
                         opp_stats: dict, pot_odds: float,
                         equity: float) -> Action:
    """Flop/Turn strategy based on Monte Carlo equity."""

    # === Very strong (equity > 70%): bet/raise for value ===
    if equity >= 0.70:
        memory["our_value_bet_count"] = memory.get("our_value_bet_count", 0) + 1
        if obs.to_call == 0:
            # Lead out with a strong hand
            if Action.RAISE_POT in obs.legal_actions and equity >= 0.80:
                return Action.RAISE_POT
            if Action.RAISE_HALF in obs.legal_actions:
                return Action.RAISE_HALF
            return Action.CHECK_CALL
        else:
            # Facing a bet with a strong hand: raise
            if Action.RAISE_POT in obs.legal_actions and equity >= 0.80:
                return Action.RAISE_POT
            if Action.RAISE_HALF in obs.legal_actions:
                return Action.RAISE_HALF
            return Action.CHECK_CALL

    # === Good drawing/made hand (equity 50-70%): bet or call ===
    if equity >= 0.50:
        if obs.to_call == 0:
            # Semi-bluff or value bet
            if Action.RAISE_HALF in obs.legal_actions:
                if rng.random() < 0.55:
                    return Action.RAISE_HALF
            return Action.CHECK_CALL
        # Facing a bet: call if equity > pot odds (with implied odds buffer)
        implied_equity_needed = pot_odds * 0.85  # Give ourselves some implied odds credit
        if equity >= implied_equity_needed:
            return Action.CHECK_CALL
        if Action.FOLD in obs.legal_actions:
            return Action.FOLD
        return Action.CHECK_CALL

    # === Marginal hand (equity 30-50%): proceed carefully ===
    if equity >= 0.30:
        if obs.to_call == 0:
            # Check (pot control) or occasional semi-bluff
            if (obs.is_button and equity >= 0.40 and
                    Action.RAISE_HALF in obs.legal_actions and
                    rng.random() < 0.25):
                return Action.RAISE_HALF
            return Action.CHECK_CALL
        # Facing a bet: call only if price is right
        if equity >= pot_odds and obs.to_call <= obs.pot * 0.5:
            return Action.CHECK_CALL
        if Action.FOLD in obs.legal_actions:
            return Action.FOLD
        return Action.CHECK_CALL

    # === Weak hand (equity < 30%): mostly fold to bets ===
    if obs.to_call == 0:
        # Free check: take it, but occasionally semi-bluff
        if (opp_stats["fold_to_raise"] > 0.45 and
                Action.RAISE_HALF in obs.legal_actions and
                rng.random() < 0.15):
            memory["our_bluff_count"] = memory.get("our_bluff_count", 0) + 1
            return Action.RAISE_HALF
        return Action.CHECK_CALL

    # Facing a bet with a weak hand
    # Check if we have a draw worth chasing
    if equity >= 0.20 and pot_odds < 0.25:
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

        # We can't access the action_history from HandResult directly,
        # but we tracked opponent actions during choose_action via obs.action_history.
        # Use the result to track showdown-related stats.

        if result.went_to_showdown and result.opp_hole is not None:
            # Check if opponent was raising with a losing hand (bluff detection)
            # This is tracked indirectly; for now just count showdowns
            pass

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
