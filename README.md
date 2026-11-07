# Stallion - Heads-Up Short Deck 5-Card PLO Poker Bot

**Team**: Team Mustangs  
**Agent Name**: Stallion  
**Challenge**: TAMU Datathon Poker Bot Challenge  

---

## Overview

Stallion is a high-performance Python poker bot designed for **Heads-Up Short Deck 5-Card Pot-Limit Omaha (PLO5)**. It communicates with the referee engine via FastAPI/Uvicorn, operates under strict 1-CPU / 4GB resource limits, and adheres to the 8-second time bank per 1,000 hands.

### Core Strategy Highlights

1. **Short Deck & PLO5 Awareness**:
   - Accounts for modified hand rankings: **Flush > Full House** and **Three of a Kind > Straight**.
   - Strictly enforces the Omaha rule (making hands using exactly 2 hole cards and 3 board cards).

2. **Preflop Evaluation Engine (<0.1 ms)**:
   - Heuristic ranking analyzing high cards (Aces/Kings/Queens), pair and trips value, suit distribution (bonuses for double-suited, penalties for 4+ suited), and rundown connectivity (including A-6-7-8-9 wheel wraps).
   - Positional awareness adjusting ranges when holding the button.

3. **Monte Carlo Equity & Pot Odds Engine**:
   - Fast sampling against random opponent holdings on flop and turn using the pre-computed bitmask evaluator (`omaha_score`).
   - Compares win equity against pot odds: `to_call / (pot + to_call)` to identify profitable calls, value raises, and semi-bluffs.
   - Exact scoring on river showdowns.

4. **Adaptive Opponent Profiling**:
   - Tracks voluntary put-in-pot (VPIP), aggression factor, and fold-to-raise tendencies across hands in `memory`.
   - Steals blinds against overly tight bots and punishes over-aggressive bots.

5. **Fault Tolerance**:
   - Zero exception tolerance: all decision paths are wrapped in safe fallbacks (`CHECK_CALL` or guaranteed legal actions) to prevent match forfeitures.

---

## Performance Benchmarks

Tested via `judge_engine.py` with duplicate format (seats swapped):

| Opponent | Deals | Result | 95% Confidence Interval | Thinking Time (8s bank) | Errors | Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **RandomBot** | 200 | **+8.488 bb/deal** | `[+5.392, +11.583]` | 0.35s | 0 | **WIN (1.0)** |
| **CallingStation** | 200 | **+9.615 bb/deal** | `[+7.697, +11.533]` | 0.55s | 0 | **WIN (1.0)** |
| **AlwaysHalf** | 200 | **+6.700 bb/deal** | `[+3.413, +9.987]` | 0.46s | 0 | **WIN (1.0)** |
| **AlwaysPot** | 1,000 | **+3.163 bb/deal** | `[+1.687, +4.639]` | 1.58s | 0 | **WIN (1.0)** |

- **Unit tests**: `pytest -q` → 12/12 passed in 2.98s
- **Smoke test**: `python tools/smoke_test.py` → PASSED (0–2 ms per request)

---

## Quick Start

### 1. Requirements
- Python 3.10+
- Dependencies:
  ```bash
  pip install -r requirements.txt
  pip install -r requirements-dev.txt
  ```

### 2. Run Tests
```bash
python -m pytest -q
```

### 3. Run Locally Against Baseline Bots
In window 1 (your bot):
```bash
python agent.py
```

In window 2 (opponent):
```bash
python baseline-bots/baseline_random.py
```

In window 3 (judge):
```bash
python judge_engine.py --deals 200
```
