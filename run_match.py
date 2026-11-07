#!/usr/bin/env python3
"""
run_match.py - One-command match runner.

Starts both bot servers in the background, runs judge_engine,
prints the full results, and terminates the background bots cleanly.

Usage:
    python3 run_match.py
    python3 run_match.py --opponent random      # default
    python3 run_match.py --opponent calling     # CallingStation
    python3 run_match.py --opponent half        # AlwaysHalf
    python3 run_match.py --opponent pot         # AlwaysPot
    python3 run_match.py --deals 100            # custom number of deals
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

OPPONENTS = {
    "random": "baseline-bots/baseline_random.py",
    "calling": "baseline-bots/baseline_calling_station.py",
    "half": "baseline-bots/baseline_always_half.py",
    "pot": "baseline-bots/baseline_always_pot.py",
}

def main():
    parser = argparse.ArgumentParser(description="Run a poker match between your bot and an opponent in one command.")
    parser.add_argument("--opponent", choices=list(OPPONENTS.keys()), default="random",
                        help="Which opponent to play against (default: random)")
    parser.add_argument("--deals", type=int, default=50,
                        help="Number of deals to play per half (default: 50)")
    parser.add_argument("--seed", type=int, default=7,
                        help="Match seed for deterministic cards (default: 7)")
    args = parser.parse_args()

    opp_file = OPPONENTS[args.opponent]
    python_cmd = sys.executable

    print(f"\n==================================================")
    print(f" Starting Match: Stallion vs {args.opponent.title()} ({args.deals} deals)")
    print(f"==================================================")

    # Start bot 1 (Stallion) on port 5008
    env1 = os.environ.copy()
    env1["PORT"] = "5008"
    p1 = subprocess.Popen(
        [python_cmd, "agent.py"],
        env=env1,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    # Start bot 2 (Opponent) on port 5009
    env2 = os.environ.copy()
    env2["PORT"] = "5009"
    p2 = subprocess.Popen(
        [python_cmd, opp_file],
        env=env2,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    # Wait a moment for servers to start
    time.sleep(1.0)

    try:
        # Run judge
        judge_cmd = [
            python_cmd, "judge_engine.py",
            "--deals", str(args.deals),
            "--seed", str(args.seed),
            "--agent1", "5008",
            "--agent2", "5009",
        ]
        proc = subprocess.run(judge_cmd)
    finally:
        # Cleanly shut down both servers
        p1.terminate()
        p2.terminate()
        p1.wait()
        p2.wait()
        print("\nAll bot processes shut down cleanly.\n")

if __name__ == "__main__":
    main()
