"""Compare occurrence indexing and local subtree counts with exhaustive answers.

    python3 experiments/check_gpu_index.py [--binary build/G-HUIM]
"""

import argparse
import itertools
from pathlib import Path
import random
import re
import subprocess
import tempfile


def answers(transactions, minimum):
    items = sorted({i for t in transactions for i in t})
    result = {}
    for n in range(1, len(items) + 1):
        for pattern in itertools.combinations(items, n):
            utility = sum(sum(t[i] for i in pattern) for t in transactions if all(i in t for i in pattern))
            if utility >= minimum:
                result[pattern] = utility
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path("build/G-HUIM"))
    args = parser.parse_args()
    rng = random.Random(743)
    random_txs = []
    for _ in range(400):
        items = sorted(rng.sample(range(1, 9), rng.randint(1, 8)))
        random_txs.append({i: rng.randint(0, 20) for i in items})
    cases = [
        ("single", [{i: i for i in range(1, 9)}], 15),
        ("merged", [{1: 3, 2: 0, 4: 7, 5: 5}, {1: 2, 2: 0, 4: 1, 5: 6}, {1: 9, 3: 4}], 12),
        ("sparse", [{1: 4, 2: 9}, {3: 12, 4: 8}, {1: 0, 5: 7}, {2: 2, 6: 3, 8: 9}], 7),
        ("huge", random_txs, 1200),
        ("pruned", random_txs, 4000),
        ("empty", random_txs, 1000000),
    ]
    configs = [
        ("index-all-output", ["--index-min-trans", "1"], True),
        ("index-large-output", ["--index-min-trans", "64"], True),
        ("static-index-output", ["--index-min-trans", "1", "--method", "static", "--block-size", "64"], True),
        ("local-four", ["--local-tail-items", "8"], False),
        ("indexed-local-warp", ["--index-min-trans", "1", "--local-tail-items", "8", "--local-tail-trans", "32"], False),
        ("indexed-local-one", ["--index-min-trans", "1", "--local-tail-items", "8", "--local-tail-trans", "1"], False),
        ("count-default", [], False),
        ("count-no-tail", ["--local-tail-items", "0"], False),
        ("count-wide-tail", ["--local-tail-items", "31", "--local-tail-trans", "32"], False),
        ("count-subset-limit", ["--local-tail-items", "12", "--local-tail-trans", "1"], False),
        ("count-static", ["--method", "static", "--block-size", "64"], False),
    ]
    with tempfile.TemporaryDirectory(prefix="ghuim-index-") as folder:
        folder = Path(folder)
        for name, transactions, minimum in cases:
            expected = answers(transactions, minimum)
            dataset = folder / f"{name}.txt"
            dataset.write_text("".join(
                f"{' '.join(map(str, t))}:{sum(t.values())}:{' '.join(map(str, t.values()))}\n"
                for t in transactions
            ))
            for config, extra, output in configs:
                destination = folder / f"{name}-{config}.txt"
                command = [str(args.binary), "--input", str(dataset), "--minutil", str(minimum),
                           "--total-memory", "1G"] + extra
                if output:
                    command += ["--output", str(destination)]
                run = subprocess.run(command, text=True, capture_output=True, timeout=60)
                text = run.stdout + run.stderr
                assert run.returncode == 0, (command, text)
                count = re.search(r"Number of patterns: (\d+)", text)
                assert count and int(count[1]) == len(expected), (name, config, len(expected), text)
                if output:
                    # The existing miner returns before opening output when root pruning
                    # leaves no tasks. In that case a missing file also means no itemsets.
                    if not destination.exists():
                        assert not expected, (name, config, "missing output")
                        continue
                    actual = {}
                    for line in destination.read_text().splitlines():
                        keys, utility = line.split("UTIL")
                        pattern = tuple(sorted(map(int, keys.split())))
                        assert pattern not in actual, (name, config, "duplicate", pattern)
                        actual[pattern] = int(utility)
                    assert actual == expected, (name, config, "wrong itemsets or utilities")
            print(f"{name}: {len(expected)} patterns, all {len(configs)} configurations match", flush=True)
    print(f"Passed {len(cases) * len(configs)} exhaustive comparisons")


if __name__ == "__main__":
    main()
