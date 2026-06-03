"""Run all Phase-1 reproductions (Fig 1b, Fig 3, Table I) into outputs/.

Pass-through args (e.g. --preset tiny|cpu|paper, --backend) are forwarded to each
script. Example:
    python scripts/run_all.py --preset tiny
    python scripts/run_all.py --preset paper --backend torch   # on the GPU machine
"""

import runpy
import sys
import os

HERE = os.path.dirname(__file__)


def _run(name):
    print(f"\n=== {name} ===", flush=True)
    sys.argv = [name] + PASS
    runpy.run_path(os.path.join(HERE, name), run_name="__main__")


if __name__ == "__main__":
    PASS = sys.argv[1:]
    for script in ("fig1b.py", "fig3.py", "table1.py"):
        _run(script)
    print("\nAll Phase-1 outputs written to outputs/.")
