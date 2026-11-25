import logging
from pathlib import Path

from backflow import backflow

logging.basicConfig(level=logging.INFO)

# forward-flow states written by flow() in main-6d-flow.py
states = Path("data/oasis4-inner-decimated-flow-states.npz")
print(f"backflow input:\n  {states}")

# R6 is the baseline; R3 (white -> white and pial -> pial) is an ablation
for space in ("R6", "R3"):
    outputs = backflow(states, space=space)
    print(f"backflow {space} outputs:\n  " + "\n  ".join(map(str, outputs)))
