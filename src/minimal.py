import logging
from pathlib import Path

from decimate import decimate
from register import register
from flow import flow

logging.basicConfig(level=logging.INFO)

moving = Path("data/oasis4-inner.vtk")
fixed = Path("data/oasis4-outer.vtk")

DEFORMETRICA_OUTPUTS = Path("data/outputs/")
DEFORMETRICA_OUTPUTS.mkdir(exist_ok=True)

DECIMATE = 0.75

moving = decimate(moving, target_reduction=DECIMATE)
fixed = decimate(fixed, target_reduction=DECIMATE)

# invoke deformetrica
moving, fixed = register(
    moving,
    fixed,
    output_dir=DEFORMETRICA_OUTPUTS,
)

# invoke chordal cmcf

moving, fixed = flow(
    moving,
    fixed,
)
