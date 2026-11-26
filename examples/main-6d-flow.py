import logging
from pathlib import Path

from decimate import decimate
from flow import flow
from register import register

logging.basicConfig(level=logging.INFO)

data_root = Path("data/")
aux_outputs = data_root.joinpath("outputs")
aux_outputs.mkdir(exist_ok=True)

moving = data_root.joinpath("oasis4-inner.vtk")
fixed = data_root.joinpath("oasis4-outer.vtk")
print(f"pipeline inputs:\n  {moving}\n  {fixed}")

moving = decimate(moving, target_reduction=0.90)
fixed = decimate(fixed, target_reduction=0.90)
print(f"decimation outputs:\n  {moving}\n  {fixed}")

moving, fixed = register(moving, fixed, output_dir=aux_outputs)
print(f"deformetrica outputs:\n  {moving}\n  {fixed}")

# the default stop_speed (1e-3) stops while the surface is still an ellipsoid
moving, fixed, states = flow(moving, fixed)
print(f"chordal flow outputs:\n  {moving}\n  {fixed}\n  {states}")
