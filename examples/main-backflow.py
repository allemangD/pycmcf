import logging
from pathlib import Path

from backflow import backflow

logging.basicConfig(level=logging.INFO)

data_root = Path("data/")

# flow() inputs (the chords) and outputs (the flowed chords) from main-6d-flow.py
u = data_root.joinpath("oasis4-inner-decimated.vtk")
v = data_root.joinpath("oasis4-outer-decimated-recon.vtk")
u_flow = data_root.joinpath("oasis4-inner-decimated-flow.vtk")
v_flow = data_root.joinpath("oasis4-outer-decimated-recon-flow.vtk")
print(f"backflow inputs:\n  {u}\n  {v}\n  {u_flow}\n  {v_flow}")

u_back, v_back = backflow(u, v, u_flow, v_flow)
print(f"backflow outputs:\n  {u_back}\n  {v_back}")
