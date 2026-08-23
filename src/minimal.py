import logging
from pathlib import Path

from decimate import decimate
from register import register

logging.basicConfig(level=logging.INFO)

PATHS = {
    "moving": Path("data/oasis4-inner.vtk"),
    "fixed": Path("data/oasis4-outer.vtk"),
}

DEFORMETRICA_OUTPUTS = Path("data/outputs/")
DEFORMETRICA_OUTPUTS.mkdir(exist_ok=True)

DECIMATE = 0.95

# downsample meshes
for key in list(PATHS):
    src = PATHS[key]
    dst = src.with_stem(f"{src.stem}-decimated")
    print(f"downsampling {src} -> {dst}")
    decimate(src, dst, target_reduction=DECIMATE)
    PATHS[key] = dst

# invoke deformetrica
PATHS["moving"], PATHS["fixed"] = register(
    PATHS["moving"],
    PATHS["fixed"],
    output_dir=DEFORMETRICA_OUTPUTS,
)

# invoke chordal cmcf
