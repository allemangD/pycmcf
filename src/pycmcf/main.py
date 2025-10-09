from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence, Self

import tqdm


def main():
    args = Args.parse()

    # Slow local imports so that argparse is fast.
    from . import method
    import numpy as np
    import vtk

    READERS = {
        ".vtk": vtk.vtkPolyDataReader,
        ".stl": vtk.vtkSTLReader,
        ".obj": vtk.vtkOBJReader,
        # todo general file type inference
    }

    reader = READERS[args.mesh.suffix](file_name=args.mesh)

    reader = vtk.vtkCleanPolyData(input_connection=reader.output_port)
    reader.ConvertLinesToPointsOff()
    reader.ConvertPolysToLinesOff()
    reader.ConvertStripsToPolysOff()

    reader = vtk.vtkTriangleFilter(input_connection=reader.output_port)

    reader.Update()

    mesh = reader.output

    if args.strip_normals:
        mesh.GetPointData().RemoveArray("normals")

    verts = np.reshape(
        mesh.GetPoints().GetData(),
        (-1, 3),
    )

    faces = np.reshape(
        mesh.GetPolys().GetConnectivityArray(),
        (-1, 3),
    )

    generator = enumerate(method.flow(
        verts,
        faces,
        steps=args.steps,
        rate_0=args.rate_0,
        rate_common_factor=args.rate_factor,
    ))

    writer = vtk.vtkPolyDataWriter(input_data=mesh)

    print(
        {
            "mesh": str(args.mesh),
            "steps": args.steps,
            "rate_0": args.rate_0,
            "rate_factor": args.rate_factor,
        }
    )
    args.output.mkdir(parents=True, exist_ok=True)

    if args.progress:
        generator = tqdm.tqdm(generator, total=args.steps + 1, leave=False)

    for idx, stage in generator:
        writer.file_name = args.output.joinpath(args.mesh.stem + f".cmcf-{idx:03}.vtk")
        np.copyto(np.asarray(mesh.GetPoints().GetData()), stage.verts)
        writer.Update()


class Args:
    __parser = argparse.ArgumentParser()

    mesh: Path
    __parser.add_argument("mesh", type=Path)

    steps: int
    __parser.add_argument(
        "-n",
        "--steps",
        type=int,
        dest="steps",
        required=True,
        help="Number of steps to run CMCF.",
    )

    rate_0: float
    __parser.add_argument(
        "-r",
        "--rate",
        type=float,
        dest="rate_0",
        required=True,
        help="Initial timestep for CMCF flow.",
    )

    rate_factor: float
    __parser.add_argument(
        "-f",
        "--rate-factor",
        type=float,
        dest="rate_factor",
        default=1.4,
        help="Common factor for rate acceleration. Set to 1 to use constant rate.",
    )

    # Some renderers get confused by mangled normals, so strip them by default.
    # In some contexts it may be desirable to transfer them to the ellipsoid
    # domain, so in that case pass `--no-strip-normals`.
    strip_normals: bool
    __parser.add_argument(
        "--strip_normals",
        type=bool,
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove normals from mesh (default).",
    )

    output: Path
    __parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default="./cmcf-out/",
        help="Results are placed in this directory with the name format: <filename>.cmcf-<stage>.vtk",
    )

    progress: bool
    __parser.add_argument(
        '--progress',
        type=bool,
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show a progress bar (default)."
    )

    # todo normalization options

    @classmethod
    def parse(cls, args: Sequence[str] | None = None) -> Self:
        res = cls()
        cls.__parser.parse_args(args, namespace=res)
        return res


if __name__ == "__main__":
    main()
