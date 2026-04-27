# Cortex Flow

Experimental augmentation of CMCF which attempts to preserve cortical thickness
during flow.

Currently, the only implementation file is `batch.py`, which contains a cached
pipeline for quick iteration.

- `original()` reads the source vtk files and normalizes.
- `decimate()` performs an initial cleanup step, also based on CMCF, which
  removes some artifacts in the white matter surface.
- `link()` performs the chord selection procedure.
- `flow_cmcf()` Classic CMCF.
- `flow_link()` CMCF with augmented Laplacian.
- `flow_phased()` CMCF with two-phase constrained update.

Finally, a helper function for generating non-cached outputs:

- `save_anim()` write the flow to an animated `.hdf` file suitable for
  visualization in ParaView.
