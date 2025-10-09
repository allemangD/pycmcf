"""
Auxiliary script to render a batch of meshes to PNG via https://f3d.app//

For example, to render a batch of cortex surfaces:

```
python f3d-render.py \
    cmcf-out \
    --filename=0 --up=+Z --grid=0 --axis=0 \
    --ambient-occlusion=1 --tone-mapping=0 --backface-type=hidden \
    --camera-direction=1,0,0 --camera-position=-300,0,0 --camera-orthographic=1
```

See https://f3d.app/doc/user/OPTIONS.html for a comprehensive list of options.

The image sequence may then be animated with ffmpeg.
"""

import concurrent.futures
import sys
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path
from subprocess import run

_, root, *args = sys.argv

root = Path(root)

with ThreadPoolExecutor(max_workers=4) as ex:
    for path in root.glob('*.vtk'):
        output = path.with_suffix('.png')
        fut: concurrent.futures.Future = ex.submit(
            run,
            [
                'f3d',
                path,
                '--output',
                output,
                *args,
            ]
        )
        fut.add_done_callback(lambda _, output=output: print(output))
