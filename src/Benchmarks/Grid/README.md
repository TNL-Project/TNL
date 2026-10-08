# Grid Benchmark

This benchmark measures the performance of entity traversals in structured grids, i.e. the `TNL::Meshes::Grid` class.
For every grid dimension, an executable is built which creates a grid of the given resolution and measures the time
needed to traverse all entities of every entity dimension (vertices, edges, faces, cells) using three traversal
variants:

1. `forAllEntities` — traversal of all entities.
2. `forInteriorEntities` — traversal of interior entities only.
3. `forBoundaryEntities` — traversal of boundary entities only.

Each traversal is timed with a set of simple per-entity operations (see `Operations.h`) to assess the cost of the
traversal itself as well as the cost of querying the entities:

1. An empty operation measuring the pure traversal overhead.
2. `entity.isBoundary()`
3. `entity.getCoordinates()`
4. `entity.getIndex()`
5. `entity.getNormals()`
6. `entity.refresh()`
7. `entity.getMesh().getDimensions()`
8. `entity.getMesh().getOrigin()`
9. `entity.getMesh().getEntitiesCounts()`

The benchmark supports single and double precision and can run on all devices supported by TNL (sequential, host,
CUDA, HIP), either selected individually or all at once.

## Executables

One executable is built per grid dimension:

| Executable | Grid dimension | Resolution options used |
| ---------- | -------------- | ----------------------- |
| `tnl-benchmark-grid-1D` | 1D | `--x-dimension` |
| `tnl-benchmark-grid-2D` | 2D | `--x-dimension`, `--y-dimension` |
| `tnl-benchmark-grid-3D` | 3D | `--x-dimension`, `--y-dimension`, `--z-dimension` |

The executables are built from the same sources for all backends: the host build uses `tnl-benchmark-grid.cpp`,
the CUDA build uses `tnl-benchmark-grid.cu`, and the HIP build uses `tnl-benchmark-grid.hip`. There are no separate
executables for CUDA or HIP — the target names stay the same, only the compiled source differs.

## Command-line options

Options specific to this benchmark:

| Option | Description | Default |
|--------|-------------|---------|
| `--x-dimension <n>` | Grid resolution in the x dimension | `100` |
| `--y-dimension <n>` | Grid resolution in the y dimension (used by the 2D and 3D executables) | `100` |
| `--z-dimension <n>` | Grid resolution in the z dimension (used by the 3D executable) | `100` |
| `--precision <type>` | Precision of the arithmetics (`float`, `double`, or `all`) | `double` |
| `--device <device>` | Device the computation will run on (`host`, `sequential`, `cuda`, `hip`, or `all`) | `all` |

In addition, the general benchmark options (`--loops`, `--min-time`, `--warmup-loops`, `--log-file`, etc.) and the
device setup options (`--openmp-max-threads`, `--cuda-device`, `--hip-device`, etc.) are available. A complete list
can be obtained with the `--help` command line argument.

## Usage

```bash
just build tnl-benchmark-grid-3D
./build/bin/tnl-benchmark-grid-3D --x-dimension 200 --y-dimension 200 --z-dimension 200
```
