# Grid Register Pressure Benchmark

This benchmark checks whether the traversal of the cells of `TNL::Meshes::Grid` costs registers (or other resources)
in kernels which need many registers themselves.
In each interior cell of a 3D grid, the kernel loads the values in the cell and in its six neighbours,
initializes $K$ local variables from them and updates the variables in four rounds,
each update depending on two other variables, so that all $K$ variables stay alive at the same time.
$K$ is a template parameter and the loops are unrolled, so that the variables are stored in registers on GPUs.
The result is a weighted sum of the variables.

The same kernel is evaluated by two implementations:

1. `parallel-for` — `TNL::Algorithms::parallelFor` over plain multi-indices with linear indexing of the neighbours.
2. `grid` — `TNL::Meshes::forInteriorEntities` with the neighbours given by `getAdjacentEntityIndex`.

The benchmark runs $K = 8, 16, 32, 48, 64, 96$, and $128$ for each selected precision and device.
Both implementations must give the same results;
the maximal difference is logged in the metadata column `max difference`.
On GPUs, it may differ by a few units in the last place,
because the compiler contracts the operations to fused multiply-add instructions differently in the two kernels.

The registers, spills, and stack frames of the kernels can be inspected by compiling the CUDA executable
with `-Xptxas -v` (or `--resource-usage`).
For example, on NVIDIA A100 in double precision, both implementations used the same number of registers
(up to 255 registers at $K = 128$, where both spill to the local memory) and took the same time for all $K$.

## Usage

```bash
just build tnl-benchmark-grid-register-pressure
./build/bin/tnl-benchmark-grid-register-pressure
```

## Command-line options

| Option | Description | Default |
|--------|-------------|---------|
| `--size` | Number of cells of the 3D grid along each axis | `256` |
| `--max-variables` | Maximal number of local variables $K$ in each cell | `128` |
| `--precision` | Precision of the arithmetics: `float`, `double`, or `all` | `double` |
| `--device` | Device the computation will run on: `sequential`, `host`, `cuda`, `hip`, or `all` | `all` |

In addition, the benchmark accepts the common options of the `TNL::Benchmarks::Benchmark` framework
(such as `--loops`, `--log-file`, or `--verbose`)
and the device setup options of `TNL::Devices::Host` and `TNL::Devices::GPU`.
