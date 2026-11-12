# Dense-Sparse Product Benchmark

Benchmark of the product `C = B A` of a dense matrix `B` (`k x n`) and a sparse matrix `A`
(`n x m`, stored by rows) computed by `TNL::Matrices::denseSparseMatrixProduct` and compared with
the product computed by the transpositions and the sparse-dense product, `C = (A^T B^T)^T`.

## Methods

- `transpose-all`: the transpositions of `B` and `A`, the sparse-dense product
  `C^T = A^T B^T` by `TNL::Matrices::sparseDenseMatrixProduct` and the transposition of `C^T`.
- `transpose-dense`: the same with `A^T` computed before the measurement, which is the case
  of the repeated products with the same sparse matrix.
- `transposed-A`: `denseSparseMatrixProduct( B, A^T, C, 1, 0, TransposeState::Transpose )`
  with `A^T` computed before the measurement. `C^T = A^T B^T` is computed by the reduction in
  the rows of `A^T`, `B` is read and `C` is written in the transposed order, so the dense
  matrices are not transposed.
- `atomic`: `denseSparseMatrixProduct( B, A, C )` with `A` stored by rows, every row of `A`
  and row of `B` is processed by one thread, which adds the products to the columns of `C`
  by atomic additions.

The dense matrices are row-major or column-major (`--organization`). The elements of `B` are
`h - 0.5`, where `h` from `[0, 1)` is given by the hash splitmix64 of the index of the element
(`getDenseMatrix` in `DenseSparseBenchmark.h`). The sum of the elements of `C` (`checksum`)
is compared between the methods.

## Reference implementations

`dense-sparse-reference.py` computes the product by PyTorch with the CSR tensor `A` and the
row-major `B` and appends the results to the same log file:

- `torch` – `B @ A`,
- `torch-transpose-dense` – `(A^T @ B^T)^T` with the precomputed `A^T` and the row-major copy
  of `B^T` (cuSPARSE SpMM on GPUs), the result is copied to a row-major tensor.

## Usage

```bash
build/cuda/bin/tnl-benchmark-dense-sparse-cuda --input-file A.mtx --log-file dense-sparse.log
python3 src/Benchmarks/DenseSparse/dense-sparse-reference.py --input-file A.mtx \
   --log-file dense-sparse.log
```
