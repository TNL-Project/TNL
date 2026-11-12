# SVM Gradient Benchmark

Benchmark of the gradient of the multiclass linear support vector machine (SVM) on sparse data,
the main operation of its training by first-order or quasi-Newton methods. It measures:

1. the gain of processing several classifiers at once in one pass over the sparse matrix (the
   sparse-dense product with narrow dense matrices instead of sparse matrix-vector products),
2. the gain of evaluating the derivative of the loss directly when the results of the
   reduction in the rows of the sparse matrix are stored.

## Problem

The samples are the rows of the sparse matrix `X` (`n x d`), `K` linear classifiers (the
columns of `W`, `d x K`) are trained by one-vs-rest with the squared hinge loss (L2-loss SVM,
the default multiclass solver of LIBLINEAR). The objective is

```text
sum_c ( |w_c|^2 / 2 + C sum_i max(0, 1 - y_ic x_i . w_c)^2 ),
```

where `y_ic = 1` when the sample `i` belongs to the class `c` and `-1` otherwise. Its gradient is

```text
W + X^T G,   G_ic = -2 C y_ic max(0, 1 - y_ic s_ic),   S = X W.
```

## Methods

- `spmv`: every classifier is processed separately, `K` sparse matrix-vector products
  `s_c = X w_c`, a kernel computing `G` from `S`, `K` products `X^T g_c` and the addition of
  `W` (the matrices are stored by columns).
- `spmm`: all classifiers at once, the sparse-dense products `S = X W` and `X^T G` by
  `TNL::Matrices::sparseDenseMatrixProduct` and the kernels computing `G` and adding `W`.
- `spmm-fused`: the same products by `TNL::Matrices::sparseDenseMatrixProductWithLambdas`,
  the storer of the first product evaluates the derivative of the loss, so `S` is never
  stored, and the storer of the second product adds `W`.

`X^T` is transposed before the measurement, so that both products are the reductions in the
rows of a sparse matrix. The implementation is in `LinearSVMGradient.h`.

## Reference implementations

`svm-reference.py` computes the same gradient by PyTorch with the same initial weights and
appends the results to the same log file (on the CPU and on the GPU, where the products are
computed by cuSPARSE):

- `torch-spmm`: the products of the CSR tensors `X` and `X^T` and the dense matrices,
- `torch-spmv`: the matrix-vector products, one classifier at once.

The initial weights are `0.2 (h - 0.5)`, where `h` from `[0, 1)` is given by the hash
splitmix64 of the index of the element (`getInitialWeights` in `SVMBenchmark.h`).

## Measured quantities

- `time` – the time of the computation of the gradient,
- `gradient norm` – the Frobenius norm of the gradient, which is compared between the
  implementations.

## Input data

`svm-datasets.py` downloads multiclass datasets of the LIBSVM collection: news20 (15,935
samples, 62,061 features, 20 classes), sector (6,412 samples, 55,197 features, 105 classes)
and rcv1 (the test set of rcv1.multiclass, 518,571 samples, 47,236 features, 53 classes). It
stores the samples in the Matrix Market format and the classes in the file with the
extension `.labels`. The number of classes is given by `--classes`, the classes of the
samples are taken modulo this number (0 means the classes of the dataset).

## Usage

```bash
python3 src/Benchmarks/SVM/svm-datasets.py --output-dir data --datasets news20 rcv1
build/cuda/bin/tnl-benchmark-svm-cuda --input-file data/rcv1.mtx --log-file svm.log
python3 src/Benchmarks/SVM/svm-reference.py --input-file data/rcv1.mtx --log-file svm.log
```
