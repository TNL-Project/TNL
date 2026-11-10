# GNN Inference Benchmark

Benchmark of the inference of two-layer graph neural networks implemented by the
sparse-dense products of TNL.

## Models

- `gcn` – the graph convolutional network of Kipf and Welling,
  `H' = act( A_hat * H * W + b )` with `A_hat = D^{-1/2} ( A + I ) D^{-1/2}`.
- `sage` – GraphSAGE with the mean aggregation (`SAGEConv` of PyTorch Geometric),
  `H' = act( mean_{N(i)}( H ) * W_l + H * W_r + b )`.

The first layer uses ReLU, the second one no activation. The row `i` of the adjacency matrix
contains the sources of the edges ending in the node `i`, its values are ignored.

## Implementation

The features are transformed first (`T = H * W`), then aggregated by
`TNL::Matrices::sparseDenseMatrixProductWithLambdas` with `T` given by the dense matrix (so
it is read by vectorized loads) and the result stored by a lambda function, which adds the
bias (and the root term `H * W_r` of GraphSAGE) and applies the activation. Hence the
aggregated features are never stored. The normalization of the adjacency matrix is computed
once in the preprocessing and stored in the values of the sparse matrix.

The dense products are computed by `TNL::Matrices::getMatrixProduct` (`TNL`) or by cuBLAS on
GPUs and Eigen on the host (`TNL + cuBLAS`, `TNL + Eigen`). The rows with `part = aggregation`
give the time of the aggregations only.

The implementation is in `GNNLayers.h`.

## Reference implementations

`gnn-reference.py` computes the same models with the same features and weights, which are
given by the hash function `hashUniform` (implemented in both files), and appends the
results to the same log file:

- `PyTorch` – the models written with `torch.sparse.mm` on the CSR tensor of the normalized
  adjacency matrix.
- `PyG` – `GCNConv` (`cached=True`) and `SAGEConv` of PyTorch Geometric with `edge_index`
  (message passing by gather and scatter).
- `PyG sparse` – the same layers with the adjacency matrix given by a `torch.sparse` CSR
  tensor.

The checksums `output sum` and `output norm` of the outputs are logged by all implementations.

## Input graphs

Any square Matrix Market file, e.g. from the SuiteSparse collection. `gnn-datasets.py`
exports the graphs of the datasets Cora, PubMed and Reddit of PyTorch Geometric.

## Usage

```bash
build/cuda/bin/tnl-benchmark-gnn-cuda --input-file graph.mtx --input-features 128 \
   --hidden-features 64 --output-features 16 --log-file gnn.log
python3 src/Benchmarks/GNN/gnn-reference.py --input-file graph.mtx --log-file gnn.log
```
