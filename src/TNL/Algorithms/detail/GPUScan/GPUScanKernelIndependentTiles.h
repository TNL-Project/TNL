// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

/* GPUScanKernelIndependentTiles - independent scan of the input tiles (the
 * first phase of the "scan-then-propagate" strategy).
 *
 * Each CUDA block scans its tile of blockSize * valuesPerThread elements with
 * GPUTileScan, writes the scanned tile to the output, and stores the block
 * aggregate in blockResults[blockIdx.x] (if blockResults is not null). The
 * tiles are scanned independently, so when there are more blocks, the block
 * results must be scanned and added to the output by GPUScanKernelUniformShift.
 *
 * Usage (see GPUScanKernelLauncher in GPUScan.h):
 * - with a single block when the whole input fits into one tile; then it
 *   performs the complete scan (valuesPerThread is reduced for small inputs),
 * - the first phase of the ScanPhaseType::WriteInFirstPhase variant when the
 *   input is larger than one tile and either it does not fit into one CUDA
 *   grid or the scan is performed in two phases explicitly (the distributed
 *   scan).
 *
 * Global memory traffic of the two-phase variant (IndependentTiles +
 * UniformShift) is 2n reads and 2n writes. The input is read only once, so the variant is
 * suitable for expensive expressions (see ScanPhaseType).
 *
 * References:
 * M. Harris, S. Sengupta, J. D. Owens, "Parallel Prefix Sum (Scan) with CUDA",
 * GPU Gems 3, chapter 39, Addison-Wesley, 2007.
 * https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch39.html
 * S. Sengupta, M. Harris, Y. Zhang, J. D. Owens, "Scan Primitives for GPU
 * Computing", Graphics Hardware 2007, pp. 97-106.
 * https://diglib.eg.org/handle/10.2312/EGGH.EGGH07.097-106
 */

#include <TNL/Backend.h>
#include <TNL/Algorithms/detail/ScanType.h>
#include <TNL/Algorithms/detail/GPUScan/GPUTileScan.h>

namespace TNL::Algorithms::detail {

template< ScanType scanType, int blockSize, int valuesPerThread, typename InputView, typename OutputView, typename Reduction >
__global__
void
GPUScanKernelIndependentTiles(
   const InputView input,
   OutputView output,
   typename InputView::IndexType begin,
   typename InputView::IndexType end,
   typename OutputView::IndexType outputBegin,
   Reduction reduction,
   typename OutputView::ValueType identity,
   typename OutputView::ValueType* blockResults )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   using ValueType = typename OutputView::ValueType;
   using TileScan = GPUTileScan< scanType, blockSize, valuesPerThread, Reduction, ValueType >;

   // allocate shared memory
   __shared__ Backend::Uninitialized< typename TileScan::Storage > storage;

   // scan from input into output
   const ValueType value =
      TileScan::scan( input, output, begin, end, outputBegin, reduction, identity, identity, storage.get() );

   // The last thread of the block stores the block result in the global memory.
   if( blockResults && threadIdx.x == blockDim.x - 1 )
      blockResults[ blockIdx.x ] = value;
#endif
}

}  // namespace TNL::Algorithms::detail
