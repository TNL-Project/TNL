// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

/* GPUScanKernelDownsweep - scan of the input tiles with known block prefixes
 * (the second phase of the "reduce-then-scan" strategy).
 *
 * Each CUDA block reads its tile of the input, scans it with GPUTileScan using
 * the scanned block result reductionResults[blockIdx.x] (combined with the
 * global shift) as the initial value, and writes the final result to the
 * output.
 *
 * Usage (see GPUScanKernelLauncher in GPUScan.h): the second phase of the
 * ScanPhaseType::WriteInSecondPhase variant, after GPUScanKernelUpsweep and
 * the scan of the block results.
 *
 * References: see GPUScanKernelUpsweep.h.
 */

#include <TNL/Backend.h>
#include <TNL/Algorithms/detail/ScanType.h>
#include <TNL/Algorithms/detail/GPUScan/GPUTileScan.h>

namespace TNL::Algorithms::detail {

template< ScanType scanType, int blockSize, int valuesPerThread, typename InputView, typename OutputView, typename Reduction >
__global__
void
GPUScanKernelDownsweep(
   const InputView input,
   OutputView output,
   typename InputView::IndexType begin,
   typename InputView::IndexType end,
   typename OutputView::IndexType outputBegin,
   Reduction reduction,
   typename OutputView::ValueType identity,
   typename OutputView::ValueType shift,
   const typename OutputView::ValueType* reductionResults )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   using ValueType = typename OutputView::ValueType;
   using TileScan = GPUTileScan< scanType, blockSize, valuesPerThread, Reduction, ValueType >;

   // allocate shared memory
   __shared__ Backend::Uninitialized< typename TileScan::Storage > storage;

   // load the reduction of the previous tiles
   shift = reduction( shift, reductionResults[ blockIdx.x ] );

   // scan from input into output
   TileScan::scan( input, output, begin, end, outputBegin, reduction, identity, shift, storage.get() );
#endif
}

}  // namespace TNL::Algorithms::detail
