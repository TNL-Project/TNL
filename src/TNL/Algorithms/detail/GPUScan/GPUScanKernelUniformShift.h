// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

/* GPUScanKernelUniformShift - adds the block prefixes to an already scanned
 * output (the second phase of the "scan-then-propagate" strategy).
 *
 * Each CUDA block combines all elements of its tile of the output with the
 * prefix of the tile blockResults[blockIdx.x] and the global shift. The block
 * results are scanned exclusively between the phases, so blockResults[i] is
 * the reduction of all tiles preceding the tile i. The input is not read.
 *
 * Usage (see GPUScanKernelLauncher in GPUScan.h):
 * - the second phase of the ScanPhaseType::WriteInFirstPhase variant, after
 *   GPUScanKernelIndependentTiles and the scan of the block results,
 * - the second phase with a single block when the input fits into one tile
 *   and the scan is performed in two phases explicitly (e.g. the distributed
 *   scan applies the shift of the preceding ranks this way).
 *
 * \param blockResults  An array of per-block shifts, i.e. the exclusive scan of
 *                      the block results of the first phase (computed by
 *                      GPUScanKernelIndependentTiles)
 * \param shift         A global shift to be applied to all elements of the
 *                      output array.
 *
 * References: see GPUScanKernelIndependentTiles.h.
 */

#include <TNL/Assert.h>
#include <TNL/Backend.h>

namespace TNL::Algorithms::detail {

template< int blockSize, int valuesPerThread, typename OutputView, typename Reduction >
__global__
void
GPUScanKernelUniformShift(
   OutputView output,
   typename OutputView::IndexType outputBegin,
   typename OutputView::IndexType outputEnd,
   Reduction reduction,
   const typename OutputView::ValueType* blockResults,
   typename OutputView::ValueType shift )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   // load the block result into a __shared__ variable first
   __shared__ Backend::Uninitialized< typename OutputView::ValueType > storage;
   if( threadIdx.x == 0 )
      storage.get() = blockResults[ blockIdx.x ];

   // update the output offset for the thread (the offset is computed in the
   // index type, since one grid may cover more elements than int can represent)
   TNL_ASSERT_EQ( blockDim.x, blockSize, "unexpected block size in GPUScanKernelUniformShift" );
   using OutputIndex = typename OutputView::IndexType;
   constexpr int maxElementsInBlock = blockSize * valuesPerThread;
   outputBegin += static_cast< OutputIndex >( blockIdx.x ) * maxElementsInBlock + threadIdx.x;

   // update the block shift
   __syncthreads();
   shift = reduction( shift, storage.get() );

   int valueIdx = 0;
   while( valueIdx < valuesPerThread && outputBegin < outputEnd ) {
      output[ outputBegin ] = reduction( output[ outputBegin ], shift );
      outputBegin += blockDim.x;
      valueIdx++;
   }
#endif
}

}  // namespace TNL::Algorithms::detail
