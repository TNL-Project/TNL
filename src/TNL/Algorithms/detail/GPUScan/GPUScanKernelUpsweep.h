// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

/* GPUScanKernelUpsweep - per-block reduction of the input (the first phase of
 * the "reduce-then-scan" strategy).
 *
 * Each CUDA block reduces its tile of blockSize * valuesPerThread elements of
 * the input and stores the result in reductionResults[blockIdx.x]. Nothing is
 * written to the output. The block results are then scanned and
 * GPUScanKernelDownsweep reads the input again and writes the final result.
 *
 * Usage (see GPUScanKernelLauncher in GPUScan.h): the first phase of the
 * ScanPhaseType::WriteInSecondPhase variant when the input is larger than one
 * tile and either it does not fit into one CUDA grid or the scan is performed
 * in two phases explicitly (the distributed scan).
 *
 * Global memory traffic of the two-phase variant (Upsweep + Downsweep) is 2n
 * reads and n writes. The input is read twice, so the variant is suitable for
 * arrays, but not for expensive expressions (see ScanPhaseType).
 *
 * References:
 * D. Merrill, A. Grimshaw, "Parallel Scan for Stream Architectures",
 * Technical Report CS2009-14, University of Virginia, 2009.
 * https://libraopen.lib.virginia.edu/public_view/kd17cs85f
 * D. Merrill, M. Garland, "Single-pass Parallel Prefix Sum with Decoupled
 * Lookback", NVIDIA Technical Report NVR-2016-002, 2016.
 * https://research.nvidia.com/publication/2016-03_Single-pass-Parallel-Prefix-Sum-Decoupled-Lookback
 * (the latter compares the "reduce-then-scan" strategy with the others)
 */

#include <TNL/Assert.h>
#include <TNL/Math.h>
#include <TNL/Algorithms/detail/CudaReductionKernel.h>

namespace TNL::Algorithms::detail {

template< int blockSize, int valuesPerThread, typename InputView, typename Reduction, typename ValueType >
__global__
void
GPUScanKernelUpsweep(
   const InputView input,
   typename InputView::IndexType begin,
   typename InputView::IndexType end,
   Reduction reduction,
   ValueType identity,
   ValueType* reductionResults )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   // verify the configuration
   TNL_ASSERT_EQ( blockDim.x, blockSize, "unexpected block size in GPUScanKernelUpsweep" );
   static_assert(
      valuesPerThread % 2,
      "valuesPerThread must be odd, otherwise there would be shared memory bank conflicts "
      "when threads access their chunks in shared memory sequentially" );

   // allocate shared memory
   using BlockReduce = CudaBlockReduce< blockSize, Reduction, ValueType >;
   union Shared
   {
      ValueType data[ blockSize * valuesPerThread ];
      typename BlockReduce::Storage blockReduceStorage;

      // initialization is not allowed for __shared__ variables, so we need to
      // disable initialization in the implicit default constructor
      __device__
      Shared() {}
   };
   __shared__ Shared storage;

   // calculate indices (the offset is computed in the index type, since one
   // grid may cover more elements than int can represent)
   using InputIndex = typename InputView::IndexType;
   constexpr int maxElementsInBlock = blockSize * valuesPerThread;
   const InputIndex blockOffset = static_cast< InputIndex >( blockIdx.x ) * maxElementsInBlock;
   const int elementsInBlock =
      static_cast< int >( TNL::min( end - begin - blockOffset, static_cast< InputIndex >( maxElementsInBlock ) ) );

   // update global array offset for the thread
   begin += blockOffset + threadIdx.x;

   // Load data into the shared memory.
   {
      int idx = threadIdx.x;
      while( idx < elementsInBlock ) {
         storage.data[ idx ] = input[ begin ];
         begin += blockDim.x;
         idx += blockDim.x;
      }
      // fill the remaining (maxElementsInBlock - elementsInBlock) values with identity
      // (this helps to avoid divergent branches in the blocks below)
      while( idx < maxElementsInBlock ) {
         storage.data[ idx ] = identity;
         idx += blockDim.x;
      }
   }
   __syncthreads();

   // Perform sequential reduction of the thread's chunk in shared memory.
   const int chunkOffset = threadIdx.x * valuesPerThread;
   ValueType value = storage.data[ chunkOffset ];
   #pragma unroll
   for( int i = 1; i < valuesPerThread; i++ )
      value = reduction( value, storage.data[ chunkOffset + i ] );
   __syncthreads();

   // Perform the parallel reduction.
   value = BlockReduce::reduce( reduction, identity, value, storage.blockReduceStorage, threadIdx.x );

   // Store the block result in the global memory.
   if( threadIdx.x == 0 )
      reductionResults[ blockIdx.x ] = value;
#endif
}

}  // namespace TNL::Algorithms::detail
