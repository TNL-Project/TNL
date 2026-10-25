// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

/* Scan (prefix sum) on GPUs.
 *
 * The input is split into tiles of blockSize * valuesPerThread elements and
 * each tile is scanned by one CUDA block (GPUTileScan, built on GPUBlockScan).
 * Scanning a tile is the easy part. The hard part is that each block also
 * needs the reduction of all preceding tiles (the prefix of its tile). The
 * strategies below differ only in how the blocks obtain their prefixes.
 *
 * 1. Single tile: GPUScanKernelIndependentTiles with one block.
 *    The whole input fits into one tile, so there is no prefix.
 *
 * 2. Decoupled lookback: GPUScanKernelLookback, one kernel launch.
 *    Each block publishes the reduction of its tile in global memory and
 *    reads the values published by the preceding blocks. The input is read
 *    once and the output is written once (n reads, n writes).
 *
 * 3. Scan-then-propagate (ScanPhaseType::WriteInFirstPhase):
 *    GPUScanKernelIndependentTiles, then GPUScanKernelUniformShift.
 *    The first phase scans each tile without its prefix, writes it to the
 *    output, and stores the reduction of the tile. The second phase adds the
 *    prefix to all elements of the output (2n reads, 2n writes).
 *
 * 4. Reduce-then-scan (ScanPhaseType::WriteInSecondPhase):
 *    GPUScanKernelUpsweep, then GPUScanKernelDownsweep.
 *    The first phase only reduces each tile. The second phase reads the input
 *    again and scans each tile with its prefix (2n reads, n writes).
 *
 * In the two-phase strategies, the tile reductions are scanned between the
 * phases (by a recursive call of the launcher), which gives the prefixes.
 *
 * GPUScanKernelLauncher::perform selects the strategy by the input size:
 *
 *    input fits into one tile        ->  1. single tile
 *    input fits into one CUDA grid   ->  2. decoupled lookback
 *    otherwise                       ->  3. or 4., selected by ScanPhaseType
 *
 * One CUDA grid has up to 2^31 - 1 blocks, so in practice the lookback is
 * used for all local scans larger than one tile, including the scans of
 * expressions. The two-phase strategies are kept for the distributed scan
 * (DistributedScan.h), which calls the phases separately via
 * performFirstPhase and performSecondPhase and adds the prefix of the
 * preceding MPI ranks in between. The lookback cannot be split this way.
 *
 * ScanPhaseType selects between the two-phase strategies:
 * - scan.h: reduce-then-scan for arrays (reading an array twice is cheaper
 *   than writing the output twice), scan-then-propagate for expressions such
 *   as a + b (evaluating them twice would be expensive). The in-place scans
 *   always use reduce-then-scan. On the GPU, this matters only for inputs
 *   larger than one grid, but the same ScanPhaseType selects the strategy of
 *   the host (OpenMP) scan in Scan.hpp, which has no lookback.
 * - distributedScan.h: the distributed scans from one array to another always
 *   use scan-then-propagate, even for arrays (see the TODO there), the
 *   in-place distributed scans use reduce-then-scan.
 *
 * Possible simplification: the first phase of the distributed scan on the
 * GPU could perform the complete local scan with the lookback and return the
 * reduction of the local range, and the second phase would only add the
 * prefix of the preceding ranks with GPUScanKernelUniformShift. The global
 * memory traffic would be the same, but the scan of the tile reductions and
 * one kernel launch would be saved, and the multi-block two-phase strategies
 * would be needed only for inputs larger than one grid.
 *
 * The lookback kernel uses larger tiles (lookbackValuesPerThread instead of
 * valuesPerThread, see the template parameters of GPUScanKernelLauncher), and
 * the single tile uses fewer values per thread for small inputs.
 *
 * See the description at the beginning of each kernel's header file for
 * details and references.
 */

#include <atomic>

#include <TNL/Backend.h>
#include <TNL/Math.h>
#include <TNL/Containers/Array.h>
#include <TNL/Algorithms/fill.h>
#include <TNL/Algorithms/detail/ScanType.h>
#include <TNL/Algorithms/detail/GPUScan/CachePool.h>
#include <TNL/Algorithms/detail/GPUScan/GPUScanKernelDownsweep.h>
#include <TNL/Algorithms/detail/GPUScan/GPUScanKernelLookback.h>
#include <TNL/Algorithms/detail/GPUScan/GPUScanKernelIndependentTiles.h>
#include <TNL/Algorithms/detail/GPUScan/GPUScanKernelUniformShift.h>
#include <TNL/Algorithms/detail/GPUScan/GPUScanKernelUpsweep.h>

namespace TNL::Algorithms::detail {

/**
 * \brief Launcher for CUDA scan kernels.
 *
 * \tparam blockSize  The CUDA block size to be used for kernel launch.
 * \tparam valuesPerThread  Number of elements processed by each thread sequentially.
 * \tparam lookbackValuesPerThread  Number of elements processed by each thread
 *         sequentially in the single-pass lookback kernel.
 */
template<
   ScanType scanType,
   ScanPhaseType phaseType,
   typename ValueType,
   // use blockSize=256 for 32-bit value types, scale with sizeof(ValueType)
   // to keep shared memory requirements constant
   int blockSize = 256 * 4 / sizeof( ValueType ),
   // valuesPerThread should be odd to avoid shared memory bank conflicts
   int valuesPerThread = 7,
   // Larger tiles pay off in the lookback kernel, because fewer blocks have to
   // wait for the prefixes of their predecessors. Measured on RTX 3060 for
   // value types of 1-16 bytes: 15 is the best (or within noise) for types up
   // to 8 bytes, 11 is the best for 16-byte types. It should be odd as well.
   int lookbackValuesPerThread = ( sizeof( ValueType ) > 8 ? 11 : 15 ) >
struct GPUScanKernelLauncher
{
   /**
    * \brief Performs both phases of prefix sum.
    *
    * \param input the input array to be scanned
    * \param output the array where the result will be stored
    * \param begin the first element in the array to be scanned
    * \param end the last element in the array to be scanned
    * \param outputBegin the first element in the output array to be written. There
    *                    must be at least `end - begin` elements in the output
    *                    array starting at the position given by `outputBegin`.
    * \param reduction Symmetric binary function representing the reduction operation
    *                  (usually addition, i.e. an instance of \ref std::plus).
    * \param identity Neutral element for given reduction operation, i.e.
    *                 value such that `reduction(identity, x) == x` for any `x`.
    * \param launch_config kernel launch parameters. Only the stream and the
    *                      `blockHostUntilFinished` flag are used, the grid and
    *                      block sizes are compile-time parameters of the kernels.
    *                      When `blockHostUntilFinished` is false, all kernels of
    *                      the scan are launched without a following
    *                      synchronization, so the scan may still be running on
    *                      the device when this function returns and the caller
    *                      is responsible for synchronizing the stream before
    *                      reading the result.
    */
   template< typename InputArray, typename OutputArray, typename Reduction >
   static void
   perform(
      const InputArray& input,
      OutputArray& output,
      typename InputArray::IndexType begin,
      typename InputArray::IndexType end,
      typename OutputArray::IndexType outputBegin,
      Reduction&& reduction,
      typename OutputArray::ValueType identity,
      const Backend::LaunchConfiguration& launch_config = {} )
   {
      if( end - begin <= blockSize * valuesPerThread ) {
         // the result of the block is not needed, so blockResults are not allocated
         performSingleBlock( input, output, begin, end, outputBegin, reduction, identity, nullptr, launch_config );
         gridsCount() = 1;
         return;
      }

      using Index = typename InputArray::IndexType;
      constexpr int maxElementsInBlock = blockSize * valuesPerThread;
      const Index numberOfBlocks = Backend::getNumberOfBlocks( end - begin, maxElementsInBlock );
      const Index numberOfGrids = Backend::getNumberOfGrids( numberOfBlocks, maxGridSize() );

      if( numberOfGrids == 1 ) {
         performLookback(
            input, output, begin, end, outputBegin, std::forward< Reduction >( reduction ), identity, launch_config );
         return;
      }

      const auto blockShifts = performFirstPhase( input, output, begin, end, outputBegin, reduction, identity, launch_config );
      if( blockShifts.getSize() > 2 )
         performSecondPhase(
            input, output, blockShifts, begin, end, outputBegin, reduction, identity, identity, launch_config );
   }

   template< typename InputArray, typename OutputArray, typename Reduction >
   static void
   performLookback(
      const InputArray& input,
      OutputArray& output,
      typename InputArray::IndexType begin,
      typename InputArray::IndexType end,
      typename OutputArray::IndexType outputBegin,
      Reduction&& reduction,
      typename OutputArray::ValueType identity,
      const Backend::LaunchConfiguration& launch_config = {} )
   {
      using Index = typename InputArray::IndexType;
      constexpr int maxElementsInBlock = blockSize * lookbackValuesPerThread;
      const Index numberOfBlocks = Backend::getNumberOfBlocks( end - begin, maxElementsInBlock );

      // Reuse a cache of the lookback states array across calls to avoid
      // cudaMalloc/cudaFree churn. The cache grows as needed; when the requested
      // size shrinks we keep the existing (larger) allocation. Each call uses a
      // new epoch, so the states written by previous calls are treated as
      // Invalid by the kernel and the array does not have to be reset. It is
      // reset (together with the tile counter) only when it is (re)allocated,
      // when the epoch would overflow, and after a failed kernel launch.
      // The cache is per device, stream and host thread (the default stream is
      // per-thread, so the same stream handle used by different host threads
      // may denote different streams). The scans sharing the cache are
      // serialized by its mutex, which is held until the kernel finishes when
      // the scan blocks the host, so the array cannot be reallocated or its
      // `epoch` changed by another host thread while the kernel uses it.
      // Asynchronous scans sharing the cache are ordered on the same stream,
      // which gives the kernel the same exclusive use of the array.
      auto& cache = LookbackCachePool< ValueType >::getInstance().getCache( launch_config.stream );
      std::scoped_lock lock( cache.mutex );
      auto& states = cache.states;
      auto& epoch = cache.epoch;
      // the size of the array is never negative, so the comparison can be done in Index
      // (which may be unsigned)
      const bool grow = static_cast< Index >( states.getSize() ) < numberOfBlocks;
      if( grow || epoch == maxLookbackEpoch ) {
         if( grow ) {
            // the array may not be reallocated while the kernels of previously
            // launched asynchronous scans are still using it
            Backend::streamSynchronize( launch_config.stream );
            states.setSize( numberOfBlocks );
         }
         Devices::GPU::LaunchConfiguration fillConfig;
         fillConfig.stream = launch_config.stream;
         fillConfig.blockHostUntilFinished = false;
         Algorithms::fill< Devices::GPU >( states.getData(), LookbackState< ValueType >{}, states.getSize(), fillConfig );
         epoch = 0;
         cache.tileCounter.setSize( 1 );
         cache.tileCounter.setValue( 0 );
         cache.tileCounterBase = 0;
      }
      epoch++;

      constexpr auto kernel = GPUScanKernelLookback<
         scanType,
         blockSize,
         lookbackValuesPerThread,
         typename InputArray::ConstViewType,
         typename OutputArray::ViewType,
         std::decay_t< Reduction >,
         ValueType >;

      Backend::LaunchConfiguration config(
         numberOfBlocks, blockSize, 0, launch_config.stream, launch_config.blockHostUntilFinished );

      try {
         Backend::launchKernel(
            kernel,
            config,
            input.getConstView(),
            output.getView(),
            begin,
            end,
            outputBegin,
            reduction,
            identity,
            states.getData(),
            epoch,
            cache.tileCounter.getData(),
            cache.tileCounterBase );
      }
      catch( ... ) {
         // It is not known how many tiles the kernel took from the counter (an
         // error reported by the launch may come from an earlier operation),
         // so the counter and tileCounterBase might not match. The next scan
         // resets the states and the counter (no GPU call is made here, since
         // the GPU may be in an error state).
         epoch = maxLookbackEpoch;
         throw;
      }
      cache.tileCounterBase += numberOfBlocks;

      gridsCount() = 1;
   }

   /* Scans the input which fits into one tile using one CUDA block. The number
    * of values per thread is chosen according to the size of the input. If
    * blockResults is not null, the result of the block is stored in
    * `blockResults[ 0 ]`. The launch configuration is used as in \ref perform.
    */
   template< typename InputArray, typename OutputArray, typename Reduction >
   static void
   performSingleBlock(
      const InputArray& input,
      OutputArray& output,
      typename InputArray::IndexType begin,
      typename InputArray::IndexType end,
      typename OutputArray::IndexType outputBegin,
      Reduction&& reduction,
      typename OutputArray::ValueType identity,
      typename OutputArray::ValueType* blockResults,
      const Backend::LaunchConfiguration& launch_config = {} )
   {
      if( end - begin <= blockSize )
         performSingleBlockImpl< 1 >(
            input, output, begin, end, outputBegin, reduction, identity, blockResults, launch_config );
      else if( end - begin <= blockSize * 3 )
         performSingleBlockImpl< 3 >(
            input, output, begin, end, outputBegin, reduction, identity, blockResults, launch_config );
      else if( end - begin <= blockSize * 5 )
         performSingleBlockImpl< 5 >(
            input, output, begin, end, outputBegin, reduction, identity, blockResults, launch_config );
      else
         performSingleBlockImpl< valuesPerThread >(
            input, output, begin, end, outputBegin, reduction, identity, blockResults, launch_config );
   }

   template< int elementsPerThread, typename InputArray, typename OutputArray, typename Reduction >
   static void
   performSingleBlockImpl(
      const InputArray& input,
      OutputArray& output,
      typename InputArray::IndexType begin,
      typename InputArray::IndexType end,
      typename OutputArray::IndexType outputBegin,
      Reduction&& reduction,
      typename OutputArray::ValueType identity,
      typename OutputArray::ValueType* blockResults,
      const Backend::LaunchConfiguration& launch_config )
   {
      constexpr auto kernel = GPUScanKernelIndependentTiles<
         scanType,
         blockSize,
         elementsPerThread,
         typename InputArray::ConstViewType,
         typename OutputArray::ViewType,
         std::decay_t< Reduction > >;
      Backend::LaunchConfiguration config( 1, blockSize, 0, launch_config.stream, launch_config.blockHostUntilFinished );
      Backend::launchKernel(
         kernel, config, input.getConstView(), output.getView(), begin, end, outputBegin, reduction, identity, blockResults );
   }

   /**
    * \brief Performs the first phase of prefix sum.
    *
    * \param input the input array to be scanned
    * \param output the array where the result will be stored
    * \param begin the first element in the array to be scanned
    * \param end the last element in the array to be scanned
    * \param outputBegin the first element in the output array to be written. There
    *                    must be at least `end - begin` elements in the output
    *                    array starting at the position given by `outputBegin`.
    * \param reduction Symmetric binary function representing the reduction operation
    *                  (usually addition, i.e. an instance of \ref std::plus).
    * \param identity Neutral element for given reduction operation, i.e.
    *                 value such that `reduction(identity, x) == x` for any `x`.
    * \param launch_config kernel launch parameters, see \ref perform. When the
    *                      input spans multiple grids and `blockHostUntilFinished`
    *                      is false, the kernels of the phase as well as the
    *                      recursive scan of the block shifts are launched
    *                      without a following synchronization.
    *
    * \return A view of the block shifts, to be passed to \ref performSecondPhase.
    *         The view is backed by a per-instantiation cache (see
    *         `BlockShiftsCachePool`), so it remains valid only until the next call
    *         of this function or of \ref performSecondPhase with the same
    *         instantiation on the same stream. The first element of the view
    *         is the identity, the (`i`+1)-th element is the reduction of the
    *         first `i` blocks, so the last element is the reduction of the
    *         whole input range.
    */
   template< typename InputArray, typename OutputArray, typename Reduction >
   static auto
   performFirstPhase(
      const InputArray& input,
      OutputArray& output,
      typename InputArray::IndexType begin,
      typename InputArray::IndexType end,
      typename OutputArray::IndexType outputBegin,
      Reduction&& reduction,
      typename OutputArray::ValueType identity,
      const Backend::LaunchConfiguration& launch_config = {} )
   {
      static_assert( std::is_same_v< ValueType, typename OutputArray::ValueType >, "invalid configuration of ValueType" );
      using Index = typename InputArray::IndexType;

      if( end <= begin ) {
         // The reduction of an empty range is the identity. The block shifts
         // are backed by the cache even in this case, so that the caller can
         // read them like in the other cases.
         auto& blockShifts = BlockShiftsCachePool< ValueType >::getInstance().getCache( launch_config.stream );
         if( static_cast< Index >( blockShifts.getSize() ) < 1 ) {
            // the array may not be reallocated while the kernels of previously
            // launched asynchronous scans are still using it
            Backend::streamSynchronize( launch_config.stream );
            blockShifts.setSize( 1 );
         }
         Devices::GPU::LaunchConfiguration fillConfig;
         fillConfig.stream = launch_config.stream;
         fillConfig.blockHostUntilFinished = launch_config.blockHostUntilFinished;
         Algorithms::fill< Devices::GPU >( blockShifts.getData(), identity, 1, fillConfig );
         return blockShifts.getView( 0, 1 );
      }

      if( end - begin <= blockSize * valuesPerThread ) {
         // The block shifts are stored in the cache also in this case (see the
         // multi-block path below), so that the returned view is always
         // cache-backed: freeing the shifts as a local array would serialize
         // the host with the pending kernels, which an asynchronous scan
         // avoids by definition. The first element is filled before the
         // kernel, ordered on the stream - a blocking kernel launch
         // synchronizes the stream, which covers the fill as well.
         auto& blockShifts = BlockShiftsCachePool< ValueType >::getInstance().getCache( launch_config.stream );
         if( static_cast< Index >( blockShifts.getSize() ) < 2 ) {
            // the array may not be reallocated while the kernels of previously
            // launched asynchronous scans are still using it
            Backend::streamSynchronize( launch_config.stream );
            blockShifts.setSize( 2 );
         }
         Devices::GPU::LaunchConfiguration fillConfig;
         fillConfig.stream = launch_config.stream;
         fillConfig.blockHostUntilFinished = false;
         Algorithms::fill< Devices::GPU >( blockShifts.getData(), identity, 1, fillConfig );

         // run the kernel with just 1 block
         // blockResults are shifted by 1, because the 0-th element should stay identity
         performSingleBlock(
            input, output, begin, end, outputBegin, reduction, identity, &blockShifts.getData()[ 1 ], launch_config );

         // Store the number of CUDA grids for the purpose of unit testing, i.e.
         // to check if we test the algorithm with more than one CUDA grid.
         gridsCount() = 1;

         // `blockShifts` now contains shift values for each block - to be used in the second phase
         return blockShifts.getView( 0, 2 );
      }
      else {
         // compute the number of grids
         constexpr int maxElementsInBlock = blockSize * valuesPerThread;
         const Index numberOfBlocks = Backend::getNumberOfBlocks( end - begin, maxElementsInBlock );
         const Index numberOfGrids = Backend::getNumberOfGrids( numberOfBlocks, maxGridSize() );

         // reuse the per-instantiation cache of the block shifts array to
         // avoid `cudaMalloc`/`cudaFree` churn: freeing the shifts as a local
         // array would serialize the host with the pending kernels, which an
         // asynchronous scan avoids by definition. The cache grows as needed;
         // when the requested size shrinks we keep the existing (larger)
         // allocation. Reusing the cache across calls is safe only for
         // kernels ordered on the same stream, which the cache keying
         // guarantees.
         auto& blockShifts = BlockShiftsCachePool< ValueType >::getInstance().getCache( launch_config.stream );
         if( static_cast< Index >( blockShifts.getSize() ) < numberOfBlocks + 1 ) {
            // the array may not be reallocated while the kernels of previously
            // launched asynchronous scans are still using it
            Backend::streamSynchronize( launch_config.stream );
            blockShifts.setSize( numberOfBlocks + 1 );
         }

         // The recursive scan of the block shifts must not enter the
         // two-phase path again - it would reuse this cache as its
         // block-results buffer and overwrite the shifts while they are
         // still being scanned.
         TNL_ASSERT_LE(
            static_cast< std::size_t >( Backend::getNumberOfBlocks( numberOfBlocks + 1, maxElementsInBlock ) ),
            maxGridSize(),
            "the recursive scan of the block shifts must not split into multiple grids" );

         // loop over all grids
         for( Index gridIdx = 0; gridIdx < numberOfGrids; gridIdx++ ) {
            // compute current grid offset and size of data to be scanned
            const Index gridOffset = gridIdx * maxGridSize() * maxElementsInBlock;
            const Index currentSize = TNL::min( end - begin - gridOffset, maxGridSize() * maxElementsInBlock );

            // set CUDA launch configuration
            Backend::LaunchConfiguration config(
               Backend::getNumberOfBlocks( currentSize, maxElementsInBlock ), blockSize, 0, launch_config.stream, false );

            // run the kernel
            switch( phaseType ) {
               case ScanPhaseType::WriteInFirstPhase:
                  {
                     constexpr auto kernel = GPUScanKernelIndependentTiles<
                        scanType,
                        blockSize,
                        valuesPerThread,
                        typename InputArray::ConstViewType,
                        typename OutputArray::ViewType,
                        Reduction >;
                     Backend::launchKernelAsync(
                        kernel,
                        config,
                        input.getConstView(),
                        output.getView(),
                        begin + gridOffset,
                        begin + gridOffset + currentSize,
                        outputBegin + gridOffset,
                        reduction,
                        identity,
                        &blockShifts.getData()[ gridIdx * maxGridSize() ] );
                     break;
                  }

               case ScanPhaseType::WriteInSecondPhase:
                  {
                     constexpr auto kernel = GPUScanKernelUpsweep<
                        blockSize,
                        valuesPerThread,
                        typename InputArray::ConstViewType,
                        Reduction,
                        typename OutputArray::ValueType >;
                     Backend::launchKernelAsync(
                        kernel,
                        config,
                        input.getConstView(),
                        begin + gridOffset,
                        begin + gridOffset + currentSize,
                        reduction,
                        identity,
                        &blockShifts.getData()[ gridIdx * maxGridSize() ] );
                     break;
                  }
            }
         }

         // synchronize the stream after all grids when the scan blocks the
         // host; for an asynchronous scan, the kernels are ordered on the
         // stream and the scan of the block shifts below is asynchronous as
         // well
         if( launch_config.blockHostUntilFinished )
            Backend::streamSynchronize( launch_config.stream );

         auto shiftsView = blockShifts.getView( 0, numberOfBlocks + 1 );

         // `blockShifts` now contains scan results for each block. The first phase
         // ends by computing an exclusive scan of this array.
         GPUScanKernelLauncher< ScanType::Exclusive, ScanPhaseType::WriteInSecondPhase, ValueType >::perform(
            shiftsView, shiftsView, 0, numberOfBlocks + 1, 0, reduction, identity, launch_config );

         // Store the number of CUDA grids for the purpose of unit testing, i.e.
         // to check if we test the algorithm with more than one CUDA grid.
         gridsCount() = numberOfGrids;

         // `blockShifts` now contains shift values for each block - to be used in the second phase
         return shiftsView;
      }
   }

   /**
    * \brief Performs the second phase of prefix sum.
    *
    * \param input the input array to be scanned
    * \param output the array where the result will be stored
    * \param blockShifts  Pointer to a GPU array containing the block shifts. It is the
    *                     result of the first phase.
    * \param begin the first element in the array to be scanned
    * \param end the last element in the array to be scanned
    * \param outputBegin the first element in the output array to be written. There
    *                    must be at least `end - begin` elements in the output
    *                    array starting at the position given by `outputBegin`.
    * \param reduction Symmetric binary function representing the reduction operation
    *                  (usually addition, i.e. an instance of \ref std::plus).
    * \param identity Neutral element for given reduction operation, i.e.
    *                 value such that `reduction(identity, x) == x` for any `x`.
    * \param shift A constant shifting all elements of the array (usually
    *              `identity`, i.e. the neutral value).
    * \param launch_config kernel launch parameters, see \ref perform.
    */
   template< typename InputArray, typename OutputArray, typename BlockShifts, typename Reduction >
   static void
   performSecondPhase(
      const InputArray& input,
      OutputArray& output,
      const BlockShifts& blockShifts,
      typename InputArray::IndexType begin,
      typename InputArray::IndexType end,
      typename OutputArray::IndexType outputBegin,
      Reduction&& reduction,
      typename OutputArray::ValueType identity,
      typename OutputArray::ValueType shift,
      const Backend::LaunchConfiguration& launch_config = {} )
   {
      static_assert( std::is_same_v< ValueType, typename OutputArray::ValueType >, "invalid configuration of ValueType" );
      using Index = typename InputArray::IndexType;

      // if the input was already scanned with just one block in the first phase,
      // it must be shifted uniformly in the second phase
      if( end - begin <= blockSize * valuesPerThread ) {
         constexpr auto kernel =
            GPUScanKernelUniformShift< blockSize, valuesPerThread, typename OutputArray::ViewType, Reduction >;
         Backend::LaunchConfiguration config( 1, blockSize, 0, launch_config.stream, launch_config.blockHostUntilFinished );
         Backend::launchKernel(
            kernel, config, output.getView(), outputBegin, outputBegin + end - begin, reduction, blockShifts.getData(), shift );
      }
      else {
         // compute the number of grids
         constexpr int maxElementsInBlock = blockSize * valuesPerThread;
         const Index numberOfBlocks = Backend::getNumberOfBlocks( end - begin, maxElementsInBlock );
         const Index numberOfGrids = Backend::getNumberOfGrids( numberOfBlocks, maxGridSize() );

         // loop over all grids
         for( Index gridIdx = 0; gridIdx < numberOfGrids; gridIdx++ ) {
            // compute current grid offset and size of data to be scanned
            const Index gridOffset = gridIdx * maxGridSize() * maxElementsInBlock;
            const Index currentSize = TNL::min( end - begin - gridOffset, maxGridSize() * maxElementsInBlock );

            // set CUDA launch configuration
            Backend::LaunchConfiguration config(
               Backend::getNumberOfBlocks( currentSize, maxElementsInBlock ), blockSize, 0, launch_config.stream, false );

            // run the kernel
            switch( phaseType ) {
               case ScanPhaseType::WriteInFirstPhase:
                  {
                     constexpr auto kernel =
                        GPUScanKernelUniformShift< blockSize, valuesPerThread, typename OutputArray::ViewType, Reduction >;
                     Backend::launchKernelAsync(
                        kernel,
                        config,
                        output.getView(),
                        outputBegin + gridOffset,
                        outputBegin + gridOffset + currentSize,
                        reduction,
                        &blockShifts.getData()[ gridIdx * maxGridSize() ],
                        shift );
                     break;
                  }

               case ScanPhaseType::WriteInSecondPhase:
                  {
                     constexpr auto kernel = GPUScanKernelDownsweep<
                        scanType,
                        blockSize,
                        valuesPerThread,
                        typename InputArray::ConstViewType,
                        typename OutputArray::ViewType,
                        Reduction >;
                     Backend::launchKernelAsync(
                        kernel,
                        config,
                        input.getConstView(),
                        output.getView(),
                        begin + gridOffset,
                        begin + gridOffset + currentSize,
                        outputBegin + gridOffset,
                        reduction,
                        identity,
                        shift,
                        &blockShifts.getData()[ gridIdx * maxGridSize() ] );
                     break;
                  }
            }
         }

         // synchronize the stream after all grids
         if( launch_config.blockHostUntilFinished )
            Backend::streamSynchronize( launch_config.stream );
      }
   }

   // The following serves for setting smaller maxGridSize so that we can force
   // the scan in CUDA to run with more than one grid in unit tests.
   static std::size_t&
   maxGridSize()
   {
      static std::size_t maxGridSize = Backend::getMaxGridXSize();
      return maxGridSize;
   }

   static void
   resetMaxGridSize()
   {
      maxGridSize() = Backend::getMaxGridXSize();
      gridsCount() = -1;
   }

   // number of grids used by the last scan (for the unit tests); it is atomic,
   // since the scans may be performed concurrently by several host threads
   static std::atomic< int >&
   gridsCount()
   {
      static std::atomic< int > gridsCount = -1;
      return gridsCount;
   }
};

}  // namespace TNL::Algorithms::detail
