// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

/* Scan (prefix sum) on GPUs.
 *
 * GPUScanKernelLauncher selects one of the scan kernels from the GPUScan
 * subdirectory, depending on the input size n, the tile size
 * blockSize * valuesPerThread, and ScanPhaseType:
 *
 * - n fits into one tile: GPUScanKernelParallel with a single CUDA block,
 * - n fits into one CUDA grid: GPUScanKernelLookback (single pass),
 * - otherwise a two-phase scan where the block results of the first phase are
 *   scanned recursively:
 *   - ScanPhaseType::WriteInFirstPhase: GPUScanKernelParallel followed by
 *     GPUScanKernelUniformShift ("scan-then-propagate"),
 *   - ScanPhaseType::WriteInSecondPhase: GPUScanKernelUpsweep followed by
 *     GPUScanKernelDownsweep ("reduce-then-scan").
 *
 * The two phases can also be called separately via performFirstPhase and
 * performSecondPhase, which is used by the distributed scan; the lookback
 * kernel is never used in this case.
 *
 * See the description at the beginning of each kernel's header file for
 * details and references.
 */

#include <TNL/Backend.h>
#include <TNL/Math.h>
#include <TNL/Containers/Array.h>
#include <TNL/Algorithms/fill.h>
#include <TNL/Algorithms/detail/ScanType.h>
#include <TNL/Algorithms/detail/GPUScan/GPUScanKernelDownsweep.h>
#include <TNL/Algorithms/detail/GPUScan/GPUScanKernelLookback.h>
#include <TNL/Algorithms/detail/GPUScan/GPUScanKernelParallel.h>
#include <TNL/Algorithms/detail/GPUScan/GPUScanKernelUniformShift.h>
#include <TNL/Algorithms/detail/GPUScan/GPUScanKernelUpsweep.h>

namespace TNL::Algorithms::detail {

/**
 * \brief Launcher for CUDA scan kernels.
 *
 * \tparam blockSize  The CUDA block size to be used for kernel launch.
 * \tparam valuesPerThread  Number of elements processed by each thread sequentially.
 */
template<
   ScanType scanType,
   ScanPhaseType phaseType,
   typename ValueType,
   // use blockSize=256 for 32-bit value types, scale with sizeof(ValueType)
   // to keep shared memory requirements constant
   int blockSize = 256 * 4 / sizeof( ValueType ),
   // valuesPerThread should be odd to avoid shared memory bank conflicts
   int valuesPerThread = 7 >
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
      typename OutputArray::ValueType identity )
   {
      if( end - begin <= blockSize * valuesPerThread ) {
         const auto blockShifts = performFirstPhase( input, output, begin, end, outputBegin, reduction, identity );
         return;
      }

      using Index = typename InputArray::IndexType;
      constexpr int maxElementsInBlock = blockSize * valuesPerThread;
      const Index numberOfBlocks = Backend::getNumberOfBlocks( end - begin, maxElementsInBlock );
      const Index numberOfGrids = Backend::getNumberOfGrids( numberOfBlocks, maxGridSize() );

      if( numberOfGrids == 1 ) {
         performLookback( input, output, begin, end, outputBegin, std::forward< Reduction >( reduction ), identity );
         return;
      }

      const auto blockShifts = performFirstPhase( input, output, begin, end, outputBegin, reduction, identity );
      if( blockShifts.getSize() > 2 )
         performSecondPhase( input, output, blockShifts, begin, end, outputBegin, reduction, identity, identity );
   }

   template< typename InputArray, typename OutputArray, typename Reduction >
   static void
   performLookback(
      const InputArray& input,
      OutputArray& output,
      typename InputArray::IndexType begin,
      typename InputArray::IndexType end,
      typename InputArray::IndexType outputBegin,
      Reduction&& reduction,
      typename OutputArray::ValueType identity )
   {
      performLookbackImpl< valuesPerThread >(
         input, output, begin, end, outputBegin, std::forward< Reduction >( reduction ), identity );
   }

   template< int elementsPerThread, typename InputArray, typename OutputArray, typename Reduction >
   static void
   performLookbackImpl(
      const InputArray& input,
      OutputArray& output,
      typename InputArray::IndexType begin,
      typename InputArray::IndexType end,
      typename InputArray::IndexType outputBegin,
      Reduction&& reduction,
      typename OutputArray::ValueType identity )
   {
      using Index = typename InputArray::IndexType;
      constexpr int maxElementsInBlock = blockSize * elementsPerThread;
      const Index numberOfBlocks = Backend::getNumberOfBlocks( end - begin, maxElementsInBlock );

      // Reuse a thread-local cache of the lookback states array across calls
      // to avoid cudaMalloc/cudaFree churn. The cache grows as needed; when
      // the requested size shrinks we keep the existing (larger) allocation
      // and only reset the actually used prefix. Resetting is still required
      // because the kernel reads predecessor statuses during the lookback
      // walk and they must start as Invalid.
      auto& states = lookbackStatesCache();
      // the size of the array is never negative, so the comparison can be done in Index
      // (which may be unsigned)
      if( static_cast< Index >( states.getSize() ) < numberOfBlocks ) {
         states.setSize( numberOfBlocks );
      }

      Algorithms::fillAsync< Devices::GPU >( states.getData(), LookbackState< ValueType >{}, numberOfBlocks );
      constexpr auto kernel = GPUScanKernelLookback<
         scanType,
         blockSize,
         elementsPerThread,
         typename InputArray::ConstViewType,
         typename OutputArray::ViewType,
         std::decay_t< Reduction >,
         ValueType >;

      Backend::LaunchConfiguration launch_config;
      launch_config.blockSize.x = blockSize;
      launch_config.gridSize.x = numberOfBlocks;

      Backend::launchKernelSync(
         kernel,
         launch_config,
         input.getConstView(),
         output.getView(),
         begin,
         end,
         outputBegin,
         reduction,
         identity,
         states.getData() );

      gridsCount() = 1;
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
      typename OutputArray::ValueType identity )
   {
      static_assert( std::is_same_v< ValueType, typename OutputArray::ValueType >, "invalid configuration of ValueType" );
      using Index = typename InputArray::IndexType;

      if( end - begin <= blockSize * valuesPerThread ) {
         // allocate array for the block results
         Containers::Array< typename OutputArray::ValueType, Devices::Cuda > blockResults;
         blockResults.setSize( 2 );
         blockResults.setElement( 0, identity );

         // run the kernel with just 1 block
         if( end - begin <= blockSize ) {
            constexpr auto kernel = GPUScanKernelParallel<
               scanType,
               blockSize,
               1,
               typename InputArray::ConstViewType,
               typename OutputArray::ViewType,
               Reduction >;
            Backend::launchKernelSync(
               kernel,
               Backend::LaunchConfiguration( 1, blockSize ),
               input.getConstView(),
               output.getView(),
               begin,
               end,
               outputBegin,
               reduction,
               identity,
               // blockResults are shifted by 1, because the 0-th element should stay identity
               &blockResults.getData()[ 1 ] );
         }
         else if( end - begin <= blockSize * 3 ) {
            constexpr auto kernel = GPUScanKernelParallel<
               scanType,
               blockSize,
               3,
               typename InputArray::ConstViewType,
               typename OutputArray::ViewType,
               Reduction >;
            Backend::launchKernelSync(
               kernel,
               Backend::LaunchConfiguration( 1, blockSize ),
               input.getConstView(),
               output.getView(),
               begin,
               end,
               outputBegin,
               reduction,
               identity,
               // blockResults are shifted by 1, because the 0-th element should stay identity
               &blockResults.getData()[ 1 ] );
         }
         else if( end - begin <= blockSize * 5 ) {
            constexpr auto kernel = GPUScanKernelParallel<
               scanType,
               blockSize,
               5,
               typename InputArray::ConstViewType,
               typename OutputArray::ViewType,
               Reduction >;
            Backend::launchKernelSync(
               kernel,
               Backend::LaunchConfiguration( 1, blockSize ),
               input.getConstView(),
               output.getView(),
               begin,
               end,
               outputBegin,
               reduction,
               identity,
               // blockResults are shifted by 1, because the 0-th element should stay identity
               &blockResults.getData()[ 1 ] );
         }
         else {
            constexpr auto kernel = GPUScanKernelParallel<
               scanType,
               blockSize,
               valuesPerThread,
               typename InputArray::ConstViewType,
               typename OutputArray::ViewType,
               Reduction >;
            Backend::launchKernelSync(
               kernel,
               Backend::LaunchConfiguration( 1, blockSize ),
               input.getConstView(),
               output.getView(),
               begin,
               end,
               outputBegin,
               reduction,
               identity,
               // blockResults are shifted by 1, because the 0-th element should stay identity
               &blockResults.getData()[ 1 ] );
         }

         // Store the number of CUDA grids for the purpose of unit testing, i.e.
         // to check if we test the algorithm with more than one CUDA grid.
         gridsCount() = 1;

         // blockResults now contains shift values for each block - to be used in the second phase
         return blockResults;
      }
      else {
         // compute the number of grids
         constexpr int maxElementsInBlock = blockSize * valuesPerThread;
         const Index numberOfBlocks = Backend::getNumberOfBlocks( end - begin, maxElementsInBlock );
         const Index numberOfGrids = Backend::getNumberOfGrids( numberOfBlocks, maxGridSize() );

         // allocate array for the block results
         Containers::Array< typename OutputArray::ValueType, Devices::Cuda > blockResults;
         blockResults.setSize( numberOfBlocks + 1 );

         // loop over all grids
         for( Index gridIdx = 0; gridIdx < numberOfGrids; gridIdx++ ) {
            // compute current grid offset and size of data to be scanned
            const Index gridOffset = gridIdx * maxGridSize() * maxElementsInBlock;
            const Index currentSize = TNL::min( end - begin - gridOffset, maxGridSize() * maxElementsInBlock );

            // set CUDA launch configuration
            Backend::LaunchConfiguration launch_config;
            launch_config.blockSize.x = blockSize;
            launch_config.gridSize.x = Backend::getNumberOfBlocks( currentSize, maxElementsInBlock );

            // run the kernel
            switch( phaseType ) {
               case ScanPhaseType::WriteInFirstPhase:
                  {
                     constexpr auto kernel = GPUScanKernelParallel<
                        scanType,
                        blockSize,
                        valuesPerThread,
                        typename InputArray::ConstViewType,
                        typename OutputArray::ViewType,
                        Reduction >;
                     Backend::launchKernelAsync(
                        kernel,
                        launch_config,
                        input.getConstView(),
                        output.getView(),
                        begin + gridOffset,
                        begin + gridOffset + currentSize,
                        outputBegin + gridOffset,
                        reduction,
                        identity,
                        &blockResults.getData()[ gridIdx * maxGridSize() ] );
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
                        launch_config,
                        input.getConstView(),
                        begin + gridOffset,
                        begin + gridOffset + currentSize,
                        reduction,
                        identity,
                        &blockResults.getData()[ gridIdx * maxGridSize() ] );
                     break;
                  }
            }
         }

         // No sync needed: null-stream kernel launches are implicitly ordered.
         GPUScanKernelLauncher< ScanType::Exclusive, ScanPhaseType::WriteInSecondPhase, ValueType >::perform(
            blockResults, blockResults, 0, blockResults.getSize(), 0, reduction, identity );

         // Store the number of CUDA grids for the purpose of unit testing, i.e.
         // to check if we test the algorithm with more than one CUDA grid.
         gridsCount() = numberOfGrids;

         // blockResults now contains shift values for each block - to be used in the second phase
         return blockResults;
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
      typename OutputArray::ValueType shift )
   {
      static_assert( std::is_same_v< ValueType, typename OutputArray::ValueType >, "invalid configuration of ValueType" );
      using Index = typename InputArray::IndexType;

      // if the input was already scanned with just one block in the first phase,
      // it must be shifted uniformly in the second phase
      if( end - begin <= blockSize * valuesPerThread ) {
         constexpr auto kernel =
            GPUScanKernelUniformShift< blockSize, valuesPerThread, typename OutputArray::ViewType, Reduction >;
         Backend::launchKernelSync(
            kernel,
            Backend::LaunchConfiguration( 1, blockSize ),
            output.getView(),
            outputBegin,
            outputBegin + end - begin,
            reduction,
            blockShifts.getData(),
            shift );
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
            Backend::LaunchConfiguration launch_config;
            launch_config.blockSize.x = blockSize;
            launch_config.gridSize.x = Backend::getNumberOfBlocks( currentSize, maxElementsInBlock );

            // run the kernel
            switch( phaseType ) {
               case ScanPhaseType::WriteInFirstPhase:
                  {
                     constexpr auto kernel =
                        GPUScanKernelUniformShift< blockSize, valuesPerThread, typename OutputArray::ViewType, Reduction >;
                     Backend::launchKernelAsync(
                        kernel,
                        launch_config,
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
                        launch_config,
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

         // synchronize the null-stream after all grids
         Backend::streamSynchronize( 0 );  // NOLINT(modernize-use-nullptr)
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

   // Per-instantiation cache of the lookback states array, reused across
   // calls to performLookbackImpl to avoid cudaMalloc/cudaFree churn.
   static Containers::Array< LookbackState< ValueType >, Devices::Cuda >&
   lookbackStatesCache()
   {
      static Containers::Array< LookbackState< ValueType >, Devices::Cuda > states;
      return states;
   }

   static int&
   gridsCount()
   {
      static int gridsCount = -1;
      return gridsCount;
   }
};

}  // namespace TNL::Algorithms::detail
