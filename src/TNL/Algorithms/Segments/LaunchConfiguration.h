// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <TNL/Backend/KernelLaunch.h>
#include <TNL/Backend/LaunchHelpers.h>

namespace TNL::Algorithms::Segments {

/**
 * \brief Enumeration for mapping threads to segments.
 *
 * This enumeration defines how threads are mapped to segments
 * during parallel operations.
 * Not every mapping is supported by every segments format and operation.
 */
enum class ThreadsToSegmentsMapping : std::uint8_t
{
   //! \brief The best native strategy of the given segments format,
   //! resolved by the format itself.
   Default,
   //! \brief The same number of threads (the threads per segment count)
   //! is assigned to every segment.
   Uniform,
   /**
    * \brief The threads are assigned to the elements of the segments
    * processed by a block, regardless of the segment boundaries.
    *
    * The threads per segment count is the average number of threads
    * per segment,
    * i.e. a block with `BlockSize` threads
    * processes `BlockSize / threadsPerSegmentCount` segments.
    */
   PerElement,
   /**
    * \brief Segments are processed by a single thread, a warp or a whole block,
    * depending on their size.
    *
    * The threads per segment count is the average number of threads
    * per segment,
    * i.e. a block with `BlockSize` threads
    * processes `BlockSize / threadsPerSegmentCount` segments.
    */
   Tiered,
   //! \brief The adaptive kernel of AdaptiveCSR,
   //! which assigns the segments to the blocks in advance.
   //! CSR, which has no adaptive kernel, resolves it to the Default mapping.
   Adaptive
};

/**
 * \brief Launch configuration for segment operations.
 *
 * This class encapsulates the configuration for launching segment operations,
 * including the mapping of threads to segments
 * and the number of threads per segment.
 * The default configuration requests the `Default` mapping,
 * i.e. the best native strategy of the segments format.
 * A segments format which does not support the requested mapping
 * or threads per segment count throws `std::invalid_argument`.
 */
struct LaunchConfiguration : public Backend::LaunchConfiguration
{
   //! \brief Constructs a configuration with the `Default` mapping.
   LaunchConfiguration() = default;

   /**
    * \brief Constructs a configuration with the given mapping
    * and threads per segment count.
    *
    * \param threadsToSegmentsMapping The mapping of threads to segments.
    * \param threadsPerSegmentCount The number of threads per segment,
    *        which must be a positive power of two.
    * \param blockSize The block size of the backend launch configuration.
    */
   LaunchConfiguration( ThreadsToSegmentsMapping threadsToSegmentsMapping, int threadsPerSegmentCount, int blockSize = 1 )
   : Backend::LaunchConfiguration( dim3{}, blockSize ),
     threadsToSegmentsMapping( threadsToSegmentsMapping )
   {
      this->setThreadsPerSegmentCount( threadsPerSegmentCount );
   }

   //! \brief Returns the mapping of threads to segments.
   [[nodiscard]] ThreadsToSegmentsMapping
   getThreadsToSegmentsMapping() const
   {
      return threadsToSegmentsMapping;
   }

   /**
    * \brief Sets the number of threads per segment.
    *
    * Its meaning depends on the mapping,
    * see \ref ThreadsToSegmentsMapping.
    *
    * \param threadsPerSegmentCount The number of threads per segment.
    * \throws std::invalid_argument
    *         if `threadsPerSegmentCount` is not a positive power of two.
    *         The current count is not changed in that case.
    */
   void
   setThreadsPerSegmentCount( int threadsPerSegmentCount )
   {
      if( threadsPerSegmentCount > 0 && ( threadsPerSegmentCount & ( threadsPerSegmentCount - 1 ) ) == 0 )
         this->threadsPerSegmentCount = threadsPerSegmentCount;
      else
         throw std::invalid_argument( "Threads per segment count must be a positive power of two." );
   }

   //! \brief Sets the number of threads per segment
   //! to the warp size of the current GPU.
   void
   setThreadsPerSegmentCountToWarpSize()
   {
      this->threadsPerSegmentCount = Backend::getWarpSize( Backend::getDevice() );
   }

   //! \brief Returns the number of threads per segment.
   [[nodiscard]] int
   getThreadsPerSegmentCount() const
   {
      return threadsPerSegmentCount;
   }

   //! \brief Sets the mapping of threads to segments.
   void
   setThreadsToSegmentsMapping( ThreadsToSegmentsMapping threadsToSegmentsMapping )
   {
      this->threadsToSegmentsMapping = threadsToSegmentsMapping;
   }

   //! \brief Returns the backend part of the launch configuration
   //! with the grid and block sizes.
   [[nodiscard]] const Backend::LaunchConfiguration&
   getBackendLaunchConfiguration() const
   {
      return *this;
   }

protected:
   ThreadsToSegmentsMapping threadsToSegmentsMapping = ThreadsToSegmentsMapping::Default;
   int threadsPerSegmentCount = 1;
};

namespace detail {

/**
 * \brief Throws an exception if `launchConfig` requests
 * other than the `Default` thread mapping.
 *
 * This is used by the segments formats
 * which have a single native kernel for each operation
 * and thus cannot honor any other thread mapping.
 */
inline void
checkDefaultThreadsToSegmentsMapping( const LaunchConfiguration& launchConfig, const std::string& segmentsName )
{
   if( launchConfig.getThreadsToSegmentsMapping() != ThreadsToSegmentsMapping::Default )
      throw std::invalid_argument(
         "Unsupported threads to segments mapping for " + segmentsName + " segments. Only the Default mapping is supported." );
}

/**
 * \brief Calls `function` with the threads per segment count
 * as a compile-time constant.
 *
 * The runtime count `threadsPerSegment` is looked up in `Counts`
 * and `function` is called with `std::integral_constant< int, Count >`
 * for the matching `Count`,
 * so that it can instantiate the kernel for that count.
 * A count which is not in `Counts` throws `std::invalid_argument`.
 *
 * \tparam Counts The supported threads per segment counts.
 * \param threadsPerSegment The threads per segment count to dispatch.
 * \param function The function to call.
 * \param segmentsName The name of the segments format used in the error message.
 */
template< int... Counts, typename Function >
void
dispatchThreadsPerSegment( int threadsPerSegment, Function&& function, const std::string& segmentsName )
{
   const bool supported =
      ( ( threadsPerSegment == Counts && ( function( std::integral_constant< int, Counts >{} ), true ) ) || ... );
   if( ! supported )
      throw std::invalid_argument(
         "Unsupported threads per segment count ( " + std::to_string( threadsPerSegment ) + " ) for " + segmentsName
         + " segments." );
}

}  // namespace detail

}  // namespace TNL::Algorithms::Segments
