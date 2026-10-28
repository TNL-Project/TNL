// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once
#include <cstdint>
#include <stdexcept>
#include <string>

#include <TNL/Backend/KernelLaunch.h>
#include <TNL/Backend/LaunchHelpers.h>

namespace TNL::Algorithms::Segments {

/**
 * \brief Enumeration for mapping threads to segments.
 *
 * This enumeration defines how threads are mapped to segments during parallel operations.
 * Not every mapping is supported by every segments format and operation.
 */
enum class ThreadsToSegmentsMapping : std::uint8_t
{
   //! \brief The best native strategy of the given segments format, resolved by the format itself.
   Default,
   //! \brief The same number of threads (the threads per segment count) is assigned to every segment.
   Uniform,
   /**
    * \brief The threads are assigned to the elements of the segments processed by a block,
    * regardless of the segment boundaries.
    *
    * The threads per segment count is the average number of threads per segment,
    * i.e. a block with `BlockSize` threads
    * processes `BlockSize / threadsPerSegmentCount` segments.
    */
   PerElement,
   /**
    * \brief Segments are processed by a single thread, a warp or a whole block,
    * depending on their size.
    *
    * The threads per segment count is the average number of threads per segment,
    * i.e. a block with `BlockSize` threads
    * processes `BlockSize / threadsPerSegmentCount` segments.
    */
   Tiered,
   //! \brief The adaptive kernel of AdaptiveCSR, which assigns the segments to the blocks in advance.
   //! CSR, which has no adaptive kernel, resolves it to the Default mapping.
   Adaptive
};

/**
 * \brief Launch configuration for segment operations.
 *
 * This class encapsulates the configuration for launching segment operations,
 * including the mapping of threads to segments and the number of threads per segment.
 */
struct LaunchConfiguration : public Backend::LaunchConfiguration
{
   LaunchConfiguration() = default;

   LaunchConfiguration( ThreadsToSegmentsMapping threadsToSegmentsMapping, int threadsPerSegmentCount, int blockSize = 1 )
   : Backend::LaunchConfiguration( dim3{}, blockSize ),
     threadsToSegmentsMapping( threadsToSegmentsMapping )
   {
      this->setThreadsPerSegmentCount( threadsPerSegmentCount );
   }

   [[nodiscard]] ThreadsToSegmentsMapping
   getThreadsToSegmentsMapping() const
   {
      return threadsToSegmentsMapping;
   }

   void
   setThreadsPerSegmentCount( int threadsPerSegmentCount )
   {
      if( threadsPerSegmentCount > 0 && ( threadsPerSegmentCount & ( threadsPerSegmentCount - 1 ) ) == 0 )
         this->threadsPerSegmentCount = threadsPerSegmentCount;
      else
         throw std::invalid_argument( "Threads per segment count must be a positive power of two." );
   }

   void
   setThreadsPerSegmentCountToWarpSize()
   {
      this->threadsPerSegmentCount = Backend::getWarpSize( Backend::getDevice() );
   }

   [[nodiscard]] int
   getThreadsPerSegmentCount() const
   {
      return threadsPerSegmentCount;
   }

   void
   setThreadsToSegmentsMapping( ThreadsToSegmentsMapping threadsToSegmentsMapping )
   {
      this->threadsToSegmentsMapping = threadsToSegmentsMapping;
   }

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
 * \brief Throws an exception if \e launchConfig requests other than the Default thread mapping.
 *
 * This is used by the segments formats which have a single native kernel for each operation and
 * thus cannot honor any other thread mapping.
 */
inline void
checkDefaultThreadsToSegmentsMapping( const LaunchConfiguration& launchConfig, const std::string& segmentsName )
{
   if( launchConfig.getThreadsToSegmentsMapping() != ThreadsToSegmentsMapping::Default )
      throw std::invalid_argument(
         "Unsupported threads to segments mapping for " + segmentsName + " segments. Only the Default mapping is supported." );
}

}  // namespace detail

}  // namespace TNL::Algorithms::Segments
