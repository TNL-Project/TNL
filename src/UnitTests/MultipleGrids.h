#pragma once

#include <string>
#include <type_traits>

#include <gtest/gtest.h>

#include <TNL/Backend/LaunchHelpers.h>
#include <TNL/Devices/GPU.h>

// Runs the test as usual and, on GPUs, once more with at most maxGridSize
// blocks per grid, so that the kernels launched as several grids are tested
// even with small data. The default maxGridSize = 1 launches every block as a
// separate grid; larger values reduce the number of launches in tests with
// large data, but the kernels must still need more than maxGridSize blocks.
template< typename Device, typename Test >
void
runWithMultipleGrids( Test&& test, std::size_t maxGridSize = 1 )
{
   test();
   if constexpr( std::is_same_v< Device, TNL::Devices::GPU > ) {
      SCOPED_TRACE( "at most " + std::to_string( maxGridSize ) + " blocks per grid" );
      TNL::Backend::detail::MaxGridXSizeOverride maxGridSizeOverride( maxGridSize );
      test();
   }
}
