// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#ifdef TNL_BENCHMARKS_HAVE_AMREX

   #include <array>
   #include <type_traits>

   #include <AMReX_BaseFab.H>
   #include <AMReX_Box.H>
   #include <AMReX_BoxArray.H>
   #include <AMReX_DistributionMapping.H>
   #include <AMReX_FabArray.H>
   #include <AMReX_Gpu.H>
   #include <AMReX_MFIter.H>

   #include <TNL/Devices/GPU.h>
   #include <TNL/Devices/Host.h>
   #include <TNL/Devices/Sequential.h>

namespace TNL::Benchmarks::AMReXSupport {

/**
 * \brief Checks whether AMReX supports the given TNL device in this build.
 *
 * AMReX built with a GPU backend runs on the GPU only, AMReX built with OpenMP corresponds to the host device and
 * AMReX built without OpenMP to the sequential device.
 */
template< typename Device >
constexpr bool
isSupported()
{
   #if defined( AMREX_USE_GPU )
   return std::is_same_v< Device, Devices::GPU >;
   #elif defined( AMREX_USE_OMP )
   return std::is_same_v< Device, Devices::Host >;
   #else
   return std::is_same_v< Device, Devices::Sequential >;
   #endif
}

/**
 * \brief Array of AMReX boxes with the given element type, `amrex::MultiFab` for `double`.
 */
template< typename Real >
using FabArray = amrex::FabArray< amrex::BaseFab< Real > >;

/**
 * \brief Returns the box of all cells of the grid with the given sizes.
 *
 * AMReX is built with three dimensions, 1D and 2D problems use boxes with one cell along the remaining axes.
 */
template< int Dimension >
amrex::Box
createDomain( const std::array< int, Dimension >& sizes )
{
   amrex::IntVect hi( 0 );
   for( int d = 0; d < Dimension; d++ )
      hi[ d ] = sizes[ d ] - 1;
   return { amrex::IntVect( 0 ), hi };
}

/**
 * \brief Returns the box of the interior cells, i.e. the cells which are not on the boundary.
 */
template< int Dimension >
amrex::Box
getInterior( const amrex::Box& domain )
{
   amrex::Box interior = domain;
   for( int d = 0; d < Dimension; d++ )
      interior.grow( d, -1 );
   return interior;
}

/**
 * \brief Returns the iteration info of `amrex::MFIter`, with tiling for the CPU and without tiling for the GPU.
 *
 * The tiles distribute the work among the OpenMP threads. The default AMReX tile size has only one tile along the
 * x axis, so 1D problems use shorter tiles to be split among the threads.
 */
template< int Dimension >
amrex::MFItInfo
getIterationInfo()
{
   amrex::MFItInfo info;
   if( amrex::TilingIfNotGPU() ) {
      amrex::IntVect tileSize = amrex::FabArrayBase::mfiter_tile_size;
      if constexpr( Dimension == 1 )
         tileSize[ 0 ] = 4096;
      info.EnableTiling( tileSize );
   }
   return info;
}

/**
 * \brief Copies the data of a TNL vector to an array of AMReX boxes or back.
 *
 * The TNL vector stores the values with the x index changing fastest, which is the layout of AMReX boxes.
 * On the CPU, the copy runs in parallel over the same tiles as the computation, so that the memory pages of the AMReX
 * arrays are first touched by the threads which use them later (this matters on NUMA systems).
 */
template< typename Real, int Dimension >
void
copy( const Real* source, FabArray< Real >& target, const std::array< int, Dimension >& sizes )
{
   const amrex::Dim3 begin{ 0, 0, 0 };
   const amrex::Dim3 end{ sizes[ 0 ], Dimension > 1 ? sizes[ 1 ] : 1, Dimension > 2 ? sizes[ 2 ] : 1 };
   const amrex::Array4< const Real > sourceArray( source, begin, end, 1 );
   #ifdef AMREX_USE_OMP
      #pragma omp parallel if( amrex::Gpu::notInLaunchRegion() )
   #endif
   for( amrex::MFIter mfi( target, getIterationInfo< Dimension >() ); mfi.isValid(); ++mfi ) {
      const auto targetArray = target.array( mfi );
      amrex::ParallelFor(
         mfi.tilebox(),
         [ = ] AMREX_GPU_DEVICE( int i, int j, int k ) noexcept
         {
            targetArray( i, j, k ) = sourceArray( i, j, k );
         } );
   }
   amrex::Gpu::streamSynchronize();
}

template< typename Real, int Dimension >
void
copy( const FabArray< Real >& source, Real* target, const std::array< int, Dimension >& sizes )
{
   const amrex::Dim3 begin{ 0, 0, 0 };
   const amrex::Dim3 end{ sizes[ 0 ], Dimension > 1 ? sizes[ 1 ] : 1, Dimension > 2 ? sizes[ 2 ] : 1 };
   const amrex::Array4< Real > targetArray( target, begin, end, 1 );
   #ifdef AMREX_USE_OMP
      #pragma omp parallel if( amrex::Gpu::notInLaunchRegion() )
   #endif
   for( amrex::MFIter mfi( source, getIterationInfo< Dimension >() ); mfi.isValid(); ++mfi ) {
      const auto sourceArray = source.const_array( mfi );
      amrex::ParallelFor(
         mfi.tilebox(),
         [ = ] AMREX_GPU_DEVICE( int i, int j, int k ) noexcept
         {
            targetArray( i, j, k ) = sourceArray( i, j, k );
         } );
   }
   amrex::Gpu::streamSynchronize();
}

}  // namespace TNL::Benchmarks::AMReXSupport

#endif
