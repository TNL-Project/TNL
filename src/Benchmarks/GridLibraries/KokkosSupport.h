// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#ifdef TNL_BENCHMARKS_HAVE_KOKKOS

   #include <algorithm>
   #include <array>
   #include <cstddef>
   #include <string>
   #include <type_traits>

   #include <Kokkos_Core.hpp>

   #include <TNL/Devices/GPU.h>
   #include <TNL/Devices/Host.h>
   #include <TNL/Devices/Sequential.h>

namespace TNL::Benchmarks::KokkosSupport {

/**
 * \brief Kokkos execution space used for the given TNL device, `void` if the device is not supported.
 *
 * The sequential device is mapped to `Kokkos::Serial`, the host device to `Kokkos::OpenMP` and the GPU device to
 * the CUDA or HIP execution space. The host device is not mapped to `Kokkos::Serial`, so that the parallel TNL
 * implementations are not compared with a sequential Kokkos code.
 */
template< typename Device >
struct ExecutionSpace
{
   using type = void;
};

   #ifdef KOKKOS_ENABLE_SERIAL
template<>
struct ExecutionSpace< Devices::Sequential >
{
   using type = Kokkos::Serial;
};
   #endif

   #ifdef KOKKOS_ENABLE_OPENMP
template<>
struct ExecutionSpace< Devices::Host >
{
   using type = Kokkos::OpenMP;
};
   #endif

   #if defined( KOKKOS_ENABLE_CUDA ) || defined( KOKKOS_ENABLE_HIP )
template<>
struct ExecutionSpace< Devices::GPU >
{
   using type = Kokkos::DefaultExecutionSpace;
};
   #endif

/**
 * \brief Checks whether Kokkos supports the given TNL device in this build.
 */
template< typename Device >
constexpr bool
isSupported()
{
   return ! std::is_void_v< typename ExecutionSpace< Device >::type >;
}

template< typename Real, int Dimension >
struct ViewDataType;

template< typename Real >
struct ViewDataType< Real, 1 >
{
   using type = Real*;
};

template< typename Real >
struct ViewDataType< Real, 2 >
{
   using type = Real**;
};

template< typename Real >
struct ViewDataType< Real, 3 >
{
   using type = Real***;
};

/**
 * \brief Kokkos view of a multi-dimensional array with the first index changing fastest, i.e. with the same
 * layout as the arrays in the TNL benchmarks.
 */
template< typename Real, int Dimension, typename ExecSpace >
using View =
   Kokkos::View< typename ViewDataType< Real, Dimension >::type, Kokkos::LayoutLeft, typename ExecSpace::memory_space >;

/**
 * \brief Unmanaged Kokkos view of the data of a TNL vector.
 */
template< typename Real, int Dimension, typename ExecSpace >
using UnmanagedView = Kokkos::View<
   typename ViewDataType< Real, Dimension >::type,
   Kokkos::LayoutLeft,
   typename ExecSpace::memory_space,
   Kokkos::MemoryTraits< Kokkos::Unmanaged > >;

/**
 * \brief Creates a Kokkos view with the given sizes.
 */
template< typename ViewType, int Dimension >
ViewType
createView( const std::string& label, const std::array< int, Dimension >& sizes )
{
   if constexpr( Dimension == 1 )
      return ViewType( label, sizes[ 0 ] );
   else if constexpr( Dimension == 2 )
      return ViewType( label, sizes[ 0 ], sizes[ 1 ] );
   else
      return ViewType( label, sizes[ 0 ], sizes[ 1 ], sizes[ 2 ] );
}

/**
 * \brief Wraps the data of a TNL vector into an unmanaged Kokkos view with the given sizes.
 */
template< typename ViewType, int Dimension, typename Real >
ViewType
wrap( Real* data, const std::array< int, Dimension >& sizes )
{
   if constexpr( Dimension == 1 )
      return ViewType( data, sizes[ 0 ] );
   else if constexpr( Dimension == 2 )
      return ViewType( data, sizes[ 0 ], sizes[ 1 ] );
   else
      return ViewType( data, sizes[ 0 ], sizes[ 1 ], sizes[ 2 ] );
}

/**
 * \brief Kokkos policy for the index range `[ begin, end )`.
 *
 * The multi-dimensional policy iterates with the first index changing fastest, which corresponds to the layout of
 * the data. The default iteration order of Kokkos on the host is the opposite one.
 */
template< typename ExecSpace, int Dimension >
auto
createPolicy( const std::array< int, Dimension >& begin, const std::array< int, Dimension >& end )
{
   // On the GPU, the tiles are the thread blocks. The default tiles of Kokkos have only 32 (2D) or 64 (3D) threads
   // (CUDA), so the GPU uses the same block sizes as TNL instead (see TNL::Algorithms::detail::ParallelFor2D and
   // ParallelFor3D). On the CPU, the default tiles (whole rows along the x axis) are kept.
   constexpr bool onHost = Kokkos::SpaceAccessibility< ExecSpace, Kokkos::HostSpace >::accessible;
   if constexpr( Dimension == 1 )
      return Kokkos::RangePolicy< ExecSpace >( begin[ 0 ], end[ 0 ] );
   else if constexpr( Dimension == 2 ) {
      using Policy = Kokkos::MDRangePolicy< ExecSpace, Kokkos::Rank< 2, Kokkos::Iterate::Left, Kokkos::Iterate::Left > >;
      if constexpr( onHost )
         return Policy( { begin[ 0 ], begin[ 1 ] }, { end[ 0 ], end[ 1 ] } );
      else
         return Policy( { begin[ 0 ], begin[ 1 ] }, { end[ 0 ], end[ 1 ] }, { 32, 8 } );
   }
   else {
      using Policy = Kokkos::MDRangePolicy< ExecSpace, Kokkos::Rank< 3, Kokkos::Iterate::Left, Kokkos::Iterate::Left > >;
      if constexpr( onHost )
         return Policy( { begin[ 0 ], begin[ 1 ], begin[ 2 ] }, { end[ 0 ], end[ 1 ], end[ 2 ] } );
      else
         return Policy( { begin[ 0 ], begin[ 1 ], begin[ 2 ] }, { end[ 0 ], end[ 1 ], end[ 2 ] }, { 16, 4, 4 } );
   }
}

/**
 * \brief Kokkos functor which passes the multi-index of the cell to a kernel of any dimension.
 */
template< int Dimension, typename Kernel >
struct CellFunctor
{
   Kernel kernel;

   KOKKOS_INLINE_FUNCTION
   void
   operator()( int i ) const
   {
      kernel( Kokkos::Array< int, Dimension >{ i } );
   }

   KOKKOS_INLINE_FUNCTION
   void
   operator()( int i, int j ) const
   {
      kernel( Kokkos::Array< int, Dimension >{ i, j } );
   }

   KOKKOS_INLINE_FUNCTION
   void
   operator()( int i, int j, int k ) const
   {
      kernel( Kokkos::Array< int, Dimension >{ i, j, k } );
   }
};

/**
 * \brief Kokkos functor for the CPU which calls the kernel for one row of indices along the x axis.
 *
 * The rows are numbered with the y index changing fastest, so that the threads get contiguous blocks of rows.
 */
template< int Dimension, typename Kernel >
struct RowFunctor
{
   Kernel kernel;
   int xBegin, xEnd, yBegin, ySize, zBegin;

   KOKKOS_INLINE_FUNCTION
   void
   operator()( int row ) const
   {
      const int j = yBegin + row % ySize;
      if constexpr( Dimension == 2 ) {
         for( int i = xBegin; i < xEnd; i++ )
            kernel( i, j );
      }
      else {
         const int k = zBegin + row / ySize;
         for( int i = xBegin; i < xEnd; i++ )
            kernel( i, j, k );
      }
   }
};

/**
 * \brief Calls `kernel( i )`, `kernel( i, j )` or `kernel( i, j, k )` for all indices with `begin <= ( i, j, k ) < end`.
 *
 * On the GPU, the multi-dimensional range is processed by `Kokkos::MDRangePolicy`. On the CPU, the rows along the x axis
 * are distributed among the threads by `Kokkos::RangePolicy` and each row is processed by a sequential loop, which is
 * how TNL traverses multi-dimensional ranges on the CPU. `Kokkos::MDRangePolicy` with the default tiling was two to ten
 * times slower than this on the CPU (AMD EPYC 7281, GCC 13).
 */
template< typename ExecSpace, int Dimension, typename Kernel >
void
parallelFor(
   const std::string& label,
   const std::array< int, Dimension >& begin,
   const std::array< int, Dimension >& end,
   const Kernel& kernel )
{
   constexpr bool onHost = Kokkos::SpaceAccessibility< ExecSpace, Kokkos::HostSpace >::accessible;
   if constexpr( onHost && Dimension > 1 ) {
      const int ySize = end[ 1 ] - begin[ 1 ];
      const int zBegin = Dimension == 3 ? begin[ Dimension - 1 ] : 0;
      const int zSize = Dimension == 3 ? end[ Dimension - 1 ] - zBegin : 1;
      Kokkos::parallel_for( label,
                            Kokkos::RangePolicy< ExecSpace >( 0, ySize * zSize ),
                            RowFunctor< Dimension, Kernel >{ kernel, begin[ 0 ], end[ 0 ], begin[ 1 ], ySize, zBegin } );
   }
   else
      Kokkos::parallel_for( label, createPolicy< ExecSpace, Dimension >( begin, end ), kernel );
}

/**
 * \brief Calls `kernel( c )` for all cells `c` with `begin <= c < end`, where `c` is `Kokkos::Array< int, Dimension >`.
 */
template< typename ExecSpace, int Dimension, typename Kernel >
void
forCells(
   const std::string& label,
   const std::array< int, Dimension >& begin,
   const std::array< int, Dimension >& end,
   const Kernel& kernel )
{
   parallelFor< ExecSpace, Dimension >( label, begin, end, CellFunctor< Dimension, Kernel >{ kernel } );
}

/**
 * \brief Returns the element of the view at the given multi-index.
 */
template< typename View, std::size_t Dimension >
KOKKOS_INLINE_FUNCTION typename View::reference_type
at( const View& view, const Kokkos::Array< int, Dimension >& c )
{
   if constexpr( Dimension == 1 )
      return view( c[ 0 ] );
   else if constexpr( Dimension == 2 )
      return view( c[ 0 ], c[ 1 ] );
   else
      return view( c[ 0 ], c[ 1 ], c[ 2 ] );
}

/**
 * \brief Returns the multi-index shifted by `shift` along the given axis.
 */
template< std::size_t Dimension >
KOKKOS_INLINE_FUNCTION Kokkos::Array< int, Dimension >
shifted( Kokkos::Array< int, Dimension > c, int axis, int shift )
{
   c[ axis ] += shift;
   return c;
}

/**
 * \brief Returns the index range of the interior cells, i.e. the cells which are not on the boundary.
 */
template< int Dimension >
std::array< std::array< int, Dimension >, 2 >
getInterior( const std::array< int, Dimension >& sizes )
{
   std::array< int, Dimension > begin;
   std::array< int, Dimension > end;
   for( int i = 0; i < Dimension; i++ ) {
      begin[ i ] = 1;
      end[ i ] = sizes[ i ] - 1;
   }
   return { begin, end };
}

}  // namespace TNL::Benchmarks::KokkosSupport

#endif
