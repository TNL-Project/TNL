// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#ifdef TNL_BENCHMARKS_HAVE_GRIDTOOLS

   #include <array>
   #include <type_traits>

   #include <gridtools/common/defs.hpp>
   #include <gridtools/common/tuple.hpp>
   #include <gridtools/common/tuple_util.hpp>
   #include <gridtools/sid/simple_ptr_holder.hpp>
   #include <gridtools/sid/synthetic.hpp>
   #include <gridtools/stencil/cartesian.hpp>
   #include <gridtools/stencil/cpu_ifirst.hpp>
   #include <gridtools/stencil/global_parameter.hpp>
   #include <gridtools/thread_pool/dummy.hpp>
   #ifdef GT_CUDACC
      #include <gridtools/stencil/gpu.hpp>
   #endif

   #include <TNL/Devices/GPU.h>
   #include <TNL/Devices/Host.h>
   #include <TNL/Devices/Sequential.h>

namespace TNL::Benchmarks::GridToolsSupport {

/**
 * \brief GridTools stencil backend used for the given TNL device, `void` if the device is not supported.
 *
 * GridTools is header-only and the backend is selected by the compiler: the CPU backend with the first index changing
 * fastest runs sequentially for the sequential device and with OpenMP for the host device, the GPU backend runs on
 * CUDA or HIP.
 */
template< typename Device >
struct Backend
{
   using type = void;
};

template<>
struct Backend< Devices::Sequential >
{
   using type = gridtools::stencil::cpu_ifirst< gridtools::thread_pool::dummy >;
};

   #ifdef _OPENMP
template<>
struct Backend< Devices::Host >
{
   using type = gridtools::stencil::cpu_ifirst< gridtools::thread_pool::omp >;
};
   #endif

   #ifdef GT_CUDACC
template<>
struct Backend< Devices::GPU >
{
   using type = gridtools::stencil::gpu<>;
};
   #endif

/**
 * \brief Checks whether GridTools supports the given TNL device in this build.
 */
template< typename Device >
constexpr bool
isSupported()
{
   return ! std::is_void_v< typename Backend< Device >::type >;
}

/**
 * \brief Extent of the stencil accessors with the offsets `[ -minus, plus ]` along the axes of the problem and no
 * offsets along the remaining axes.
 *
 * GridTools stencils are three-dimensional, 1D and 2D problems use one cell along the remaining axes.
 */
template< int Dimension, int Minus = 1, int Plus = 1 >
using Extent = gridtools::stencil::extent<
   -Minus,
   Plus,
   1 < Dimension ? -Minus : 0,
   1 < Dimension ? Plus : 0,
   2 < Dimension ? -Minus : 0,
   2 < Dimension ? Plus : 0 >;

/**
 * \brief Interior of the grid with the given sizes, i.e. the cells which are not on the boundary.
 *
 * GridTools iterates over the computational domain `[ 0, size )` along each axis relative to the origin of the data,
 * so the data are wrapped with the origin at the first interior cell.
 */
template< int Dimension >
struct Interior
{
   explicit Interior( const std::array< int, Dimension >& gridSizes )
   {
      strides = { 1, gridSizes[ 0 ], Dimension > 1 ? gridSizes[ 0 ] * gridSizes[ 1 ] : gridSizes[ 0 ] };
      for( int d = 0; d < 3; d++ )
         sizes[ d ] = d < Dimension ? gridSizes[ d ] - 2 : 1;
      offset = 0;
      for( int d = 0; d < Dimension; d++ )
         offset += strides[ d ];
   }

   /**
    * \brief Returns the GridTools grid, i.e. the iteration space, of the interior cells.
    */
   [[nodiscard]] auto
   getGrid() const
   {
      return gridtools::stencil::make_grid( sizes[ 0 ], sizes[ 1 ], sizes[ 2 ] );
   }

   /**
    * \brief Wraps the data of a TNL vector into a GridTools SID with the origin at the first interior cell.
    */
   template< typename Real >
   [[nodiscard]] auto
   wrap( Real* data ) const
   {
      using gridtools::sid::property;
      return gridtools::sid::synthetic()
         .set< property::origin >( gridtools::sid::host_device::simple_ptr_holder< Real* >( data + offset ) )
         .template set< property::strides >(
            gridtools::tuple( gridtools::integral_constant< int, 1 >(), strides[ 1 ], strides[ 2 ] ) )
         .template set< property::strides_kind, StridesKind >();
   }

   //! \brief Tag telling GridTools that all wrapped vectors have the same strides.
   struct StridesKind
   {};

   std::array< int, 3 > strides;
   std::array< int, 3 > sizes;
   int offset;
};

/**
 * \brief Returns the unit offset along the given axis for the stencil accessors.
 */
GT_FUNCTION constexpr int
unit( int axis, int d )
{
   return axis == d ? 1 : 0;
}

}  // namespace TNL::Benchmarks::GridToolsSupport

#endif
