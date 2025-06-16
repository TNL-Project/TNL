// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <utility>

#include <TNL/Containers/NDArray.h>
#include <TNL/Algorithms/detail/Reduction.h>
#include <TNL/Algorithms/Reduction2D.h>
#include <TNL/Algorithms/Reduction3D.h>

namespace TNL::Containers {
/*
template< typename Result, typename DataFetcher, typename Reduction, typename Index, typename Output >
void constexpr Reduction4D_sequential( Result identity,
                                       DataFetcher dataFetcher,
                                       Reduction reduction,
                                       Index size,
                                       int m,
                                       int n,
                                       int l,
                                       Output result )
{
   TNL_ASSERT_GT( size, 0, "The size of datasets must be positive." );
   TNL_ASSERT_GT( m, 0, "The number of datasets must be positive." );
   TNL_ASSERT_GT( n, 0, "The number of datasets must be positive." );
   TNL_ASSERT_GT( l, 0, "The number of datasets must be positive." );

   for (int i = 0; i < m; i++){
      for (int j = 0; j < n; j++){
         for (int k = 0; k < l; k++){
            result(i, j, k) = identity;
         }
      }
   }

   for( int v = 0; v < size; v++ ) {
      for( int i = 0; i < m; i++ ) {
         for( int j = 0; j < n; j++ ) {
            for( int k = 0; k < l; k++ ) {
               result(i, j, k) = reduction( result(i, j, k), dataFetcher( v, i, j, k ) );
            }
         }
      }
   }
}
*/

/**
 * \brief `nd_reduce` implements [(parallel) reduction](https://en.wikipedia.org/wiki/Reduce_(parallel_pattern))
 * across specified axis of N-dimensional array (\ref NDArray).
 *
 * This function reduces an N-dimensional array to a lower-dimensional array by applying a binary operation
 * along the specified axis. The reduction operation must be associative and commutative for correct results
 * in parallel execution. The function handles different array dimensions (1D to 6D) with customizable axis
 * configuration.
 *
 * \tparam axis is the dimension index to reduce over (compile-time constant)
 * \tparam Input is the `NDArray` type providing access to data and dimensions
 * \tparam Reduction is a binary function object for reduction operation
 * \tparam Output is the resulting array type with reduced dimensions
 *
 * \param input is the input `NDArray` to be reduced
 * \param reduction is a binary function object defining the reduction operation
 *                  This can be a user-defined lambda function or an instance of
 *                  some \ref ReductionFunctionObjects.
 * \param identity is the [identity element](https://en.wikipedia.org/wiki/Identity_element)
 *                 for the reduction operation
 * \param output is the result array with dimensions reduced by removing the specified axis
 *
 * The function handles different dimensions through template specialization:
 *
 * - 1D: Direct scalar reduction
 * - 2D: Reduction along specified axis using 2D reduction kernel
 * - 3D: Reduction along specified axis using 3D reduction kernel
 * - 4D-6D: Uses permutation sequences to handle axis ordering with parallel execution
 *
 * The reduction operation is applied across all elements along the specified axis.
 * For higher dimensions (4D+), the implementation uses
 * \ref TNL::Algorithms::parallelFor "parallelFor" with a kernel that processes sub-ranges
 * of the output dimensions.
 *
 * \par Example (2D Reduction)
 *
 * \code
 * // Create 2D array
 * using Array = TNL::Containers::NDArray< double,
 *                                         TNL::Containers::SizesHolder< int, 0, 0 >,
 *                                         std::index_sequence< 0, 1 >,
 *                                         TNL::Devices::Host >;
 * Array array;
 * array.setSizes( 3, 4 );
 * // Initialize array with values...
 *
 * // Create output 1D array
 * TNL::Containers::Array< double, TNL::Devices::Host, int > result;
 *
 * // Perform reduction along axis 0
 * TNL::Containers::nd_reduce< 0 >( arr, std::plus<>(), 0.0, result );
 * \endcode
 *
 * \par Example (3D Reduction)
 *
 * \code
 * // Create 3D array
 * using Array = TNL::Containers::NDArray< float,
 *                                         TNL::Containers::SizesHolder< int, 0, 0, 0 >,
 *                                         std::index_sequence< 0, 1, 2 >,
 *                                         TNL::Devices::Host >;
 * Array array;
 * array.setSizes( 3, 4, 5 );
 *
 * // Create output 2D array
 * using OutputArray = TNL::Containers::NDArray< float,
 *                                               TNL::Containers::SizesHolder< int, 0, 0 >,
 *                                               std::index_sequence< 0, 1 >,
 *                                               TNL::Devices::Host >;
 * OutputArray result;
 *
 * // Perform maximum reduction along axis 1 (returns 2D array of size `(3, 5)`)
 * TNL::Containers::nd_reduce< 1 >( arr, TNL::Max(), -INFINITY, result );
 * \endcode
 *
 * \note The output array dimensions are automatically set based on the input dimensions
 *       with the specified axis removed. For example, reducing a 3D array with dimensions
 *       \f$(X,Y,Z)\f$ along axis 1 will produce an output array with dimensions \f$(X,Z)\f$.
 *
 * \note The function uses permutation sequences to handle non-contiguous memory access
 *       patterns when the reduction axis is not the first dimension. This ensures
 *       correct memory access patterns for performance optimization.
 *
 * \see reduce for 1D reduction implementation
 */
template< std::size_t axis = 0, typename Input, typename Reduction, typename Output >
void
nd_reduce( const Input& input, Reduction reduction, typename Input::ValueType identity, Output& output )
{
   //1. Dimension of input (1D, 2D, 3D)
   constexpr std::size_t dimension = Input::getDimension();

   static_assert( axis <= dimension, "Axis must be less than or equal to dimension" );

   using DeviceType = typename Input::DeviceType;
   using IndexType = typename Input::IndexType;

   auto input_view = input.getConstView();

   if constexpr( dimension == 1 ) {
      output = Algorithms::detail::Reduction< DeviceType >::reduce(
         0, input.template getSize< 0 >(), input_view, reduction, identity );
   }
   else if constexpr( dimension == 2 ) {
      //2. size of input -> allocate memory for output
      output.setSizes( input.template getSize< 1 - axis >() );

      auto output_view = output.getView();
      //3. define dataFetcher
      auto fetch = [ = ] __cuda_callable__( IndexType i, IndexType k )
      {
         if( axis == 0 )
            return input_view( i, k );
         else
            return input_view( k, i );
      };

      // 4. call reduction function for specific dimension
      Algorithms::Reduction2D< DeviceType >::reduce(
         identity, fetch, reduction, input.template getSize< axis >(), input.template getSize< 1 - axis >(), output_view );
   }
   else if constexpr( dimension == 3 ) {
      //2. size of input -> allocate memory for output
      constexpr std::size_t axis1 = std::max( 2 - axis, std::size_t( 1 ) ) - 1;
      constexpr std::size_t axis2 = std::min( dimension - axis, std::size_t( 2 ) );

      using Permutation = std::index_sequence< axis, axis1, axis2 >;

      output.setSizes( input.template getSize< axis1 >(), input.template getSize< axis2 >() );

      auto output_view = output.getView();

      // 3. define dataFetcher
      auto fetch = [ = ] __cuda_callable__( IndexType i, IndexType j, IndexType k )
      {
         return detail::call_with_unpermuted_arguments< Permutation >( input_view, i, j, k );
      };

      auto result = [ = ] __cuda_callable__( IndexType m, IndexType n ) mutable -> typename Output::ValueType&
      {
         return output_view( m, n );
      };

      // 4. call reduction function for specific dimension
      Algorithms::Reduction3D< DeviceType >::reduce( identity,
                                                     fetch,
                                                     reduction,
                                                     input.template getSize< axis >(),
                                                     input.template getSize< axis1 >(),
                                                     input.template getSize< axis2 >(),
                                                     result );
   }
   else if constexpr( dimension == 4 ) {
      //2. size of input -> allocate memory for output
      constexpr std::size_t axis1 = axis >= 1 ? 0 : 1;
      constexpr std::size_t axis2 = axis >= 2 ? 1 : 2;
      constexpr std::size_t axis3 = axis == 3 ? 2 : 3;

      static_assert( axis + axis1 + axis2 + axis3 == 6, "Wrong indexing of axis" );

      using Permutation = std::index_sequence< axis, axis1, axis2, axis3 >;

      output.setSizes(
         input.template getSize< axis1 >(), input.template getSize< axis2 >(), input.template getSize< axis3 >() );

      auto output_view = output.getView();

      using MultiIndex = Containers::StaticArray< 3, IndexType >;

      // 3. define dataFetcher
      auto kernel = [ = ] __cuda_callable__( MultiIndex idx ) mutable
      {
         auto fetch = [ = ]( IndexType l )
         {
            return detail::call_with_unpermuted_arguments< Permutation >( input_view, l, idx[ 0 ], idx[ 1 ], idx[ 2 ] );
         };
         output_view( idx[ 0 ], idx[ 1 ], idx[ 2 ] ) = Algorithms::detail::Reduction< Devices::Sequential >::reduce(
            0, input_view.template getSize< axis >(), fetch, reduction, identity );
      };

      const MultiIndex begin = { 0, 0, 0 };
      const MultiIndex end = { input.template getSize< axis1 >(),
                               input.template getSize< axis2 >(),
                               input.template getSize< axis3 >() };
      Algorithms::parallelFor< DeviceType >( begin, end, kernel );
   }
   else if constexpr( dimension == 5 ) {
      //2. size of input -> allocate memory for output
      constexpr std::size_t axis1 = axis >= 1 ? 0 : 1;
      constexpr std::size_t axis2 = axis >= 2 ? 1 : 2;
      constexpr std::size_t axis3 = axis >= 3 ? 2 : 3;
      constexpr std::size_t axis4 = axis == 4 ? 3 : 4;

      static_assert( axis + axis1 + axis2 + axis3 + axis4 == 10, "Wrong indexing of axis" );

      using Permutation = std::index_sequence< axis, axis1, axis2, axis3, axis4 >;

      output.setSizes( input.template getSize< axis1 >(),
                       input.template getSize< axis2 >(),
                       input.template getSize< axis3 >(),
                       input.template getSize< axis4 >() );

      using MultiIndex = Containers::StaticArray< 3, IndexType >;

      auto output_view = output.getView();

      // 3. define dataFetcher
      auto kernel = [ = ] __cuda_callable__( MultiIndex idx ) mutable
      {
         auto fetch = [ = ]( IndexType i, IndexType j )
         {
            return detail::call_with_unpermuted_arguments< Permutation >( input_view, i, j, idx[ 0 ], idx[ 1 ], idx[ 2 ] );
         };

         auto result = [ = ]( IndexType k ) mutable -> typename Output::ValueType&
         {
            return output_view( k, idx[ 0 ], idx[ 1 ], idx[ 2 ] );
         };

         Algorithms::Reduction2D< Devices::Sequential >::reduce(
            identity, fetch, reduction, input.template getSize< axis >(), input.template getSize< axis1 >(), result );
      };

      const MultiIndex begin = { 0, 0, 0 };
      const MultiIndex end = { input.template getSize< axis2 >(),
                               input.template getSize< axis3 >(),
                               input.template getSize< axis4 >() };
      Algorithms::parallelFor< DeviceType >( begin, end, kernel );
   }
   else if constexpr( dimension == 6 ) {
      //2. size of input -> allocate memory for output
      constexpr std::size_t axis1 = axis >= 1 ? 0 : 1;
      constexpr std::size_t axis2 = axis >= 2 ? 1 : 2;
      constexpr std::size_t axis3 = axis >= 3 ? 2 : 3;
      constexpr std::size_t axis4 = axis >= 4 ? 3 : 4;
      constexpr std::size_t axis5 = axis == 5 ? 4 : 5;

      static_assert( axis + axis1 + axis2 + axis3 + axis4 + axis5 == 15, "Wrong indexing of axis" );

      using Permutation = std::index_sequence< axis, axis1, axis2, axis3, axis4, axis5 >;

      output.setSizes( input.template getSize< axis1 >(),
                       input.template getSize< axis2 >(),
                       input.template getSize< axis3 >(),
                       input.template getSize< axis4 >(),
                       input.template getSize< axis5 >() );

      using MultiIndex = Containers::StaticArray< 3, IndexType >;

      auto output_view = output.getView();

      // 3. define dataFetcher
      auto kernel = [ = ] __cuda_callable__( MultiIndex idx ) mutable
      {
         auto fetch = [ = ]( IndexType i, IndexType j, IndexType k )
         {
            return detail::call_with_unpermuted_arguments< Permutation >( input_view, i, j, k, idx[ 0 ], idx[ 1 ], idx[ 2 ] );
         };

         auto result = [ = ]( IndexType m, IndexType n ) mutable -> typename Output::ValueType&
         {
            return output_view( m, n, idx[ 0 ], idx[ 1 ], idx[ 2 ] );
         };

         Algorithms::Reduction3D< Devices::Sequential >::reduce( identity,
                                                                 fetch,
                                                                 reduction,
                                                                 input.template getSize< axis >(),
                                                                 input.template getSize< axis1 >(),
                                                                 input.template getSize< axis2 >(),
                                                                 result );
      };

      const MultiIndex begin = { 0, 0, 0 };
      const MultiIndex end = { input.template getSize< axis3 >(),
                               input.template getSize< axis4 >(),
                               input.template getSize< axis5 >() };
      Algorithms::parallelFor< DeviceType >( begin, end, kernel );
   }
   /*
   else if constexpr( dimension == 7 ) {
      //2. size of input -> allocate memory for output
      constexpr std::size_t axis1 = axis >= 1 ? 0 : 1;
      constexpr std::size_t axis2 = axis >= 2 ? 1 : 2;
      constexpr std::size_t axis3 = axis >= 3 ? 2 : 3;
      constexpr std::size_t axis4 = axis >= 4 ? 3 : 4;
      constexpr std::size_t axis5 = axis >= 5 ? 4 : 5;
      constexpr std::size_t axis6 = axis == 6 ? 5 : 6;

      static_assert( axis + axis1 + axis2 + axis3 + axis4 + axis5 + axis6 == 21, "Wrong indexing of axis" );

      using Permutation = std::index_sequence< axis, axis1, axis2, axis3, axis4, axis5, axis6 >;

      output.setSizes( input.template getSize< axis1 >(),
                       input.template getSize< axis2 >(),
                       input.template getSize< axis3 >(),
                       input.template getSize< axis4 >(),
                       input.template getSize< axis5 >(),
                       input.template getSize< axis6 >() );

      auto output_view = output.getView();

      using MultiIndex = Containers::StaticArray< 3, IndexType >;

      // 3. define dataFetcher
      auto kernel = [ = ] __cuda_callable__( MultiIndex idx ) mutable
      {
         auto fetch = [ = ]( IndexType i, IndexType j, IndexType k, IndexType l )
         {
            return detail::call_with_unpermuted_arguments< Permutation >(
               input_view, i, j, k, l, idx[ 0 ], idx[ 1 ], idx[ 2 ] );
         };

         auto result = [ = ]( IndexType m, IndexType n, IndexType b ) mutable -> typename Output::ValueType&
         {
            return output_view( m, n, b, idx[0], idx[1], idx[2] );
         };

         Reduction4D_sequential( 0,
                                 fetch,
                                 reduction,
                                 input.template getSize< axis >(),
                                 input.template getSize< axis1 >(),
                                 input.template getSize< axis2 >(),
                                 input.template getSize< axis3 >(),
                                 result );
      };

      const MultiIndex begin = { 0, 0, 0 };
      const MultiIndex end = { input.template getSize< axis4 >(),
                               input.template getSize< axis5 >(),
                               input.template getSize< axis6 >() };
      Algorithms::parallelFor< DeviceType >( begin, end, kernel );
   }
   */
}

/**
 * \brief `nd_reduce` implements [(parallel) reduction](https://en.wikipedia.org/wiki/Reduce_(parallel_pattern))
 * across specified axis of N-dimensional array (\ref NDArray) with automatic output allocation.
 *
 * This overloaded version automatically creates and returns a new `NDArray` with dimensions reduced by removing
 * the specified axis. It wraps the original \ref nd_reduce function that requires an output array parameter.
 *
 * \tparam axis is the dimension index to reduce over (compile-time constant)
 * \tparam Input is the NDArray type providing access to data and dimensions
 * \tparam Reduction is a binary function object for reduction operation
 *
 * \param input is the input `NDArray` to be reduced
 * \param reduction is a binary function object defining the reduction operation
 *                  This can be a user-defined lambda function or an instance of
 *                  some \ref ReductionFunctionObjects.
 * \param identity is the [identity element](https://en.wikipedia.org/wiki/Identity_element)
 *                 for the reduction operation
 * \return auto-deduced `NDArray` with dimensions reduced by removing the specified axis
 *
 * The function creates an output array with dimensions determined by removing the specified axis from the input.
 * For example, reducing a 3D array with dimensions \f$(X,Y,Z)\f$ along axis 1 will produce a 2D array with dimensions
 * \f$(X,Z)\f$.
 *
 * The reduction operation is applied across all elements along the specified axis using the same logic
 * as the original \ref nd_reduce function.
 *
 * \par Example (2D Reduction)
 *
 * \code
 * // Create 2D array
 * using Array = TNL::Containers::NDArray< double,
 *                                         TNL::Containers::SizesHolder< int, 0, 0 >,
 *                                         std::index_sequence< 0, 1 >,
 *                                         TNL::Devices::Host >;
 * Array array;
 * array.setSizes( 3, 4 );
 * // Initialize array with values...
 *
 * // Perform reduction along axis 0 (returns 1D array of size 4)
 * auto result = TNL::Containers::nd_reduce< 0 >( array, std::plus<>(), 0.0 );
 * \endcode
 *
 * \par Example (3D Reduction)
 *
 * \code
 * // Create 3D array
 * using Array = TNL::Containers::NDArray< float,
 *                                         TNL::Containers::SizesHolder< int, 0, 0, 0 >,
 *                                         std::index_sequence< 0, 1, 2 >,
 *                                         TNL::Devices::Host >;
 * Array array;
 * array.setSizes( 3, 4, 5 );
 *
 * // Perform maximum reduction along axis 1 (returns 2D array of size `(3, 5)`)
 * auto result = TNL::Containers::nd_reduce< 1 >( array, TNL::Max(), -INFINITY );
 * \endcode
 *
 * \note This overload provides more convenient usage by automatically creating the output array with
 *       appropriate dimensions. The output array type is deduced based on input dimensionality.
 *
 * \note The returned array uses the same device type and memory layout as the input array.
 *
 * \note For 1D inputs, this function returns a scalar value rather than an array.
 *
 * \see nd_reduce for the original function with explicit output parameter
 */
template< std::size_t axis = 0, typename Input, typename Reduction >
auto
nd_reduce( const Input& input, Reduction reduction, typename Input::ValueType identity )
{
   constexpr std::size_t dimension = Input::getDimension();

   static_assert( axis <= dimension, "Axis must be less than or equal to dimension" );

   if constexpr( dimension == 1 ) {
      typename Input::ValueType output;
      nd_reduce< axis >( input, reduction, identity, output );
      return output;
   }
   else if constexpr( dimension == 2 ) {
      NDArray< typename Input::ValueType,
               SizesHolder< typename Input::ValueType, 0 >,
               std::index_sequence< 0 >,
               typename Input::DeviceType >
         output;
      nd_reduce< axis >( input, reduction, identity, output );
      return output;
   }
   else if constexpr( dimension == 3 ) {
      NDArray< typename Input::ValueType,
               SizesHolder< typename Input::ValueType, 0, 0 >,
               std::index_sequence< 0, 1 >,
               typename Input::DeviceType >
         output;
      nd_reduce< axis >( input, reduction, identity, output );
      return output;
   }
   else if constexpr( dimension == 4 ) {
      NDArray< typename Input::ValueType,
               SizesHolder< typename Input::ValueType, 0, 0, 0 >,
               std::index_sequence< 0, 1, 2 >,
               typename Input::DeviceType >
         output;
      nd_reduce< axis >( input, reduction, identity, output );
      return output;
   }
   else if constexpr( dimension == 5 ) {
      NDArray< typename Input::ValueType,
               SizesHolder< typename Input::ValueType, 0, 0, 0, 0 >,
               std::index_sequence< 0, 1, 2, 3 >,
               typename Input::DeviceType >
         output;
      nd_reduce< axis >( input, reduction, identity, output );
      return output;
   }
   else if constexpr( dimension == 6 ) {
      NDArray< typename Input::ValueType,
               SizesHolder< typename Input::ValueType, 0, 0, 0, 0, 0 >,
               std::index_sequence< 0, 1, 2, 3, 4 >,
               typename Input::DeviceType >
         output;
      nd_reduce< axis >( input, reduction, identity, output );
      return output;
   }
   /*
   else if constexpr( dimension == 7 ) {
      NDArray< typename Input::ValueType,
               SizesHolder< typename Input::ValueType, 0, 0, 0, 0, 0, 0 >,
               std::index_sequence< 0, 1, 2, 3, 4, 5 >,
               typename Input::DeviceType > output;
      nd_reduce< axis >( input, reduction, identity, output );
      return output;
   }
   */
}

}  // namespace TNL::Containers

/*
The `nd_reduce_sequential` function template aims to extend reduction operations
to handle any dimensional arrays efficiently by combining sequential processing
for the initial dimensions with parallel processing for the last three dimensions
using GPU acceleration.

**How It Would Work**:
- The current library is limited to 3D reduction operations and extends to 6D
  using parallel for loops. However, this extension is limited.
- The `nd_reduce_sequential` function builds upon a simple 4D sequential
  implementation (`Reduction4D_sequential`).
- By using parameter packs, the function handles reductions on any number of
  dimensions, performing sequential reduction on the initial dimensions and
  parallel processing on the last three dimensions using GPU acceleration.
*/

/*
template< typename Result, typename DataFetcher, typename Reduction, typename Index, typename... RemainingDims, typename Output
> void constexpr nd_reduce_sequential( Result identity, DataFetcher dataFetcher, Reduction reduction, Index size,
                                     RemainingDims... dims,
                                     Output result )
{
   //RemainingDims...

   TNL_ASSERT_GT( size, 0, "The size of datasets must be positive." );

   for( int i = 0; i < RemainingDims...; i++ ) {
      result( i ) = identity;
   }

   for( int v = 0; v < size; v++ ) {
      for( int i = 0; i < RemainingDims...; i++ ) {
         result( i ) = reduction( result( i ), dataFetcher( v, i ) );
      }
   }

   //or use staticFor

   TNL::Algorithms::staticFor< Index, dims... >(
      [ & ]( ... )
      {
         ...
      } );
}

else if constexpr( dimension >= 4 ) {
   using MultiDims = Containers::StaticArray< dimension - 1, IndexType >;

   using MultiIndex = Containers::StaticArray< 3, IndexType >;

   MultiDims dims;
   for( int i = 0; i < dims.getSize(); ++i ) {
      dims[ i ] = axis >= i + 1 ? i : i + 1;
   }

   using Permutation = std::index_sequence< axis, dims... >;

   //output.setSizes(input.template getSize<dims>()...);
   auto setOutputSizes = [ &output, &input ]( auto... indices )
   {
      output.setSizes( input.template getSize< indices >()... );
   };
   std::apply( setOutputSizes, dims );

   auto output_view = output.getView();

   auto kernel = [ = ] __cuda_callable__( MultiIndex idx ) mutable
   {
      auto fetch = [ = ] __cuda_callable__( IndexType i, auto... remaining_dims )
      {
         return detail::call_with_unpermuted_arguments< Permutation >(
            input_view, i, remaining_dims..., idx[ 0 ], idx[ 1 ], idx[ 2 ] );
      };
      auto result = [ = ] __cuda_callable__( auto... remaining_dims ) mutable -> typename Output::ValueType&
      {
         return output_view( remaining_dims..., idx[ 0 ], idx[ 1 ], idx[ 2 ] );
      };
      nd_reduce_sequential( 0, fetch, reduction, input.template getSize< axis >(), dims..., result );
   };

   // Extract the last three dimensions for the parallel loop
   const MultiIndex begin = { 0, 0, 0 };
   const MultiIndex end = { input.template getSize< dims[ dimension - 3 ] >(),
                            input.template getSize< dims[ dimension - 2 ] >(),
                            input.template getSize< dims[ dimension - 1 ] >() };

   Algorithms::parallelFor< DeviceType >( begin, end, kernel );
}
*/
