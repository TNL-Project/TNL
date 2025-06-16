// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Devices/Sequential.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

namespace TNL::Algorithms {

template< typename Device >
struct Reduction2D;

template<>
struct Reduction2D< Devices::Sequential >
{
   /**
    * \brief Performs reduction on a 2D dataset into a 1D output.
    *
    * This function applies a reduction operation across all elements along the first dimension
    * of a 2D dataset, combining values for each position in the second dimension.
    * The reduction starts with an identity element and aggregates all values along the first dimension.
    *
    * \tparam Result Type representing the identity element and result values.
    * \tparam Fetch Callable type used to fetch values from the 2D dataset.
    * \tparam Reduction Callable type representing the reduction operation.
    * \tparam Index Integral type used for the first dimension's size.
    * \tparam Output Callable type used to store the resulting values.
    *
    * \param identity [in] The [identity element](https://en.wikipedia.org/wiki/Identity_element)
    *                 for the reduction operation. This value does not affect the result of the reduction.
    * \param fetch [in] Callable object such that `fetch(i, j)` yields a value from the 2D dataset given
    *              by `i` ranging from `0` to `size-1` and `j` from `0` to `n-1`.
    * \param reduction [in] Callable object representing the reduction operation. Examples include
    *                  instances of \ref std::plus, \ref std::logical_and, \ref std::logical_or, etc.
    * \param size [in] The size of the first dimension (reduction axis) of the 2D dataset.
    * \param n [in] Number of elements in the second dimension of the 2D dataset.
    * \param result [out] Callable object returning a modifiable reference to the output array.
    *               Used as `result(j) = value` for `j` from `0` to `n-1`.
    *               For example, an \ref TNL::Containers::ArrayView "ArrayView" of size `n`.
    */
   template< typename Result, typename Fetch, typename Reduction, typename Index, typename Output >
   static constexpr void
   reduce( Result identity, Fetch fetch, Reduction reduction, Index size, int n, Output result );
};

template<>
struct Reduction2D< Devices::Host >
{
   /**
    * \brief Performs reduction on a 2D dataset into a 1D output.
    *
    * This function applies a reduction operation across all elements along the first dimension
    * of a 2D dataset, combining values for each position in the second dimension.
    * The reduction starts with an identity element and aggregates all values along the first dimension.
    *
    * \tparam Result Type representing the identity element and result values.
    * \tparam Fetch Callable type used to fetch values from the 2D dataset.
    * \tparam Reduction Callable type representing the reduction operation.
    * \tparam Index Integral type used for the first dimension's size.
    * \tparam Output Callable type used to store the resulting values.
    *
    * \param identity [in] The [identity element](https://en.wikipedia.org/wiki/Identity_element)
    *                 for the reduction operation. This value does not affect the result of the reduction.
    * \param fetch [in] Callable object such that `fetch(i, j)` yields a value from the 2D dataset given
    *              by `i` ranging from `0` to `size-1` and `j` from `0` to `n-1`.
    * \param reduction [in] Callable object representing the reduction operation. Examples include
    *                  instances of \ref std::plus, \ref std::logical_and, \ref std::logical_or, etc.
    * \param size [in] The size of the first dimension (reduction axis) of the 2D dataset.
    * \param n [in] Number of elements in the second dimension of the 2D dataset.
    * \param result [out] Callable object returning a modifiable reference to the output array.
    *               Used as `result(j) = value` for `j` from `0` to `n-1`.
    *               For example, an \ref TNL::Containers::ArrayView "ArrayView" of size `n`.
    */
   template< typename Result, typename Fetch, typename Reduction, typename Index, typename Output >
   static void
   reduce( Result identity, Fetch fetch, Reduction reduction, Index size, int n, Output result );
};

template<>
struct Reduction2D< Devices::Cuda >
{
   /**
    * \brief Performs reduction on a 2D dataset into a 1D output.
    *
    * This function applies a reduction operation across all elements along the first dimension
    * of a 2D dataset, combining values for each position in the second dimension.
    * The reduction starts with an identity element and aggregates all values along the first dimension.
    *
    * \tparam Result Type representing the identity element and result values.
    * \tparam Fetch Callable type used to fetch values from the 2D dataset.
    * \tparam Reduction Callable type representing the reduction operation.
    * \tparam Index Integral type used for the first dimension's size.
    * \tparam Output Callable type used to store the resulting values.
    *
    * \param identity [in] The [identity element](https://en.wikipedia.org/wiki/Identity_element)
    *                 for the reduction operation. This value does not affect the result of the reduction.
    * \param fetch [in] Callable object such that `fetch(i, j)` yields a value from the 2D dataset given
    *              by `i` ranging from `0` to `size-1` and `j` from `0` to `n-1`.
    * \param reduction [in] Callable object representing the reduction operation. Examples include
    *                  instances of \ref std::plus, \ref std::logical_and, \ref std::logical_or, etc.
    * \param size [in] The size of the first dimension (reduction axis) of the 2D dataset.
    * \param n [in] Number of elements in the second dimension of the 2D dataset.
    * \param hostResult [out] Callable object returning a modifiable reference to the output array.
    *                   Used as `result(j) = value` for `j` from `0` to `n-1`.
    *                   For example, an \ref TNL::Containers::ArrayView "ArrayView" of size `n`.
    *                   Note that the output array must be allocated on \ref TNL::Devices::Host "Host".
    */
   template< typename Result, typename Fetch, typename Reduction, typename Index, typename Output >
   static void
   reduce( Result identity, Fetch fetch, Reduction reduction, Index size, int n, Output hostResult );
};

}  // namespace TNL::Algorithms

#include "Reduction2D.hpp"
