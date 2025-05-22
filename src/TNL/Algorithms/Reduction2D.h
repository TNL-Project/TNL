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
    * Parameters:
    *    identity: the [identity element](https://en.wikipedia.org/wiki/Identity_element)
    *              for the reduction operation, i.e. element which does not
    *              change the result of the reduction
    *    fetch: callable object such that `fetch( i, j )` yields the i-th value to be
    *           reduced from the j-th dataset (i = 0,...,size-1; j = 0,...,n-1)
    *    reduction: callable object representing the reduction operation
    *               for example, it can be an instance of std::plus, std::logical_and,
    *               std::logical_or etc.
    *    size: the size of each dataset
    *    n: number of datasets to be reduced
    *    result: callable object that returns a modifiable reference to the output array,
    *            it is used as `result( j ) = value` for `j = 0,...,n-1`.
    *            For example, it can be an `ArrayView` of size `n`.
    */
   template< typename Result, typename Fetch, typename Reduction, typename Index, typename Output >
   static constexpr void
   reduce( Result identity, Fetch fetch, Reduction reduction, Index size, int n, Output result );
};

template<>
struct Reduction2D< Devices::Host >
{
   /**
    * Parameters:
    *    identity: the [identity element](https://en.wikipedia.org/wiki/Identity_element)
    *              for the reduction operation, i.e. element which does not
    *              change the result of the reduction
    *    fetch: callable object such that `fetch( i, j )` yields the i-th value to be
    *           reduced from the j-th dataset (i = 0,...,size-1; j = 0,...,n-1)
    *    reduction: callable object representing the reduction operation
    *               for example, it can be an instance of std::plus, std::logical_and,
    *               std::logical_or etc.
    *    size: the size of each dataset
    *    n: number of datasets to be reduced
    *    result: callable object that returns a modifiable reference to the output array,
    *            it is used as `result( j ) = value` for `j = 0,...,n-1`
    *            For example, it can be an `ArrayView` of size `n`.
    */
   template< typename Result, typename Fetch, typename Reduction, typename Index, typename Output >
   static void
   reduce( Result identity, Fetch fetch, Reduction reduction, Index size, int n, Output result );
};

template<>
struct Reduction2D< Devices::Cuda >
{
   /**
    * Parameters:
    *    identity: the [identity element](https://en.wikipedia.org/wiki/Identity_element)
    *              for the reduction operation, i.e. element which does not
    *              change the result of the reduction
    *    fetch: callable object such that `fetch( i, j )` yields the i-th value to be
    *           reduced from the j-th dataset (i = 0,...,size-1; j = 0,...,n-1)
    *    reduction: callable object representing the reduction operation
    *               for example, it can be an instance of std::plus, std::logical_and,
    *               std::logical_or etc.
    *    size: the size of each dataset
    *    n: number of datasets to be reduced
    *    hostResult: callable object that returns a modifiable reference to the output array,
    *                it is used as `hostResult( j ) = value` for `j = 0,...,n-1`
    *                For example, it can be an `ArrayView` of size `n`.
    */
   template< typename Result, typename Fetch, typename Reduction, typename Index, typename Output >
   static void
   reduce( Result identity, Fetch fetch, Reduction reduction, Index size, int n, Output hostResult );
};

}  // namespace TNL::Algorithms

#include "Reduction2D.hpp"
