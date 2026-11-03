// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Backend/Macros.h>

namespace TNL::Graphs::Algorithms::detail {

// The lambdas passed to the edge traversals must be __cuda_callable__ even
// for Devices::Host, because the traversals wrap them in extended lambdas.
// These wrappers of the host-only TNL::Atomic< ..., Devices::Host > operations
// suppress the nvcc warning about calling a __host__ function from them.

TNL_NVCC_HD_WARNING_DISABLE
template< typename Atomic, typename Value >
__cuda_callable__
bool
hostAtomicCompareExchange( Atomic& atomic, Value& expected, const Value& desired )
{
   return atomic.compare_exchange_strong( expected, desired );
}

TNL_NVCC_HD_WARNING_DISABLE
template< typename Atomic, typename Value >
__cuda_callable__
Value
hostAtomicFetchMin( Atomic& atomic, const Value& value )
{
   return atomic.fetch_min( value );
}

}  // namespace TNL::Graphs::Algorithms::detail
