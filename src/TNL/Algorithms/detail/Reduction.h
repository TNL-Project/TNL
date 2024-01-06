// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <utility>  // std::pair, std::forward

#include <TNL/Devices/Sequential.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

namespace TNL::Algorithms::detail {

template< typename Device >
struct Reduction;

template<>
struct Reduction< Devices::Sequential >
{
   template< typename Index, typename Result, typename Fetch, typename Reduce >
   static constexpr Result
   reduce( Index begin, Index end, Fetch&& fetch, Reduce&& reduce, const Result& identity );

   template< typename Index, typename Result, typename Fetch, typename Reduce >
   static constexpr std::pair< Result, Index >
   reduceWithArgument( Index begin, Index end, Fetch&& fetch, Reduce&& reduce, const Result& identity );
};

template<>
struct Reduction< Devices::Host >
{
   template< typename Index, typename Result, typename Fetch, typename Reduce >
   static Result
   reduce( Index begin, Index end, Fetch&& fetch, Reduce&& reduce, const Result& identity );

   template< typename Index, typename Result, typename Fetch, typename Reduce >
   static std::pair< Result, Index >
   reduceWithArgument( Index begin, Index end, Fetch&& fetch, Reduce&& reduce, const Result& identity );
};

template<>
struct Reduction< Devices::Cuda >
{
   template< typename Index, typename Result, typename Fetch, typename Reduce >
   static Result
   reduce( Index begin, Index end, Fetch&& fetch, Reduce&& reduce, const Result& identity );

   template< typename Index, typename Result, typename Fetch, typename Reduce >
   static std::pair< Result, Index >
   reduceWithArgument( Index begin, Index end, Fetch&& fetch, Reduce&& reduce, const Result& identity );
};

}  // namespace TNL::Algorithms::detail

#include <TNL/Algorithms/detail/Reduction.hpp>
