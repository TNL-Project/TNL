// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstddef>
#include <TNL/Math.h>

namespace TNL::Solvers::ODE::Methods {

/**
 * \brief Third order [Strong Stability Preserving
 * Runge-Kutta](https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods) method.
 *
 * \tparam Value is arithmetic type used for computations.
 */
template< typename Value = double >
struct SSPRK3
{
   using ValueType = Value;

   static constexpr std::size_t
   getStages()
   {
      return Stages;
   }

   static constexpr bool
   isAdaptive()
   {
      return false;
   }

   static constexpr ValueType
   getCoefficient( const std::size_t stage, const std::size_t i )
   {
      return k_coefficients[ stage ][ i ];
   }

   static constexpr ValueType
   getTimeCoefficient( std::size_t i )
   {
      return time_coefficients[ i ];
   }

   static constexpr ValueType
   getUpdateCoefficient( std::size_t i )
   {
      return update_coefficients[ i ];
   }

protected:
   static constexpr std::size_t Stages = 3;

   // clang-format off
   static constexpr std::array< std::array< Value, Stages>, Stages > k_coefficients {
      std::array< Value, Stages >{     0.0,     0.0, 0.0 },
      std::array< Value, Stages >{     1.0,     0.0, 0.0 },
      std::array< Value, Stages >{ 1.0/4.0, 1.0/4.0, 0.0 }
   };

   static constexpr std::array< Value, Stages > time_coefficients { 0.0, 1.0, 1.0/2.0 };

   static constexpr std::array< Value, Stages > update_coefficients { 1.0/6.0, 1.0/6.0, 2.0/3.0 };
   // clang-format on
};

}  // namespace TNL::Solvers::ODE::Methods
