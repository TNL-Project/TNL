// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstddef>
#include <TNL/Math.h>

namespace TNL::Solvers::ODE::Methods {

/**
 * \brief Second order [Heun's](https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods) method and Heun-Euler method
 * with adaptive time step.
 *
 * \tparam Value is arithmetic type used for computations.
 */
template< typename Value = double >
struct Heun2
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
      return true;
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
      return higher_order_update_coefficients[ i ];
   }

   static constexpr ValueType
   getErrorCoefficient( std::size_t i )
   {
      return higher_order_update_coefficients[ i ] - lower_order_update_coefficients[ i ];
   }

protected:
   static constexpr std::size_t Stages = 2;

   // clang-format off
   static constexpr std::array< std::array< Value, Stages>, Stages > k_coefficients {
      std::array< Value, Stages >{ 0.0, 0.0 },
      std::array< Value, Stages >{ 1.0, 0.0 }
   };

   static constexpr std::array< Value, Stages > time_coefficients{ 0.0, 1.0 };

   static constexpr std::array< Value, Stages > higher_order_update_coefficients{ 1.0/2.0, 1.0/2.0 };
   static constexpr std::array< Value, Stages > lower_order_update_coefficients { 1.0,     0.0 };
   // clang-format on
};

}  // namespace TNL::Solvers::ODE::Methods
