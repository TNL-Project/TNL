// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstddef>
#include <TNL/Math.h>

namespace TNL::Solvers::ODE::Methods {

/**
 * \brief Third order [Bogacki-Shampin](https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods) method with adaptive
 * time step.
 *
 * \tparam Value is arithmetic type used for computations.
 */
template< typename Value = double >
struct BogackiShampin
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
      // clang-format off
      constexpr std::array< std::array< Value, Stages>, Stages > k_coefficients {
         std::array< Value, Stages >{     0.0,     0.0, 0.0,     0.0 },
         std::array< Value, Stages >{ 1.0/2.0,     0.0, 0.0,     0.0 },
         std::array< Value, Stages >{     0.0, 3.0/4.0, 0.0,     0.0 },
         std::array< Value, Stages >{ 2.0/9.0, 1.0/3.0, 4.0/9.0, 0.0 }
      };
      // clang-format on
      return k_coefficients[ stage ][ i ];
   }

   static constexpr ValueType
   getTimeCoefficient( std::size_t i )
   {
      constexpr std::array< Value, Stages > time_coefficients{ 0.0, 1.0 / 2.0, 3.0 / 4.0, 1.0 };
      return time_coefficients[ i ];
   }

   static constexpr ValueType
   getUpdateCoefficient( std::size_t i )
   {
      constexpr std::array< Value, Stages > higher_order_update_coefficients{ 7.0 / 24.0, 1.0 / 4.0, 1.0 / 3.0, 1.0 / 8.0 };
      return higher_order_update_coefficients[ i ];
   }

   static constexpr ValueType
   getErrorCoefficient( std::size_t i )
   {
      constexpr std::array< Value, Stages > lower_order_update_coefficients{ 2.0 / 9.0, 1.0 / 3.0, 4.0 / 9.0, 0.0 };
      return getUpdateCoefficient( i ) - lower_order_update_coefficients[ i ];
   }

protected:
   static constexpr std::size_t Stages = 4;
};

}  // namespace TNL::Solvers::ODE::Methods
