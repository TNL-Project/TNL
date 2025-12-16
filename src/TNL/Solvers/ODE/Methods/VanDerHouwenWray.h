// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstddef>
#include <TNL/Math.h>

namespace TNL::Solvers::ODE::Methods {

/**
 * \brief Third order [Van der Houwen's-Wray's](https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods) method.
 *
 * \tparam Value is arithmetic type used for computations.
 */
template< typename Value = double >
struct VanDerHouwenWray
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
      // clang-format off
      constexpr std::array< std::array< Value, Stages>, Stages > k_coefficients {
         std::array< Value, Stages >{      0.0,      0.0, 0.0 },
         std::array< Value, Stages >{ 8.0/15.0,      0.0, 0.0 },
         std::array< Value, Stages >{ 1.0/ 4.0, 5.0/12.0, 0.0 }
      };
      // clang-format on
      return k_coefficients[ stage ][ i ];
   }

   static constexpr ValueType
   getTimeCoefficient( std::size_t i )
   {
      constexpr std::array< Value, Stages > time_coefficients{ 0.0, 8.0 / 15.0, 2.0 / 3.0 };
      return time_coefficients[ i ];
   }

   static constexpr ValueType
   getUpdateCoefficient( std::size_t i )
   {
      constexpr std::array< Value, Stages > update_coefficients{ 1.0 / 4.0, 0.0, 3.0 / 4.0 };
      return update_coefficients[ i ];
   }

protected:
   static constexpr std::size_t Stages = 3;
};

}  // namespace TNL::Solvers::ODE::Methods
