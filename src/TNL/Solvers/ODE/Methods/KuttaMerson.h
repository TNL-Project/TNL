// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstddef>
#include <TNL/Math.h>

namespace TNL::Solvers::ODE::Methods {

/**
 * \brief Fourth order [Runge-Kutta-Merson](https://encyclopediaofmath.org/wiki/Kutta-Merson_method) method with adaptive step
 * size.
 *
 * \tparam Value is arithmetic type used for computations.
 */
template< typename Value = double >
struct KuttaMerson
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
      constexpr std::array< std::array< Value, Stages >, Stages > k_coefficients {
         std::array< Value, Stages >{     0.0,     0.0,   0.0, 0.0, 0.0 },
         std::array< Value, Stages >{ 1.0/3.0,     0.0,   0.0, 0.0, 0.0 },
         std::array< Value, Stages >{ 1.0/6.0, 1.0/6.0,   0.0, 0.0, 0.0 },
         std::array< Value, Stages >{   0.125,     0.0, 0.375, 0.0, 0.0 },
         std::array< Value, Stages >{     0.5,     0.0,  -1.5, 2.0, 0.0 }
      };
      // clang-format on
      return k_coefficients[ stage ][ i ];
   }

   static constexpr ValueType
   getTimeCoefficient( std::size_t i )
   {
      constexpr std::array< Value, Stages > time_coefficients{ 0.0, 1.0 / 3.0, 1.0 / 3.0, 0.5, 1.0 };
      return time_coefficients[ i ];
   }

   static constexpr ValueType
   getUpdateCoefficient( std::size_t i )
   {
      constexpr std::array< Value, Stages > update_coefficients{ 1.0 / 6.0, 0.0, 0.0, 2.0 / 3.0, 1.0 / 6.0 };
      return update_coefficients[ i ];
   }

   static constexpr ValueType
   getErrorCoefficient( std::size_t i )
   {
      constexpr std::array< Value, Stages > error_coefficients{ 0.2 / 3.0, 0.0, -0.3, 0.8 / 3.0, -0.1 / 3.0 };
      return error_coefficients[ i ];
   }

protected:
   static constexpr std::size_t Stages = 5;
};

}  // namespace TNL::Solvers::ODE::Methods
