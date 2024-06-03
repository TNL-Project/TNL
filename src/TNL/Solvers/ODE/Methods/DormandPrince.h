// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstddef>
#include <TNL/Math.h>

namespace TNL::Solvers::ODE::Methods {

/**
 * \brief Fifth order [Dormand-Prince](https://en.wikipedia.org/wiki/Dormand%E2%80%93Prince_method) method also known as ode45
 * from [Matlab](https://www.mathworks.com/help/simulink/gui/solver.html) with adaptive step size.
 *
 * \tparam Value is arithmetic type used for computations.
 */
template< typename Value = double >
struct DormandPrince
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
   static constexpr std::size_t Stages = 7;

   // clang-format off
   static constexpr std::array< std::array< Value, Stages>, Stages > k_coefficients {
      std::array< Value, Stages >{           0.0,              0.0,            0.0,           0.0,             0.0,       0.0 },
      std::array< Value, Stages >{       1.0/5.0,              0.0,            0.0,           0.0,             0.0,       0.0 },
      std::array< Value, Stages >{       3.0/40.0,        9.0/40.0,            0.0,           0.0,             0.0,       0.0 },
      std::array< Value, Stages >{      44.0/45.0,      -56.0/15.0,       32.0/9.0,           0.0,             0.0,       0.0 },
      std::array< Value, Stages >{ 19372.0/6561.0,	-25360.0/2187.0, 64448.0/6561.0,  -212.0/729.0,             0.0,       0.0 },
      std::array< Value, Stages >{  9017.0/3168.0,	    -355.0/33.0, 46732.0/5247.0,    49.0/176.0, -5103.0/18656.0,       0.0 },
      std::array< Value, Stages >{     35.0/384.0,             0.0,   500.0/1113.0,   125.0/192.0,  -2187.0/6784.0, 11.0/84.0 }
   };

   static constexpr std::array< Value, Stages > time_coefficients{ 0.0, 1.0/5.0, 3.0/10.0, 4.0/5.0, 8.0/9.0, 1.0, 1.0 };

   static constexpr std::array< Value, Stages > higher_order_update_coefficients{ 35.0/384.0,     0.0, 500.0/1113.0,   125.0/192.0, -2187.0/6784.0,    11.0/84.0,    0.0 };
   static constexpr std::array< Value, Stages > lower_order_update_coefficients { 5179.0/57600.0, 0.0, 7571.0/16695.0, 393.0/640.0, -92097.0/339200.0, 187.0/2100.0,	1.0/40.0 };
   // clang-format on
};

}  // namespace TNL::Solvers::ODE::Methods
