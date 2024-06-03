// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstddef>
#include <TNL/Math.h>

namespace TNL::Solvers::ODE::Methods {

/**
 * \brief Fifth order [Cash-Karp](https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods) method with adaptive time
 * step.
 *
 * \tparam Value is arithmetic type used for computations.
 */
template< typename Value = double >
struct CashKarp
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
   static constexpr std::size_t Stages = 6;

   // clang-format off
   static constexpr std::array< std::array< Value, Stages>, Stages > k_coefficients {
      std::array< Value, Stages >{     0.0,            0.0,        0.0,    0.0,     0.0,            0.0        },
      std::array< Value, Stages >{     1.0/    5.0,    0.0,        0.0,    0.0,     0.0,            0.0        },
      std::array< Value, Stages >{     3.0/   40.0,   9.0/ 40.0,   0.0,    0.0,     0.0,            0.0        },
      std::array< Value, Stages >{     3.0/   10.0,  -9.0/ 10.0,   6.0/    5.0,     0.0,            0.0        },
      std::array< Value, Stages >{   -11.0/   54.0,   5.0/  2.0, -70.0/   27.0,    35.0/    27.0,   0.0        },
      std::array< Value, Stages >{  1631.0/55296.0, 175.0/512.0, 575.0/13824.0, 44275.0/110592.0, 253.0/4096.0 }
   };

   static constexpr std::array< Value, Stages > time_coefficients { 0.0, 1.0/5.0, 3.0/10.0, 3.0/5.0, 1.0, 7.0/8.0 };

   static constexpr std::array< Value, Stages > higher_order_update_coefficients{ 37.0/378.0,     0.0, 250.0/621.0,     125.0/594.0,     0.0,           512.0/1771.0 };
   static constexpr std::array< Value, Stages > lower_order_update_coefficients { 2825.0/27648.0, 0.0, 18575.0/48384.0, 13525.0/55296.0, 277.0/14336.0, 1.0/4.0 };
   // clang-format on
};

}  // namespace TNL::Solvers::ODE::Methods
