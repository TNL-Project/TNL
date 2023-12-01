// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstddef>
#include <TNL/Cuda/CudaCallable.h>
#include <TNL/Math.h>

namespace TNL::Solvers::ODE::Methods {

/**
 * \brief Fifth order [Runge-Kutta-Fehlberg](https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods) method with adaptive time step.
 *
 * \tparam Value is arithmetic type used for computations.
 */
template< typename Value = double >
struct Fehlberg5
{
   using ValueType = Value;

   static constexpr size_t Stages = 6;

   static constexpr size_t getStages() { return Stages; }

   static constexpr bool isAdaptive() { return true; }

   static constexpr ValueType getCoefficient( const size_t stage, const size_t i ) {
      return k_coefficients[ stage ][ i ];
   }

   static constexpr ValueType getTimeCoefficient( size_t i ) {
      return time_coefficients[ i ];
   }

   static constexpr ValueType getUpdateCoefficient( size_t i ) {
      return higher_order_update_coefficients[ i ];
   }

   static constexpr ValueType getErrorCoefficient( size_t i ) {
      return higher_order_update_coefficients[ i ] - lower_order_update_coefficients[ i ];
   }

protected:

   static constexpr std::array< std::array< Value, Stages>, Stages > k_coefficients {
      std::array< Value, Stages >{    0.0,            0.0,           0.0,            0.0,           0.0,        0.0 },
      std::array< Value, Stages >{    1.0/   4.0,     0.0,           0.0,            0.0,           0.0,        0.0 },
      std::array< Value, Stages >{    3.0/  32.0,     9.0/  32.0,    0.0,            0.0,           0.0,        0.0 },
      std::array< Value, Stages >{ 1932.0/2197.0, -7200.0/2197.0, 7296.0/2197.0,     0.0,           0.0,        0.0 },
      std::array< Value, Stages >{  439.0/ 216.0,    -8.0,        3680.0/ 513.0,  -845.0/4104.0,    0.0,        0.0 },
      std::array< Value, Stages >{   -8.0/  27.0,     2.0,        -3544.0/2565.0, 1859.0/4104.0,  -11.0/  40.0, 0.0 }
   };

   static constexpr std::array< Value, Stages > time_coefficients { 0.0, 1.0/4.0, 3.0/8.0, 12.0/13.0, 1.0, 1.0/2.0 };

   static constexpr std::array< Value, Stages > higher_order_update_coefficients{ 16.0/135.0, 0.0, 6656.0/12825.0, 28561.0/56430.0, -9.0/50.0, 2.0/55.0 };
   static constexpr std::array< Value, Stages > lower_order_update_coefficients { 25.0/216.0, 0.0, 1408.0/ 2565.0,  2197.0/ 4104.0, -1.0/5.0,  0.0 };
};

} // namespace TNL::Solvers::ODE::Methods
