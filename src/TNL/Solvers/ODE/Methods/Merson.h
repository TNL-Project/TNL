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
 * \brief Runge-Kutta-Merson method with adaptive step size.
 *
 * The method is described in as (see. https://encyclopediaofmath.org/wiki/Kutta-Merson_method)
 * \f[
 *  k1 = f( t, u )
 *  k2 = f( t+1/3*tau, u+tau * (  1/3*k1                           ) )
 *  k3 = f( t+1/3*tau, u+tau * (  1/6*k1 + 1/6*k2                  ) )
 *  k4 = f( t+1/2*tau, u+tau * (  1/8*k1 +         + 3/8*k3        ) )
 *  k5 = f( t+tau,     u+tau * (  1/2*k1           - 3/2*k3 + 2*k4 ) )
 * \f]
 *
 * \tparam Value is arithmetic type used for computations.
 */
template< typename Value = double >
struct Merson
{
   using ValueType = Value;

   static constexpr size_t Stages = 5;

   static constexpr size_t getStages() { return Stages; }

   static constexpr bool isAdaptive() { return true; }

   static constexpr ValueType getCoefficient( const size_t stage, const size_t i ) {
      return k_coefficients[ stage ][ i ];
   }

   static constexpr ValueType getTimeCoefficient( size_t i ) {
      return time_coefficients[ i ];
   }

   static constexpr ValueType getUpdateCoefficient( size_t i ) {
      return update_coefficients[ i ];
   }

   static constexpr ValueType getErrorCoefficient( size_t i ) {
      return error_coefficients[ i ];
   }

protected:

   static constexpr std::array< std::array< Value, Stages>, Stages > k_coefficients {
      std::array< Value, Stages >{     0.0,     0.0,   0.0, 0.0 },
      std::array< Value, Stages >{ 1.0/3.0,     0.0,   0.0, 0.0 },
      std::array< Value, Stages >{ 1.0/6.0, 1.0/6.0,   0.0, 0.0 },
      std::array< Value, Stages >{   0.125,     0.0, 0.375, 0.0 },
      std::array< Value, Stages >{     0.5,     0.0,  -1.5, 2.0 }
   };

   static constexpr std::array< Value, Stages > time_coefficients { 0.0, 1.0/3.0, 1.0/3.0, 0.5, 1.0 };

   static constexpr std::array< Value, Stages > update_coefficients { 1.0/6.0, 0.0, 0.0, 2.0/3.0, 1.0/6.0 };

   static constexpr std::array< Value, Stages > error_coefficients { 0.2/3.0, 0.0, -0.3, 0.8/3.0, -0.1/3.0 };
};

} // namespace TNL::Solvers::ODE::Methods
