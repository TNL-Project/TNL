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
 * \brief Fourth order [Runge-Kutta-Merson](https://encyclopediaofmath.org/wiki/Kutta-Merson_method) method with adaptive step
 * size.
 *
 * \tparam Value is arithmetic type used for computations.
 */
template< typename Value = double >
struct KuttaMerson
{
   using ValueType = Value;

   static constexpr size_t
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
   getCoefficient( const size_t stage, const size_t i )
   {
      return k_coefficients[ stage ][ i ];
   }

   static constexpr ValueType
   getTimeCoefficient( size_t i )
   {
      return time_coefficients[ i ];
   }

   static constexpr ValueType
   getUpdateCoefficient( size_t i )
   {
      return update_coefficients[ i ];
   }

   static constexpr ValueType
   getErrorCoefficient( size_t i )
   {
      return error_coefficients[ i ];
   }

protected:
   static constexpr size_t Stages = 5;

   // clang-format off
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
   // clang-format on
};

}  // namespace TNL::Solvers::ODE::Methods
