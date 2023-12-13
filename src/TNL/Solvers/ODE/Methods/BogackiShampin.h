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
 * \brief Third order [Bogacki-Shampin](https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods) method with adaptive
 * time step.
 *
 * \tparam Value is arithmetic type used for computations.
 */
template< typename Value = double >
struct BogackiShampin
{
   using ValueType = Value;

   static constexpr size_t Stages = 4;

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
      return higher_order_update_coefficients[ i ];
   }

   static constexpr ValueType
   getErrorCoefficient( size_t i )
   {
      return higher_order_update_coefficients[ i ] - lower_order_update_coefficients[ i ];
   }

protected:
   // clang-format off
   static constexpr std::array< std::array< Value, Stages>, Stages > k_coefficients {
      std::array< Value, Stages >{     0.0,     0.0, 0.0,     0.0 },
      std::array< Value, Stages >{ 1.0/2.0,     0.0, 0.0,     0.0 },
      std::array< Value, Stages >{     0.0, 3.0/4.0, 0.0,     0.0 },
      std::array< Value, Stages >{ 2.0/9.0, 1.0/3.0, 4.0/9.0, 0.0 }
   };

   static constexpr std::array< Value, Stages > time_coefficients { 0.0, 1.0/2.0, 3.0/4.0, 1.0 };

   static constexpr std::array< Value, Stages > higher_order_update_coefficients{ 7.0/24.0, 1.0/4.0, 1.0/3.0, 1.0/8.0 };
   static constexpr std::array< Value, Stages > lower_order_update_coefficients { 2.0/9.0,  1.0/3.0, 4.0/9.0, 0.0 };
   // clang-format on
};

}  // namespace TNL::Solvers::ODE::Methods
