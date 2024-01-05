// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

namespace TNL::Solvers::ODE::Methods {

/**
 * \brief First order [Euler](https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods) method.
 *
 * \tparam Value is arithmetic type used for computations.
 */
template< typename Value = double >
struct Euler
{
   using ValueType = Value;

   static constexpr std::size_t
   getStages()
   {
      return 1;
   }

   static constexpr bool
   isAdaptive()
   {
      return false;
   }

   static constexpr ValueType
   getCoefficients( const std::size_t stage, const std::size_t i )
   {
      return 1;
   }

   static constexpr ValueType
   getTimeCoefficient( std::size_t i )
   {
      return 0;
   }

   static constexpr ValueType
   getUpdateCoefficient( std::size_t i )
   {
      return 1;
   }
};

}  // namespace TNL::Solvers::ODE::Methods
