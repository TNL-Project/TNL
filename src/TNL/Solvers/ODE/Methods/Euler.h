// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

namespace TNL::Solvers::ODE::Methods {

template< typename Value = double >
struct Euler
{
   using ValueType = Value;

   static constexpr size_t getStages() { return 1; }

   static constexpr bool isAdaptive() { return false; }

   /**
    * \brief Static method for setup of configuration parameters.
    *
    * \param config is the config description.
    * \param prefix is the prefix of the configuration parameters for this solver.
    */
   static void configSetup( Config::ConfigDescription& config, const String& prefix ) {}

   /**
    * \brief Method for setup of the explicit solver based on configuration parameters.
    *
    * \param parameters is the container for configuration parameters.
    * \param prefix is the prefix of the configuration parameters for this solver.
    * \return true if the parameters where parsed successfully.
    * \return false if the method did not succeed to read the configuration parameters.
    */
   bool setup( const Config::ParameterContainer& parameters, const String& prefix ) { return true; }

   static constexpr ValueType getCoefficients( const size_t stage, const size_t i ) { return 1; }

   static constexpr ValueType getTimeCoefficient( size_t i ) { return 0; }

   static constexpr ValueType getUpdateCoefficient( size_t i ) { return 1; }

   __cuda_callable__
   bool acceptStep( const ValueType& error ) { return true; }

   __cuda_callable__
   ValueType computeTau( const ValueType& error, const ValueType& currentTau ) { return currentTau; }
};

} // namespace TNL::Solvers::ODE::Methods
