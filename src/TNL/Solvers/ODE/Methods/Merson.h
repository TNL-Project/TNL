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

   /**
    * \brief Static method for setup of configuration parameters.
    *
    * \param config is the config description.
    * \param prefix is the prefix of the configuration parameters for this solver.
    */
   static void configSetup( Config::ConfigDescription& config, const String& prefix ) {
      config.addEntry< double >( prefix + "merson-adaptivity",
                                 "Time step adaptivity controlling coefficient (the smaller the more precise the computation is, "
                                 "zero means no adaptivity).",
                                 1.0e-4 );
   }

   /**
    * \brief Method for setup of the explicit solver based on configuration parameters.
    *
    * \param parameters is the container for configuration parameters.
    * \param prefix is the prefix of the configuration parameters for this solver.
    * \return true if the parameters where parsed successfully.
    * \return false if the method did not succeed to read the configuration parameters.
    */
   bool setup( const Config::ParameterContainer& parameters, const String& prefix ) {
      if( parameters.checkParameter( prefix + "merson-adaptivity" ) )
         this->setAdaptivity( parameters.getParameter< double >( prefix + "merson-adaptivity" ) );
      return true;
   }

   /**
    * \brief Setter of the parameter controlling the adaptive choice of the integration time step.
    *
    * The smaller the parameter is the smaller the integration time step tends to be.
    * Reasonable values for this parameters are approximately from interval \f$ [10^{-12},10^{-2}] \f$.
    * \param adaptivity new value of the parameter controlling the adaptive choice of
    *    integration time step.
    */
   __cuda_callable__
   void setAdaptivity( const ValueType& adaptivity ) {
      this->adaptivity = adaptivity;
   }

   /**
    * \brief Getter of the parameter controlling the adaptive choice of the integration time step.
    *
    * \returns the current value of the parameter controlling the adaptive choice of
    *    integration time step.
    */
   __cuda_callable__
   ValueType getAdaptivity() const {
      return adaptivity;
   }

   static constexpr ValueType getCoefficient( const size_t stage, const size_t i ) {
      return k_coefficients[ stage ][ i ];
   }

   static constexpr ValueType getTimeCoefficient( size_t i ) {
      return time_coefficients[ i ];
   }

   static constexpr ValueType getUpdateCoefficient( size_t i ) {
      return update_coefficients[ i ];
   }

   static constexpr ValueType getTimeCoefficient( size_t i ) {
      return error_coefficients[ i ];
   }

   template< typename Vector >
   static ValueType getError( const std::array< Vector, Stages >& k, const ValueType& tau ) {
      return max( tau / 3.0 * abs( 0.2 * k[ 0 ] - 0.9 * k[ 2 ] + 0.8 * k[ 3 ] - 0.1 * k[ 4 ] ) );
   }

   template< typename Vector >
   __cuda_callable__
   static ValueType getStaticError( const std::array< Vector, Stages >& k, const ValueType& tau ) {
      return max( tau / 3.0 * abs( 0.2 * k[ 0 ] - 0.9 * k[ 2 ] + 0.8 * k[ 3 ] - 0.1 * k[ 4 ] ) );
   }

   __cuda_callable__
   bool acceptStep( const ValueType& error ) {
      return this->adaptivity == 0.0 || error < this->adaptivity;
   }

   __cuda_callable__
   ValueType computeTau( const ValueType& error, const ValueType& currentTau ) {
      if( adaptivity != 0.0 && error != 0.0 )
         return currentTau * 0.8 * TNL::pow( adaptivity / error, 0.2 );
      return currentTau;
   }

protected:

   /****
    * Adaptivity controls the accuracy of the solver
    */
   ValueType adaptivity = 0.00001;

   static constexpr std::array< std::array< Value, Stages>, Stages > k_coefficients {
      std::array< Value, Stages >{     0.0,     0.0,   0.0, 0.0 },
      std::array< Value, Stages >{ 1.0/3.0,     0.0,   0.0, 0.0 },
      std::array< Value, Stages >{ 1.0/6.0, 1.0/6.0,   0.0, 0.0 },
      std::array< Value, Stages >{   0.125,     0.0, 0.375, 0.0 },
      std::array< Value, Stages >{     0.5,     0.0,  -1.5, 2.0 }
   };

   static constexpr std::array< Value, Stages > time_coefficients { 0.0, 1.0/3.0, 1.0/3.0, 0.5, 1.0 };

   static constexpr std::array< Value, Stages > update_coefficients { 1.0/6.0, 0.0, 0.0, 2.0/3.0, 1.0/6.0 };

   static constexpr std::array< Value, Stages > error_coefficients { 0.2, 0.0, -0.9, 0.8, -0.1 };
};

} // namespace TNL::Solvers::ODE::Methods
