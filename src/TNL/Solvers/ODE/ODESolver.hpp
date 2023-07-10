// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Math.h>
#include <TNL/Config/ParameterContainer.h>
#include <TNL/Containers/StaticArray.h>
#include <TNL/Containers/Expressions/LinearCombination.h>

#include "ODESolver.h"

namespace TNL::Solvers::ODE {

// std::integral_constant is used due to nvcc. In version 12.2, it does not
// allow  partial specialization with nontype template parameters.

template< typename Method,
          size_t Stage >
struct CoefficientsExtractor
{
   using ValueType = typename Method::ValueType;

   static constexpr size_t getSize() {
      return Method::getStages();
   }

   static constexpr ValueType getValue( size_t i ) {
      return Method::getCoefficient( Stage, i );
   }
};

template< typename Method >
struct UpdateCoefficientsExtractor
{
   static constexpr size_t getSize() {
      return Method::getStages();
   }

   using ValueType = typename Method::ValueType;

   static constexpr ValueType getValue( size_t i ) {
      return Method::getUpdateCoefficient( i );
   }
};

template< typename Method,
          typename Stage = std::integral_constant< size_t, 0 >,
          typename Stages = std::integral_constant< size_t, Method::getStages() > >
struct ODESolverEvaluator {

   template< typename VectorView, typename ConstVectorView, typename Value, typename RHSFunction >
   static void computeKVectors( std::array< VectorView, Stages::value >& k,
         const Value& time,
         const Value& currentTau,
         const ConstVectorView& u,
         VectorView aux,
         RHSFunction&& rhsFunction ) {
      if constexpr( Stage::value == 0 ) { // k1 = f( t, u )
         rhsFunction( time + Method::getTimeCoefficient( Stage::value ) * currentTau, currentTau, u, k[ Stage::value ] );
         ODESolverEvaluator< Method, std::integral_constant< size_t, Stage::value + 1 > >::computeKVectors( k, time, currentTau, u, aux, rhsFunction );
      } else {
         using Coefficients = CoefficientsExtractor< Method, Stage::value >;
         using Formula = Containers::Expressions::LinearCombination< Coefficients, VectorView >;
         aux = u + currentTau * Formula::evaluateArray( k );
         rhsFunction( time + Method::getTimeCoefficient( Stage::value ) * currentTau, currentTau, aux, k[ Stage::value ] );
         ODESolverEvaluator< Method, std::integral_constant< size_t, Stage::value + 1 > >::computeKVectors( k, time, currentTau, u, aux, rhsFunction );
      }
   }
};

template< typename Method >
struct ODESolverEvaluator< Method, std::integral_constant< size_t, Method::getStages() >, std::integral_constant< size_t, Method::getStages() > >
{
   static constexpr size_t Stages = Method::getStages();

   template< typename VectorView, typename ConstVectorView, typename Value, typename RHSFunction >
   static void computeKVectors( std::array< VectorView, Stages >& k_vectors,
         const Value& time,
         const Value& currentTau,
         const ConstVectorView& u,
         const VectorView& aux,
         RHSFunction&& rhsFunction ) {}
};


template< typename Method, typename Vector, typename SolverMonitor >
void
ODESolver< Method, Vector, SolverMonitor >::configSetup( Config::ConfigDescription& config, const String& prefix )
{
   ExplicitSolver< RealType, IndexType >::configSetup( config, prefix );
   config.addEntry< double >( prefix + "merson-adaptivity",
                              "Time step adaptivity controlling coefficient (the smaller the more precise the computation is, "
                              "zero means no adaptivity).",
                              1.0e-4 );
}

template< typename Method, typename Vector, typename SolverMonitor >
bool
ODESolver< Method, Vector, SolverMonitor >::setup( const Config::ParameterContainer& parameters, const String& prefix )
{
   ExplicitSolver< Vector, SolverMonitor >::setup( parameters, prefix );
   if( parameters.checkParameter( prefix + "merson-adaptivity" ) )
      this->setAdaptivity( parameters.getParameter< double >( prefix + "merson-adaptivity" ) );
   return true;
}

template< typename Method, typename Vector, typename SolverMonitor >
Method&
ODESolver< Method, Vector, SolverMonitor >::getMethod() {
   return this->method;
}

template< typename Method, typename Vector, typename SolverMonitor >
const Method&
ODESolver< Method, Vector, SolverMonitor >::getMethod() const {
   return this->method;
}

template< typename Method, typename Vector, typename SolverMonitor >
template< typename RHSFunction >
bool
ODESolver< Method, Vector, SolverMonitor >::solve( VectorType& u, RHSFunction&& rhsFunction )
{
   if( this->getTau() == 0.0 ) {
      std::cerr << "The time step for the ODE solver is zero." << std::endl;
      return false;
   }

   using VectorView = typename Vector::ViewType;
   std::array< VectorView, Stages > k_views;

   /////
   // Setup the supporting vectors
   for( int i = 0; i < Stages; i++ ) {
       k_vectors[ i ].setLike( u );
       k_vectors[ i ] = 0;
       k_views[ i ].bind( k_vectors[ i ] );
   }
   kAux.setLike( u );
   kAux = 0;

   /////
   // Set necessary parameters
   RealType& time = this->time;
   RealType currentTau = min( this->getTau(), this->getMaxTau() );
   if( time + currentTau > this->getStopTime() )
      currentTau = this->getStopTime() - time;
   if( currentTau == 0.0 )
      return true;
   this->resetIterations();
   this->setResidue( this->getConvergenceResidue() + 1.0 );

   /////
   // Start the main loop
   while( this->checkNextIteration() ) {

      ODESolverEvaluator< Method >::computeKVectors( k_views, time, currentTau, u.getConstView(), kAux.getView(), rhsFunction );

      /////
      // Compute an error of the approximation.
      RealType error( 0.0 );
      if constexpr( Method::isAdaptive() )
         error = Method::getError( k_views, currentTau );

      VectorType update( u );
      if( method.acceptStep( error ) ) {
         RealType lastResidue = this->getResidue();

         using UpdateCoefficients = UpdateCoefficientsExtractor< Method >;
         using UpdateExpression = Containers::Expressions::LinearCombination< UpdateCoefficients, Vector >;
         this->setResidue(
            addAndReduceAbs( u, currentTau * UpdateExpression::evaluateArray( k_vectors ), TNL::Plus{}, 0.0 ) /
            ( currentTau * (RealType) u.getSize() ) );
         time += currentTau;
         update = currentTau * UpdateExpression::evaluateArray( k_vectors );

         /////
         // When time is close to stopTime the new residue
         // may be inaccurate significantly.
         if( abs( time - this->stopTime ) < 1.0e-7 )
            this->setResidue( lastResidue );

         if( ! this->nextIteration() )
            return false;
      }

      /////
      // Compute the new time step.
      currentTau = min( method.computeTau( error, currentTau ), this->getMaxTau() );
      if( time + currentTau > this->getStopTime() )
         currentTau = this->getStopTime() - time;  // we don't want to keep such tau
      else
         this->tau = currentTau;

      //std::cout << "time = " << time << ", tau = " << currentTau << ", residue = " << this->getResidue() << " u = " << u << std::endl;

      /////
      // Check stop conditions.
      if( time >= this->getStopTime()
          || ( this->getConvergenceResidue() != 0.0 && this->getResidue() < this->getConvergenceResidue() ) )
         return true;
   }
   return this->checkConvergence();
}

}  // namespace TNL::Solvers::ODE
