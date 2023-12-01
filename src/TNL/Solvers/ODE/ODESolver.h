// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Containers/StaticArray.h>
#include <TNL/Config/ConfigDescription.h>
#include <TNL/Solvers/ODE/ExplicitSolver.h>
#include <TNL/Solvers/ODE/StaticExplicitSolver.h>
#include <TNL/TypeTraits.h>

namespace TNL::Solvers::ODE {

/**
 * \brief Integrator or solver of system of ordinary differential equations.
 *
 * This solver can be used for the numerical solution of
 * [ordinary differential equations](https://en.wikipedia.org/wiki/Ordinary_differential_equation) having the
 * following form:
 *
 * \f$ \frac{d \vec u}{dt} = \vec f( t, \vec u) \text{ on } (0,T) \f$
 *
 * \f$ \vec u( 0 )  = \vec u_{ini} \f$.
 * The unknown vector \f$ \vec x \in R^n \f$ can expressed by a \ref TNL::Containers::Vector or \ref TNL::Containers::StaticVector.
 * In the later case, the solver can be executed even within GPU kernels. The method which is supposed to be used
 * by the solver is represented by the template parameter \ref Method.
 *
 * The following example demonstrates the use the solver:
 *
 * \includelineno Solvers/ODE/ODESolver-HeatEquationExample.h
 *
 * \tparam Method is a method which is supposed to be used for the numerical integration.
 * \tparam Value is a vector (\ref TNL::Containers::Vector or \ref TNL::Containers::StaticVector) representing \f$ \vec x \in R^n \f$.
 */
template< typename Method,
          typename Vector,
          typename SolverMonitor = IterativeSolverMonitor< typename Vector::RealType, typename Vector::IndexType >,
          bool IsStatic = IsStaticArrayType< Vector >() >
struct ODESolver;

template< typename Method,
          typename Vector,
          typename SolverMonitor >
struct ODESolver< Method, Vector, SolverMonitor, true > :
   public StaticExplicitSolver< GetRealType< Vector >, GetIndexType < Vector > >
{
public:

   static constexpr int Stages = Method::getStages();
   /**
    * \brief Type of floating-point arithemtics.
    */
   using RealType = GetRealType< Vector >;

   using VectorType = Vector;

   using  ValueType = typename VectorType::ValueType;

   /**
    * \brief Type for indexing.
    */
   using IndexType = GetIndexType< Vector >;

   static constexpr bool isStatic() { return true; }

   /**
    * \brief Type of object used for monitoring the convergence.
    *
    * Can be \ref TNL::Solvers::IterativeSolverMonitor.
    */
   using SolverMonitorType = SolverMonitor;

   /**
    * \brief Default constructor.
    */
   __cuda_callable__ ODESolver();

   __cuda_callable__ ODESolver( const ODESolver& solver );

   /**
    * \brief Static method for setup of configuration parameters.
    *
    * \param config is the config description.
    * \param prefix is the prefix of the configuration parameters for this solver.
    */
   static void
   configSetup( Config::ConfigDescription& config, const String& prefix = "" );

   /**
    * \brief Method for setup of the explicit solver based on configuration parameters.
    *
    * \param parameters is the container for configuration parameters.
    * \param prefix is the prefix of the configuration parameters for this solver.
    * \return true if the parameters where parsed successfully.
    * \return false if the method did not succeed to read the configuration parameters.
    */
   bool
   setup( const Config::ParameterContainer& parameters, const String& prefix = "" );

   /**
    * \brief Setter of the parameter controlling the adaptive choice of the integration time step.
    *
    * The smaller the parameter is the smaller the integration time step tends to be.
    * Reasonable values for this parameters are approximately from interval \f$ [10^{-12},10^{-2}] \f$.
    * \param adaptivity new value of the parameter controlling the adaptive choice of
    *    integration time step.
    */
   __cuda_callable__
   void setAdaptivity( const RealType& adaptivity ) {
      this->adaptivity = adaptivity;
   }

   /**
    * \brief Getter of the parameter controlling the adaptive choice of the integration time step.
    *
    * \returns the current value of the parameter controlling the adaptive choice of
    *    integration time step.
    */
   __cuda_callable__
   RealType getAdaptivity() const {
      return adaptivity;
   }

   /**
    * \brief Gives reference to the underlying method.
    *
    * \return reference to the underlying method.
    */
   __cuda_callable__
   Method& getMethod();

   /**
    * \brief Gives constant reference to the underlying method.
    *
    * \return constant reference to the underlying method.
    */
   __cuda_callable__
   const Method& getMethod() const;

   /**
    * \brief Solve ODE given by a lambda function.
    *
    * \tparam RHSFunction is type of a lambda function representing the right-hand side of the ODE system.
    *    The definition of the lambda function reads as:
    * ```
    * auto f = [=] ( const Real& t, const Real& tau, const VectorType& u, VectorType& fu ) {...}
    * ```
    * where `t` is the current time of the evolution, `tau` is the current time step, `u` is the solution at the current time,
    * `fu` is variable/static vector into which the lambda function is suppsed to evaluate the function \f$ f(t, \vec x) \f$ at
    * the current time \f$ t \f$.
    * \param u is a variable/static vector representing the solution of the ODE system at current time.
    * \param f is the lambda function representing the right-hand side of the ODE system.
    * \return `true` if steady state solution has been reached, `false` otherwise.
    */
   template< typename RHSFunction >
   __cuda_callable__ bool
   solve( VectorType& u, RHSFunction&& f );

protected:

   /****
    * Adaptivity controls the accuracy of the solver
    */
   RealType adaptivity = 0.00001;

   std::array< VectorType, Stages > k_vectors;

   VectorType kAux;

   Method method;
};

template< typename Method,
          typename Vector,
          typename SolverMonitor >
struct ODESolver< Method, Vector, SolverMonitor, false > :
   public ExplicitSolver< typename Vector::RealType, typename Vector::IndexType, SolverMonitor >
{
public:

   static constexpr int Stages = Method::getStages();
   /**
    * \brief Type of floating-point arithemtics.
    */
   using RealType = GetRealType< Vector >;

   using ValueType = typename Vector::ValueType;

   /**
    * \brief Device where the solver is supposed to be executed.
    */
   using DeviceType = typename Vector::DeviceType;

   /**
    * \brief Type for indexing.
    */
   using IndexType = typename Vector::IndexType;

   /**
    * \brief Type of unknown variable \f$ \vec x \f$.
    */
   using VectorType = Vector;

   /**
    * \brief Alias for type of unknown variable \f$ \vec x \f$.
    *
    * Note, \e VectorType can be \ref TNL::Containers::VectorView but
    * \e DofVectorType is always \ref TNL::Containers::Vector.
    */
   using DofVectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;

   /**
    * \brief Type of object used for monitoring the convergence.
    *
    * Can be \ref TNL::Solvers::IterativeSolverMonitor.
    */
   using SolverMonitorType = SolverMonitor;

   static constexpr bool isStatic() { return false; }

   /**
    * \brief Default constructor.
    */
   ODESolver();

   /**
    * \brief Static method for setup of configuration parameters.
    *
    * \param config is the config description.
    * \param prefix is the prefix of the configuration parameters for this solver.
    */
   static void
   configSetup( Config::ConfigDescription& config, const String& prefix = "" );

   /**
    * \brief Method for setup of the explicit solver based on configuration parameters.
    *
    * \param parameters is the container for configuration parameters.
    * \param prefix is the prefix of the configuration parameters for this solver.
    * \return true if the parameters where parsed successfully.
    * \return false if the method did not succeed to read the configuration parameters.
    */
   bool
   setup( const Config::ParameterContainer& parameters, const String& prefix = "" );

   /**
    * \brief Setter of the parameter controlling the adaptive choice of the integration time step.
    *
    * The smaller the parameter is the smaller the integration time step tends to be.
    * Reasonable values for this parameters are approximately from interval \f$ [10^{-12},10^{-2}] \f$.
    * \param adaptivity new value of the parameter controlling the adaptive choice of
    *    integration time step.
    */
   __cuda_callable__
   void setAdaptivity( const RealType& adaptivity ) {
      this->adaptivity = adaptivity;
   }

   /**
    * \brief Getter of the parameter controlling the adaptive choice of the integration time step.
    *
    * \returns the current value of the parameter controlling the adaptive choice of
    *    integration time step.
    */
   __cuda_callable__
   RealType getAdaptivity() const {
      return adaptivity;
   }

   /**
    * \brief Gives reference to the underlying method.
    *
    * \return reference to the underlying method.
    */
   Method& getMethod();

   /**
    * \brief Gives constant reference to the underlying method.
    *
    * \return constant reference to the underlying method.
    */
   const Method& getMethod() const;

   /**
    * \brief Solve ODE given by a lambda function.
    *
    * \tparam RHSFunction is type of a lambda function representing the right-hand side of the ODE system.
    *    The definition of the lambda function reads as:
    * ```
    * auto f = [=] ( const Real& t, const Real& tau, const VectorType& u, VectorType& fu ) {...}
    * ```
    * where `t` is the current time of the evolution, `tau` is the current time step, `u` is the solution at the current time,
    * `fu` is variable/static vector into which the lambda function is suppsed to evaluate the function \f$ f(t, \vec x) \f$ at
    * the current time \f$ t \f$.
    * \param u is a variable/static vector representing the solution of the ODE system at current time.
    * \param f is the lambda function representing the right-hand side of the ODE system.
    * \return `true` if steady state solution has been reached, `false` otherwise.
    */
   template< typename RHSFunction >
   bool
   solve( VectorType& u, RHSFunction&& f );

protected:

   /****
    * Adaptivity controls the accuracy of the solver
    */
   RealType adaptivity = 0.00001;

   std::array< VectorType, Stages > k_vectors;

   VectorType kAux;

   Method method;

};

}  // namespace TNL::Solvers::ODE

#include <TNL/Solvers/ODE/ODESolver.hpp>
