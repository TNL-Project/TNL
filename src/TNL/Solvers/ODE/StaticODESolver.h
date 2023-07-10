// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Config/ConfigDescription.h>
#include <TNL/Solvers/ODE/StaticExplicitSolver.h>

namespace TNL::Solvers::ODE {

/**
 * \brief Solver of ODEs with the first order of accuracy.
 *
 * This solver is based on the [Euler method](https://en.wikipedia.org/wiki/Euler_method) for solving of
 * [ordinary differential equations](https://en.wikipedia.org/wiki/Ordinary_differential_equation) having the
 * following form:
 *
 * \f$ \frac{d \vec u}{dt} = \vec f( t, \vec u) \text{ on } (0,T) \f$
 *
 * \f$ \vec u( 0 )  = \vec u_{ini} \f$.
 * It is supposed to be used when the unknown \f$ \vec x \in R^n \f$ is expressed by a \ref Containers::Vector.
 *
 * For problems where \f$ \vec x\f$ is represented by \ref TNL::Containers::StaticVector,
 * see \ref TNL::Solvers::ODE::StaticMerson<Containers::StaticVector<Size_,Real>>.
 * For problems where \f$ x\f$ is represented by floating-point number, see \ref TNL::Solvers::ODE::StaticMerson.
 *
 * The following example demonstrates the use the solvers:
 *
 * \includelineno Solvers/ODE/ODESolver-HeatEquationExample.h
 *
 * \tparam Vector is type of vector storing \f$ \vec x \in R^n \f$, mostly \ref TNL::Containers::Vector
 *    or \ref TNL::Containers::VectorView.
 */
template< typename Method, typename Vector >
class StaticODESolver : public StaticExplicitSolver< Vector >
{
public:

   static constexpr int Stages = Method::getStages();

   /**
    * \brief Type of floating-point arithmetics.
    */
   using ValueType = typename Vector::RealType;

   using RealType = ValueType;

   /**
    * \brief Type for indexing.
    */
   using IndexType = int;

   /**
    * \brief Type of unknown variable \f$ x \f$.
    */
   using VectorType = Vector;

   /**
    * \brief Alias for type of unknown variable \f$ x \f$.
    */
   using DofVectorType = VectorType;

   /**
    * \brief Default constructor.
    */
   __cuda_callable__
   StaticODESolver() = default;

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
   std::array< VectorType, Stages > k_vectors;

   VectorType kAux;

   Method method;
};

}  // namespace TNL::Solvers::ODE

#include <TNL/Solvers/ODE/StaticODESolver.hpp>
