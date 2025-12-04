// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Containers/DistributedVector.h>
#include <TNL/Containers/StaticArray.h>
#include <TNL/Config/ConfigDescription.h>
#include <TNL/Solvers/ODE/ExplicitSolver.h>
#include <TNL/Solvers/ODE/StaticExplicitSolver.h>
#include <TNL/TypeTraits.h>

/***
 * \brief Namespace for solvers of ordinary differential equations.
 */
namespace TNL::Solvers::ODE {

/**
 * \brief Integrator or solver of systems of ordinary differential equations.
 *
 * This solver can be used for the numerical solution of
 * [ordinary differential equations](https://en.wikipedia.org/wiki/Ordinary_differential_equation) having the
 * following form:
 *
 * \f$ \frac{d \vec u}{dt} = \vec f( t, \vec u) \text{ on } (0,T), \f$
 *
 * and the initial condition
 *
 * \f$ \vec u( 0 )  = \vec u_{ini} \f$.
 *
 * The vector \f$ \vec u(t) \f$ can be represented using different types of containers, depending on the size and
 * nature of the ODE system:
 *
 * 1. **Static vectors** (\ref TNL::Containers::StaticVector): This is suitable for small systems of ODEs with a fixed number of
 * unknowns. Utilizing `StaticVector` allows the ODE solver to be executed within GPU kernels. This capability is particularly
 * useful for scenarios like running multiple sequential solvers in parallel, as in the case of \ref
 * TNL::Algorithms::parallelFor.
 * 2. **Dynamic vectors** (\ref TNL::Containers::Vector or \ref TNL::Containers::VectorView): These are preferred when dealing
 * with large systems of ODEs, such as those arising in the solution of [parabolic partial differential
 * equations](https://en.wikipedia.org/wiki/Parabolic_partial_differential_equation) using the
 * [method of lines](https://en.wikipedia.org/wiki/Method_of_lines). In these instances, the solver typically handles a single,
 * large-scale problem that can be executed in parallel internally.
 *
 * The method, which is supposed to be used by the solver, is represented by the template parameter \e Method.
 *
 * The following examples demonstrates the use the solver with the static vector
 *
 * \includelineno Solvers/ODE/StaticODESolver-LorenzParallelExample.h
 *
 * and with the dynamic vector
 *
 * \includelineno Solvers/ODE/ODESolver-HeatEquationExample.h
 *
 * \tparam Method is a method (one from \ref TNL::Solvers::ODE namespace) which is supposed to be used
 * for the numerical integration. \tparam Vector is a vector (\ref TNL::Containers::Vector, \ref TNL::Containers::VectorView, or
 * \ref TNL::Containers::StaticVector) representing \f$ \vec x \in R^n \f$.
 */
template< typename Method,
          typename Vector,
          typename SolverMonitor = IterativeSolverMonitor< typename Vector::RealType >,
          bool IsStatic = IsStaticArrayType< Vector >() >
struct ODESolver;

template< typename Method, typename Vector, typename SolverMonitor >
struct ODESolver< Method, Vector, SolverMonitor, true > : public StaticExplicitSolver< GetValueType_t< Vector >, std::size_t >
{
public:
   static constexpr int Stages = Method::getStages();
   /**
    * \brief Type of floating-point arithemtics.
    */
   using RealType = GetValueType_t< Vector >;

   using VectorType = Vector;

   using ValueType = typename VectorType::ValueType;

   /**
    * \brief Type for indexing.
    */
   using IndexType = std::size_t;

   [[nodiscard]] static constexpr bool
   isStatic()
   {
      return true;
   }

   /**
    * \brief Type of object used for monitoring the convergence.
    *
    * Can be \ref TNL::Solvers::IterativeSolverMonitor.
    */
   using SolverMonitorType = SolverMonitor;

   /**
    * \brief Default constructor.
    */
   __cuda_callable__
   ODESolver();

   __cuda_callable__
   ODESolver( const ODESolver& solver );

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
   void
   setAdaptivity( const RealType& adaptivity )
   {
      this->adaptivity = adaptivity;
   }

   /**
    * \brief Getter of the parameter controlling the adaptive choice of the integration time step.
    *
    * \returns the current value of the parameter controlling the adaptive choice of
    *    integration time step.
    */
   [[nodiscard]] __cuda_callable__
   RealType
   getAdaptivity() const
   {
      return adaptivity;
   }

   /**
    * \brief Gives reference to the underlying method.
    *
    * \return reference to the underlying method.
    */
   [[nodiscard]] __cuda_callable__
   Method&
   getMethod();

   /**
    * \brief Gives constant reference to the underlying method.
    *
    * \return constant reference to the underlying method.
    */
   [[nodiscard]] __cuda_callable__
   const Method&
   getMethod() const;

   /**
    * \brief Solve ODE given by a lambda function.
    *
    * \tparam RHSFunction is type of a lambda function representing the right-hand side of the ODE system.
    *    The definition of the lambda function reads as:
    * ```
    * auto f = [=] ( const Real& t, const Real& tau, const VectorType& u, VectorType& fu ) {...}
    * ```
    * where `t` is the current time of the evolution, `tau` is the current time step, `u` is the solution at the current time,
    * `fu` is variable/static vector into which the lambda function is supposed to evaluate the function \f$ f(t, \vec x) \f$ at
    * the current time \f$ t \f$.
    * \param u is a variable/static vector representing the solution of the ODE system at current time.
    * \param rhsFunction is the lambda function representing the right-hand side of the ODE system.
    * \param params are the parameters which are supposed to be passed to the lambda function \e f. This is due to the fact that
    * the CUDA compiler does not allow nested lambda functions: "An extended __host__ __device__ lambda cannot be defined inside
    * an extended __host__ __device__  lambda expression".
    * \return `true` if steady state solution has been reached, `false` otherwise.
    *
    * \par Example
    * \include Solvers/ODE/StaticODESolver-SineExample.h
    *
    * \include Solvers/ODE/StaticODESolver-LorenzParallelExample.h.
    */
   template< typename RHSFunction, typename... Params >
   __cuda_callable__
   bool
   solve( VectorType& u, RHSFunction&& rhsFunction, Params&&... params );

   /**
    * \brief Setup auxiliary vectors of the solver.
    *
    * This method is supposed to be called before the first call of the method \ref iterate. It is
    * not necessary to call this method before the method \ref solve is used.
    *
    * \param u this parameter is only for consistency with the ODE solver for dynamic vectors.
    */
   void __cuda_callable__
   init( const VectorType& u );

   /**
    * \brief Performs one iteration of the solver.
    *
    * This method can be used for hybrid solvers which combine various ODE solvers. Otherwise, use of \ref solve
    * is recommended. Before the first call of this method, the method \ref init has to be called.
    *
    * \tparam RHSFunction is type of a lambda function representing the right-hand side of the ODE system.
    * \tparam Params are the parameters which are supposed to be passed to the lambda function \e f.
    * \param u is a variable/static vector representing the solution of the ODE system at current time.
    * \param time is the current time of the evolution. The variable is increased by \e tau.
    * \param currentTau is the current time step. It can be changed by the solver if the adaptive time step control is used.
    * \param rhsFunction is the lambda function representing the right-hand side of the ODE system.  The definition of the
    * lambda function is the same as in the method \ref solve.
    * \param params are the parameters which are supposed to be passed to the lambda function \e f.
    *
    * \par Example
    * \include Solvers/ODE/StaticODESolver-SineExample_iterate.h
    */
   template< typename RHSFunction, typename... Params >
   void __cuda_callable__
   iterate( VectorType& u, RealType& time, RealType& currentTau, RHSFunction&& rhsFunction, Params&&... params );

   /**
    * \brief This method is just for consistency with the ODE solver for dynamic vectors.
    */
   void __cuda_callable__
   reset();

protected:
   /****
    * Adaptivity controls the accuracy of the solver
    */
   RealType adaptivity = 0.00001;

   std::array< VectorType, Stages > k_vectors;

   VectorType kAux;

   Method method;
};

template< typename Method, typename Vector, typename SolverMonitor >
struct ODESolver< Method, Vector, SolverMonitor, false >
: public ExplicitSolver< typename Vector::RealType, typename Vector::IndexType, SolverMonitor >
{
public:
   static constexpr int Stages = Method::getStages();
   /**
    * \brief Type of floating-point arithemtics.
    */
   using RealType = GetValueType_t< Vector >;

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
   using DofVectorType = std::conditional_t<  //
      HasGetCommunicatorMethod< Vector >::value,
      TNL::Containers::DistributedVector< RealType, DeviceType, IndexType >,
      TNL::Containers::Vector< RealType, DeviceType, IndexType > >;

   /**
    * \brief Type of object used for monitoring the convergence.
    *
    * Can be \ref TNL::Solvers::IterativeSolverMonitor.
    */
   using SolverMonitorType = SolverMonitor;

   [[nodiscard]] static constexpr bool
   isStatic()
   {
      return false;
   }

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
   [[nodiscard]] bool
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
   void
   setAdaptivity( const RealType& adaptivity )
   {
      this->adaptivity = adaptivity;
   }

   /**
    * \brief Getter of the parameter controlling the adaptive choice of the integration time step.
    *
    * \returns the current value of the parameter controlling the adaptive choice of
    *    integration time step.
    */
   [[nodiscard]] __cuda_callable__
   RealType
   getAdaptivity() const
   {
      return adaptivity;
   }

   /**
    * \brief Gives reference to the underlying method.
    *
    * \return reference to the underlying method.
    */
   [[nodiscard]] Method&
   getMethod();

   /**
    * \brief Gives constant reference to the underlying method.
    *
    * \return constant reference to the underlying method.
    */
   [[nodiscard]] const Method&
   getMethod() const;

   /**
    * \brief Solve ODE given by a lambda function.
    *
    * \tparam RHSFunction is type of a lambda function representing the right-hand side of the ODE system.
    *    The definition of the lambda function reads as:
    * ```
    * auto f = [=] ( const Real& t, const Real& tau, const VectorType& u, VectorType& fu ) {...}
    * ```
    * where `t` is the current time of the evolution, `tau` is the current time step, `u` is the solution at the current time,
    * `fu` is variable/static vector into which the lambda function is supposed to evaluate the function \f$ f(t, \vec x) \f$ at
    * the current time \f$ t \f$.
    * \param u is a variable/static vector representing the solution of the ODE system at current time.
    * \param rhsFunction is the lambda function representing the right-hand side of the ODE system.
    * \param params are the parameters which are supposed to be passed to the lambda function \e f. This is due to the fact that
    * the CUDA compiler does not allow nested lambda functions: "An extended __host__ __device__ lambda cannot be defined inside
    * an extended __host__ __device__  lambda expression".
    * \return `true` if steady state solution has been reached, `false` otherwise.
    *
    * \par Example
    * \include Solvers/ODE/ODESolver-HeatEquationWithMonitorExample.h
    */
   template< typename RHSFunction, typename... Params >
   bool
   solve( VectorType& u, RHSFunction&& rhsFunction, Params&&... params );

   /**
    * \brief Setup auxiliary vectors of the solver.
    *
    * This method is supposed to be called before the first call of the method \ref iterate. It is
    * not necessary to call this method before the method \ref solve is used. Also this methods
    * needs to be called everytime the size of \e u changes.
    *
    * \param u is a variable/dynamic vector representing the solution of the ODE system at current time.
    */
   void
   init( const VectorType& u );

   /**
    * \brief Performs one iteration of the solver.
    *
    * This method can be used for hybrid solvers which combine various ODE solvers. Otherwise, use of \ref solve
    * is recommended. Before the first call of this method, the method \ref init has to be called.
    *
    * \tparam RHSFunction is type of a lambda function representing the right-hand side of the ODE system.
    * \tparam Params are the parameters which are supposed to be passed to the lambda function \e f.
    * \param u is a variable/static vector representing the solution of the ODE system at current time.
    * \param time is the current time of the evolution. The variable is increased by \e tau.
    * \param currentTau is the current time step. It can be changed by the solver if the adaptive time step control is used.
    * \param rhsFunction is the lambda function representing the right-hand side of the ODE system. The definition of the lambda
    * function is the same as in the method \ref solve.
    * \param params are the parameters which are supposed to be passed to the lambda function \e f.
    *
    * \par Example
    * \include Solvers/ODE/StaticODESolver-SineExample_iterate.h
    */
   template< typename RHSFunction, typename... Params >
   void
   iterate( VectorType& u, RealType& time, RealType& currentTau, RHSFunction&& rhsFunction, Params&&... params );

   /**
    * \brief Resets the solver.
    *
    * This method frees memory allocated by the solver. If it is called, the method \ref init has to be called before
    * the next call of the method \ref iterate.
    */
   void
   reset();

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
