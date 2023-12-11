// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

/**
 * \brief Namespace for numerical methods for ODE solvers.
 *
 * This namespace contains numerical methods for \ref TNL::Solvers::ODE::ODESolver.
 *
 * TNL provides several methods for ODE solution, categorized based on their order of accuracy:
 *
 * **1-order accuracy methods:**
 * 1. \ref TNL::Solvers::ODE::Methods::Euler or \ref TNL::Solvers::ODE::Methods::Matlab::ode1 - the [forward
 * Euler](https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods) method.
 * 2. \ref TNL::Solvers::ODE::Methods::Midpoint - the [explicit
midpoint](https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods) method.
 *
 * **2-nd order accuracy methods**
 * 1. \ref TNL::Solvers::ODE::Methods::Heun2 or \ref TNL::Solvers::ODE::Methods::Matlab::ode2 - the
 * [Heun](https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods) method with adaptive choice of the time step.
 * 2. \ref TNL::Solvers::ODE::Methods::Ralston2 - the
 * [Ralston](https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods) method.
 * 3. \ref TNL::Solvers::ODE::Methods::Fehlberg2 - the
 * [Fehlberg](https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods) method with adaptive choice of the time step.
 *
 * **3-rd order accuracy methods**
 * 1. \ref TNL::Solvers::ODE::Methods::Kutta - the [Kutta](https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods)
 * method.
 * 2. \ref TNL::Solvers::ODE::Methods::Heun3 - the [Heun](https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods)
 * method.
 * 3. \ref TNL::Solvers::ODE::Methods::VanDerHouwenWray - the
 * [Van der Houwen/Wray](https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods) method.
 * 4. \ref TNL::Solvers::ODE::Methods::Ralston3 - the
 * [Ralston](https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods) method.
 * 5. \ref TNL::Solvers::ODE::Methods::SSPRK3 - the
 * [Strong Stability Preserving Runge-Kutta](https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods) method.
 * 6. \ref TNL::Solvers::ODE::Methods::BogackiShampin or \ref TNL::Solvers::ODE::Methods::Matlab::ode23 -
 * [Bogacki-Shampin](https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods) method with adaptive choice of the time
 * step.
 *
 * **4-th order accuracy method**
 * 1. \ref TNL::Solvers::ODE::Methods::OriginalRungeKutta - the
 * ["original" Runge-Kutta](https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods) method.
 * 2. \ref TNL::Solvers::ODE::Methods::Rule38 - [3/8 rule](https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods)
 * method.
 * 3. \ref TNL::Solvers::ODE::Methods::Ralston4 - the
 * [Ralston](https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods) method.
 * 4. \ref TNL::Solvers::ODE::Methods::Merson - the
[Runge-Kutta-Merson](https://encyclopediaofmath.org/wiki/Kutta-Merson_method) * method with adaptive choice of the time step.

 * **5-th order accuracy method**
 * 1. \ref TNL::Solvers::ODE::Methods::CashKarp - the
 * [Cash-Karp](https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods) method with adaptive choice of the time step.
 * 2. \ref TNL::Solvers::ODE::Methods::DormandPrince or \ref TNL::Solvers::ODE::Methods::Matlab::ode45 - the
 * [Dormand-Prince](https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods) with adaptive choice of the time step.
 * 3. \ref TNL::Solvers::ODE::Methods::Fehlberg5 - the
 * [Fehlberg](https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods) method with adaptive choice of the time step.
 *
 * The vector \f$ \vec u(t) \f$ in ODE solvers can be represented using different types of containers, depending on the size and
nature of the ODE system:
 * 1. Static vectors (\ref TNL::Containers::StaticVector): This is suitable for small systems of ODEs with a fixed number of
 * unknowns. Utilizing StaticVector allows the ODE solver to be executed within GPU kernels. This capability is particularly
 * useful for scenarios like running multiple sequential solvers in parallel, as in the case of
 *  \ref TNL::Algorithms::parallelFor.
 * 2. Dynamic vectors (\ref TNL::Containers::Vector or \ref TNL::Containers::VectorView): These are preferred when dealing with
 * large systems of ODEs, such as those arising in the solution of
 * [parabolic partial differential equations](https://en.wikipedia.org/wiki/Parabolic_partial_differential_equation) using the
[method of lines](https://en.wikipedia.org/wiki/Method_of_lines).
  * In these instances, the solver typically handles a single, large-scale problem that can be executed in parallel internally.
  *
  */
namespace TNL::Solvers::ODE::Methods {}
