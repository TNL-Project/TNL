// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

/**
 * \brief Namespace for solvers of ordinary differential equations.
 *
 * This namespace contains solvers of
 * [ordinary differential equations](https://en.wikipedia.org/wiki/Ordinary_differential_equation) characterized by the
following equations:
 *
 * \f[ \frac{d \vec u}{dt} = \vec f( t, \vec u) \text{ on } (0,T), \f]
 * \f[  \vec u( 0 )  = \vec u_{ini}. \f]
 *
 * This class of problems can be solved by \ref TNL::Solvers::ODE::ODESolver and some of the numerical methods defined
 * in \ref TNL::Solvers::ODE::Methods namespace.
 *
 */
namespace TNL::Solvers::ODE {}
