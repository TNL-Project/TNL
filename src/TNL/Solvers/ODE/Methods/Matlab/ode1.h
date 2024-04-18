// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#include <TNL/Solvers/ODE/Methods/Euler.h>

#pragma once

namespace TNL::Solvers::ODE::Methods::Matlab {

/**
 * \brief First order [Euler](https://en.wikipedia.org/wiki/Dormand%E2%80%93Prince_method) method also known as ode1.
 *
 * \tparam Value is arithmetic type used for computations.
 */
template< typename Value = double >
using ode1 = Euler< Value >;

}  // namespace TNL::Solvers::ODE::Methods::Matlab
