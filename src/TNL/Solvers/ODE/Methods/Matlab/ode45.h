// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#include <TNL/Solvers/ODE/Methods/DormandPrince.h>

#pragma once

namespace TNL::Solvers::ODE::Methods::Matlab {

/**
 * \brief Fifth order [Dormand-Prince](https://en.wikipedia.org/wiki/Dormand%E2%80%93Prince_method) method also known as ode45
 * from [Matlab](https://www.mathworks.com/help/simulink/gui/solver.html) with adaptive step size.
 *
 * \tparam Value is arithmetic type used for computations.
 */
template< typename Value = double >
using ode45 = DormandPrince< Value >;

}  // namespace TNL::Solvers::ODE::Methods::Matlab
