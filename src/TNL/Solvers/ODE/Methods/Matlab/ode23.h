// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#include <TNL/Solvers/ODE/Methods/DormandPrince.h>

#pragma once

namespace TNL::Solvers::ODE::Methods::Matlab {

/**
 * \brief Third order [Bogacki-Shampin](https://en.wikipedia.org/wiki/Dormand%E2%80%93Prince_method) method also known as ode23
 * from [Matlab](https://www.mathworks.com/help/simulink/gui/solver.html) with adaptive step size.
 *
 * \tparam Value is arithmetic type used for computations.
 */
template< typename Value = double >
using ode23 = BogackiShampin< Value >;

}  // namespace TNL::Solvers::ODE::Methods::Matlab
