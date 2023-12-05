// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#include <TNL/Solvers/ODE/Methods/Heun2.h>

#pragma once

namespace TNL::Solvers::ODE::Methods::Matlab {

/**
 * \brief Second order [Heun](https://en.wikipedia.org/wiki/Dormand%E2%80%93Prince_method) method also known as ode2
 * from [Matlab](https://www.mathworks.com/help/simulink/gui/solver.html) with adaptive step size.
 *
 * \tparam Value is arithmetic type used for computations.
 */
template< typename Value = double >
using ode2 = Heun2< Value >;

}  // namespace TNL::Solvers::ODE::Methods::Matlab
