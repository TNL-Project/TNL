// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#include <TNL/Solvers/ODE/Mathods/DormandPrince.h>

#pragma once

namespace TNL::Solvers::ODE::Methods::Matlab {

/**
 * \brief Dormand-Prince method also known as ode45 from Matlab.
 *
 * See [Wikipedia](https://en.wikipedia.org/wiki/Dormand%E2%80%93Prince_method) for details.
 *
 * \tparam Value is arithmetic type used for computations.
 */
template< typename Value = double >
struct ode45 : public DormandPrince< Value >
{
};

} // namespace TNL::Solvers::ODE::Methods::Matlab
