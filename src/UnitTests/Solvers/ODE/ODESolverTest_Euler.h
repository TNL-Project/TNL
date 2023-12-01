#pragma once

#include <TNL/Solvers/ODE/ODESolver.h>
#include <TNL/Solvers/ODE/Methods/Euler.h>

template< typename DofVector >
using ODETestSolver = TNL::Solvers::ODE::ODESolver< TNL::Solvers::ODE::Methods::Euler<>, DofVector >;

#include "ODESolverTest.h"
