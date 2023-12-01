#pragma once

#include <TNL/Containers/StaticVector.h>
#include <TNL/Solvers/ODE/ODESolver.h>
#include <TNL/Solvers/ODE/Methods/Merson.h>

template< typename DofVector >
using StaticODETestSolver =
   TNL::Solvers::ODE::ODESolver< TNL::Solvers::ODE::Methods::Merson<>, TNL::Containers::StaticVector< 1, Real > >;

#include "ODESolverTest.h"
