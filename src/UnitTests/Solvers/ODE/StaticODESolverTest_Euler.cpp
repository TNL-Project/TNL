#include <TNL/Containers/StaticVector.h>
#include <TNL/Solvers/ODE/ODESolver.h>
#include <TNL/Solvers/ODE/Methods/Euler.h>

using ODEMethod = TNL::Solvers::ODE::Methods::Euler<>;

constexpr double expected_eoc = 1.0;

#include "ODEStaticSolverTest.h"
