#include <TNL/Containers/StaticVector.h>
#include <TNL/Solvers/ODE/ODESolver.h>
#include <TNL/Solvers/ODE/Methods/RungeKutta.h>

using ODEMethod = TNL::Solvers::ODE::Methods::RungeKutta<>;

constexpr double expected_eoc = 4.0;

#include "ODEStaticSolverTest.h"
