#include <TNL/Containers/StaticVector.h>
#include <TNL/Solvers/ODE/ODESolver.h>
#include <TNL/Solvers/ODE/Methods/OriginalRungeKutta.h>

using ODEMethod = TNL::Solvers::ODE::Methods::OriginalRungeKutta<>;

constexpr double expected_eoc = 4.0;

#include "ODEStaticSolverTest.h"
