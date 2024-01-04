#include <TNL/Containers/StaticVector.h>
#include <TNL/Solvers/ODE/ODESolver.h>
#include <TNL/Solvers/ODE/Methods/KuttaMerson.h>

using ODEMethod = TNL::Solvers::ODE::Methods::KuttaMerson<>;

constexpr double expected_eoc = 4.0;

#include "ODEStaticSolverTest.h"
