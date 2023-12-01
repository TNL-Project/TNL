#include <TNL/Containers/StaticVector.h>
#include <TNL/Solvers/ODE/ODESolver.h>
#include <TNL/Solvers/ODE/Methods/Fehlberg2.h>

using ODEMethod = TNL::Solvers::ODE::Methods::Fehlberg2<>;

constexpr double expected_eoc = 2.0;

#include "ODEStaticSolverTest.h"
