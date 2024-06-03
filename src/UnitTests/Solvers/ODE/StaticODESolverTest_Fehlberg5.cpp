#include <TNL/Containers/StaticVector.h>
#include <TNL/Solvers/ODE/ODESolver.h>
#include <TNL/Solvers/ODE/Methods/Fehlberg5.h>

using ODEMethod = TNL::Solvers::ODE::Methods::Fehlberg5<>;

constexpr double expected_eoc = 5.0;

#include "ODEStaticSolverTest.h"
