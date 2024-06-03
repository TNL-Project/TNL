#include <TNL/Containers/StaticVector.h>
#include <TNL/Solvers/ODE/ODESolver.h>
#include <TNL/Solvers/ODE/Methods/DormandPrince.h>

using ODEMethod = TNL::Solvers::ODE::Methods::DormandPrince<>;

constexpr double expected_eoc = 5.0;

#include "ODEStaticSolverTest.h"
