#pragma once

//#include <TNL/Solvers/ODE/StaticEuler.h>

//template< typename Real >
//using ODETestSolver = TNL::Solvers::ODE::StaticEuler< Real >;


#include <TNL/Solvers/ODE/StaticODESolver.h>
#include <TNL/Solvers/ODE/Methods/Euler.h>

template< typename DofVector >
using StaticODETestSolver = TNL::Solvers::ODE::ODESolver< TNL::Solvers::ODE::Methods::Euler<>, Real >;


#include "ODEStaticSolverTest.h"
