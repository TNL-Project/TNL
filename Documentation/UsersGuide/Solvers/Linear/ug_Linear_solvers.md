# Linear solvers  {#ug_Linear_solvers}

[TOC]

## Introduction

Solvers of linear systems are one of the most important algorithms in scientific
computations. TNL offers the following iterative methods:

1. Stationary methods
   1. [Jacobi method][Jacobi] (\ref TNL::Solvers::Linear::Jacobi)
   2. [Successive-overrelaxation method, SOR][SOR]
      (\ref TNL::Solvers::Linear::SOR)
2. Krylov subspace methods
   1. [Conjugate gradient method, CG][CG] (\ref TNL::Solvers::Linear::CG)
   2. [Biconjugate gradient stabilized method, BICGStab][BICGStab]
      (\ref TNL::Solvers::Linear::BICGStab)
   3. [Biconjugate gradient stabilized method, BICGStab(l)][BICGStab(l)]
      (\ref TNL::Solvers::Linear::BICGStabL)
   4. [Transpose-free quasi-minimal residual method, TFQMR][TFQMR]
      (\ref TNL::Solvers::Linear::TFQMR)
   5. [Generalized minimal residual method, GMRES][GMRES]
      (\ref TNL::Solvers::Linear::GMRES) with various methods of
      orthogonalization:
      1. Classical Gramm-Schmidt, CGS
      2. Classical Gramm-Schmidt with reorthogonalization, CGSR
      3. Modified Gramm-Schmidt, MGS
      4. Modified Gramm-Schmidt with reorthogonalization, MGSR
      5. Compact WY form of the Householder reflections, CWY

The iterative solvers (not the stationary solvers like
\ref TNL::Solvers::Linear::Jacobi and \ref TNL::Solvers::Linear::SOR)
can be combined with the following preconditioners:

1. [Diagonal or Jacobi](http://netlib.org/linalg/html_templates/node55.html)
   (\ref TNL::Solvers::Linear::Preconditioners::Diagonal)
2. ILU (Incomplete LU) - CPU only currently
   1. [ILU(0)][ILU(0)] (\ref TNL::Solvers::Linear::Preconditioners::ILU0)
   2. [ILUT (ILU with thresholding)][ILUT]
      (\ref TNL::Solvers::Linear::Preconditioners::ILUT)

[Jacobi]: https://en.wikipedia.org/wiki/Jacobi_method
[SOR]: https://en.wikipedia.org/wiki/Successive_over-relaxation
[CG]: https://en.wikipedia.org/wiki/Conjugate_gradient_method
[BICGStab]: https://en.wikipedia.org/wiki/Biconjugate_gradient_stabilized_method
[BICGStab(l)]: https://dspace.library.uu.nl/bitstream/handle/1874/16827/sleijpen_93_bicgstab.pdf
[TFQMR]: https://second.wiki/wiki/algoritmo_tfqmr
[GMRES]: https://en.wikipedia.org/wiki/Generalized_minimal_residual_method
[ILU(0)]: https://en.wikipedia.org/wiki/Incomplete_LU_factorization
[ILUT]: https://www-users.cse.umn.edu/~saad/PDF/umsi-92-38.pdf

## Iterative solvers of linear systems

### Basic setup

All iterative solvers for linear systems can be found in the namespace
\ref TNL::Solvers::Linear. The following example shows the use the iterative
solvers:

\includelineno Solvers/Linear/IterativeLinearSolverExample.cpp

In this example we solve a linear system \f$ A \vec x = \vec b \f$ where

\f[
A = \left(
\begin{array}{cccc}
 2.5 & -1   &      &      &      \\
-1   &  2.5 & -1   &      &      \\
     & -1   &  2.5 & -1   &      \\
     &      & -1   &  2.5 & -1   \\
     &      &      & -1   &  2.5 \\
\end{array}
\right)
\f]

The right-hand side vector \f$\vec b \f$ is set to
\f$( 1.5, 0.5, 0.5, 0.5, 1.5 )^T \f$ so that the exact solution is
\f$ \vec x = ( 1, 1, 1, 1, 1 )^T \f$. The elements of the matrix \f$ A \f$ are
set using the method \ref TNL::Matrices::SparseMatrix::forAllRows. In this
example, we use the sparse matrix but any other matrix type can be used as well
(see the namespace \ref TNL::Matrices). Next we set the solution vector
\f$ \vec x = ( 1, 1, 1, 1, 1 )^T \f$ and multiply it with matrix \f$ A \f$ to
get the right-hand side vector \f$ \vec b \f$. Finally, we reset the vector
\f$ \vec x \f$ to zero.

To solve the linear system in the example, we use the TFQMR solver. Other
solvers can be used as well (see the namespace \ref TNL::Solvers::Linear). The
solver needs only one template parameter which is the matrix type. Next we
create an instance of the solver and set the matrix of the linear system. Note
that the matrix is passed to the solver as a \ref std::shared_ptr. Then we set
the stopping criterion for the iterative method in terms of the relative residue,
i.e. \f$ \lVert \vec b - A \vec x \rVert / \lVert b \rVert \f$. The solver is
executed by calling the \ref TNL::Solvers::Linear::LinearSolver::solve method
which accepts the right-hand side vector \f$ \vec b \f$ and the solution vector
\f$ \vec x \f$.

The result looks as follows:

\include IterativeLinearSolverExample.out

### Setup with a solver monitor

Solution of large linear systems may take a lot of time. In such situations, it
is useful to be able to monitor the convergence of the solver or the solver
status in general. For this purpose, TNL provides a solver monitor which can
show various metrics in real time, such as current number of iterations,
current residue of the approximate solution, etc. The solver monitor in TNL
runs in a separate thread and it refreshes the status of the solver with a
configurable refresh rate (once per 500 ms by default). The use of the solver
monitor is demonstrated in the following example.

\includelineno Solvers/Linear/IterativeLinearSolverWithMonitorExample.cpp

First, we set up the same linear system as in the previous example, we create
an instance of the Jacobi solver and we pass the matrix of the linear system to
the solver. Then, we set the relaxation parameter \f$ \omega \f$ of the
\ref TNL::Solvers::Linear::Jacobi "Jacobi" solver to 0.0005. The reason is to
artificially slow down the convergence, because we want to see some iterations
in this example. Next, we create an instance of the solver monitor and a special
thread for the monitor (an instance of the \ref TNL::Solvers::SolverMonitorThread
class). We use the following methods to configure the solver monitor:

* \ref TNL::Solvers::SolverMonitor::setRefreshRate sets the refresh rate of the
  monitor to 10 milliseconds.
* \ref TNL::Solvers::IterativeSolverMonitor::setVerbose sets verbosity
  of the monitor to 1.
* \ref TNL::Solvers::IterativeSolverMonitor::setStage sets a name of the solver
  stage. The monitor stages serve for distinguishing between different stages
  of more complex solvers (for example when the linear system solver is embedded
  into a time-dependent PDE solver).

Next, we call \ref TNL::Solvers::IterativeSolver::setSolverMonitor to connect
the solver with the monitor and we set the convergence criterion based on the
relative residue. Finally, we start the solver using the
\ref TNL::Solvers::Linear::Jacobi::solve method and when the solver finishes,
we stop the monitor using \ref TNL::Solvers::SolverMonitor::stopMainLoop.

The result looks as follows:

\include IterativeLinearSolverWithMonitorExample.out

The monitoring of the solver can be improved by time elapsed since the
beginning of the computation as demonstrated in the following example:

\includelineno Solvers/Linear/IterativeLinearSolverWithTimerExample.cpp

The only changes are around the lines where we create an instance of
\ref TNL::Timer, connect it with the monitor using
\ref TNL::Solvers::SolverMonitor::setTimer and start the timer with
\ref TNL::Timer::start.

The result looks as follows:

\include IterativeLinearSolverWithTimerExample.out

### Setup with preconditioner

Preconditioners of iterative solvers can significantly improve the performance
of the solver. In the case of the linear systems, they are used mainly with the
Krylov subspace methods. Preconditioners cannot be used with the starionary
methods (\ref TNL::Solvers::Linear::Jacobi and \ref TNL::Solvers::Linear::SOR).
The following example shows how to setup an iterative solver of linear systems
with preconditioning.

\includelineno Solvers/Linear/IterativeLinearSolverWithPreconditionerExample.cpp

In this example, we solve the same problem as in all other examples in this
section. The only differences concerning the preconditioner happen in the solver
setup. Similarly to the matrix of the linear system, the preconditioner needs to
be passed to the solver as a \ref std::shared_ptr. When the preconditioner
object is created, we have to initialize it using the
\ref TNL::Solvers::Linear::Preconditioners::Preconditioner::update "update"
method, which has to be called everytime the matrix of the linear system
changes. This is important, for example, when solving time-dependent PDEs, but
it does not happen in this example. Finally, we need to connect the solver with
the preconditioner using the
\ref TNL::Solvers::Linear::LinearSolver::setPreconditioner "setPreconditioner"
method.

The result looks as follows:

\include IterativeLinearSolverWithPreconditionerExample.out

### Choosing the solver and preconditioner type at runtime

When developing a numerical solver, one often has to search for a combination
of various methods and algorithms that fit given requirements the best. To make
this easier, TNL provides the functions \ref TNL::Solvers::getLinearSolver
and \ref TNL::Solvers::getPreconditioner for selecting the linear solver and
preconditioner at runtime. The following example shows how to use these
functions:

\includelineno Solvers/Linear/IterativeLinearSolverWithRuntimeTypesExample.cpp

We still stay with the same problem and the only changes are in the solver
setup. We first use \ref TNL::Solvers::getLinearSolver to get a shared pointer
holding the solver and then \ref TNL::Solvers::getPreconditioner to get a shared
pointer holding the preconditioner. The rest of the code is the same as in the
previous examples with the only difference that we work with the pointer
`solver_ptr` instead of the direct instance `solver` of the solver type.

The result looks as follows:

\include IterativeLinearSolverWithRuntimeTypesExample.out
