# ODE Solvers Benchmark

This benchmark evaluates the performance of ODE solvers in TNL by solving a simple ODE:

$$
\frac{{\rm d} u( t )}{\rm d t} = e^t \ on \ \langle 0 , 1 \rangle,
$$

with the initial condition:

$$
u(0) = e^0.
$$

The exact solution at $t=1$ is $u(1) = e^1$. The benchmark calculates the Experimental Order of Convergence (EOC) by solving the problem with various time steps. The EOC is accurately evaluated only if the adaptive choice of the time step is disabled.

The benchmark tests both static and dynamic variants of the ODE solver. In both cases, a number of identical problems are solved in parallel. The goals of the benchmark are:

1. To reveal the order of convergence of the solver.
2. To assess the solver's performance and the time required to solve the problem.
3. To compare the performance of the current implementation with the legacy implementation in TNL, ensuring no performance degradation (execute the benchmark with the parameter `--legacy-solvers true`).

The benchmark can be executed using the following command:

```bash
tnl-benchmark-ode-solvers
```

The benchmark supports single or double precision, can be run on CPU or GPU, and with or without adaptive choice of integration time step. A list of all possible setup parameters can be obtained with:

```bash
tnl-benchmark-ode-solvers --help
```









