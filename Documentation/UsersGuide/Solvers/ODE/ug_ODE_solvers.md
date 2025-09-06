# ODE solvers  {#ug_ODE_solvers}

[TOC]

## Introduction

In this section, we describe solvers of [ordinary differential equations](https://en.wikipedia.org/wiki/Ordinary_differential_equation) (ODE) characterized by the following equation:

\f[ \frac{\mathrm{d} \vec u(t)}{\mathrm{d}t} = \vec f( t, \vec u(t)) \text{ on } (0,T), \f]

and the initial condition

\f[  \vec u( 0 )  = \vec u_{ini}, \f]

where \f$ T>0 \f$. This class of problems can be solved by \ref TNL::Solvers::ODE::ODESolver which incorporates the following template parameters:

1. `Method` - specifies the numerical method used for solving the ODE.
2. `Vector` - denotes a container used to represent the vector \f$ \vec u(t) \f$.
3. `SolverMonitor` - is a tool for tracking the progress of the ODE solver.

### Method

TNL provides several methods for ODE solution, categorized based on their order of accuracy:

**1-st order accuracy methods:**
1. \ref TNL::Solvers::ODE::Methods::Euler or \ref TNL::Solvers::ODE::Methods::Matlab::ode1 - the [forward Euler](https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods) method.
2. \ref TNL::Solvers::ODE::Methods::Midpoint - the [explicit midpoint](https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods) method.

**2-nd order accuracy methods:**
1. \ref TNL::Solvers::ODE::Methods::Heun2 or \ref TNL::Solvers::ODE::Methods::Matlab::ode2 - the [Heun](https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods) method with adaptive choice of the time step.
2. \ref TNL::Solvers::ODE::Methods::Ralston2 - the [Ralston](https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods) method.
3. \ref TNL::Solvers::ODE::Methods::Fehlberg2 - the [Fehlberg](https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods) method with adaptive choice of the time step.


**3-rd order accuracy methods:**
1. \ref TNL::Solvers::ODE::Methods::Kutta - the [Kutta](https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods) method.
2. \ref TNL::Solvers::ODE::Methods::Heun3 - the [Heun](https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods) method.
3. \ref TNL::Solvers::ODE::Methods::VanDerHouwenWray - the [Van der Houwen/Wray](https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods) method.
4. \ref TNL::Solvers::ODE::Methods::Ralston3 - the [Ralston](https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods) method.
5. \ref TNL::Solvers::ODE::Methods::SSPRK3 - the [Strong Stability Preserving Runge-Kutta](https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods) method.
6. \ref TNL::Solvers::ODE::Methods::BogackiShampin or \ref TNL::Solvers::ODE::Methods::Matlab::ode23 - [Bogacki-Shampin](https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods) method with adaptive choice of the time step.

**4-th order accuracy methods:**
1. \ref TNL::Solvers::ODE::Methods::OriginalRungeKutta - the ["original" Runge-Kutta](https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods) method.
2. \ref TNL::Solvers::ODE::Methods::Rule38 - [3/8 rule](https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods) method.
3. \ref TNL::Solvers::ODE::Methods::Ralston4 - the [Ralston](https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods) method.
4. \ref TNL::Solvers::ODE::Methods::KuttaMerson - the [Runge-Kutta-Merson](https://encyclopediaofmath.org/wiki/Kutta-Merson_method) method with adaptive choice of the time step.

**5-th order accuracy methods:**
1. \ref TNL::Solvers::ODE::Methods::CashKarp - the [Cash-Karp](https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods) method with adaptive choice of the time step.
2. \ref TNL::Solvers::ODE::Methods::DormandPrince or \ref TNL::Solvers::ODE::Methods::Matlab::ode45 - the [Dormand-Prince](https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods) with adaptive choice of the time step.
3. \ref TNL::Solvers::ODE::Methods::Fehlberg5 - the [Fehlberg](https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods) method with adaptive choice of the time step.

### Vector

The vector \f$ \vec u(t) \f$ in ODE solvers can be represented using different types of containers, depending on the size and nature of the ODE system:

1. **Static vectors** (\ref TNL::Containers::StaticVector): This is suitable for small systems of ODEs with a fixed number of unknowns. Utilizing `StaticVector` allows the ODE solver to be executed within GPU kernels. This capability is particularly useful for scenarios like running multiple sequential solvers in parallel, as in the case of \ref TNL::Algorithms::parallelFor.
2. **Dynamic vectors** (\ref TNL::Containers::Vector or \ref TNL::Containers::VectorView): These are preferred when dealing with large systems of ODEs, such as those arising in the solution of [parabolic partial differential equations](https://en.wikipedia.org/wiki/Parabolic_partial_differential_equation) using the [method of lines](https://en.wikipedia.org/wiki/Method_of_lines). In these instances, the solver typically handles a single, large-scale problem that can be executed in parallel internally.

## Static ODE solvers

### Scalar problem

Static solvers are primarily intended for scenarios where \f$ x \in R \f$ is scalar or \f$ \vec x \in R^n \f$ is vector with a relatively small dimension. We will demonstrate this through a scalar problem defined as follows:

\f[ \frac{\mathrm{d}u}{\mathrm{d}t} = t \sin ( t ) \text{ on } (0,T), \f]

with the initial condition

\f[ u( 0 )  = 0. \f]

First, we define the `Real` type to represent floating-point arithmetic, here chosen as `double`.

\snippetlineno Solvers/ODE/StaticODESolver-SineExample.h Real definition

Next we define the main function of the solver:

\snippetlineno Solvers/ODE/StaticODESolver-SineExample.h Main function

We begin the main function by defining necessary types:

\snippetlineno Solvers/ODE/StaticODESolver-SineExample.h Types definition

We set `Vector` as `StaticVector` with a size of one. We also choose the basic Euler method as the `Method`. Finally, we construct the `ODESolver` type.

Next we define the time-related variables:

\snippetlineno Solvers/ODE/StaticODESolver-SineExample.h Time variables

Here:
1. `final_t` represents the size of the time interval \f$ (0,T)\f$.
2. `tau` is the integration time step.
3. `output_time_step` denotes checkpoints in which we will print value of the solution \f$ u(t)\f$.

Next, we initialize the solver:

\snippetlineno Solvers/ODE/StaticODESolver-SineExample.h Solver setup

We create an instance of the `ODESolver`, set the integration time step (using `setTau`) and the initial time of the solver with `setTime`. Then, we initialize the variable `u` (representing the ODE solution) to the initial state \f$ u(0) = 0\f$.

We proceed to the main loop iterating over the time interval \f$ (0,T) \f$:

\snippetlineno Solvers/ODE/StaticODESolver-SineExample.h Time loop

We iterate with the time variable \f$ t \f$ (represented by `getTime`) until the time \f$ T \f$ (represented by `final_t`) with step given by the frequency of the checkpoints (represented by `output_time_steps`). We let the solver to iterate until the next checkpoint or the end of the interval \f$(0,T) \f$ depending on what occurs first (it is expressed by `TNL::min(solver.getTime()+output_time_step,final_t)`). The lambda function `f` represents the right-hand side \f$ f \f$ of the ODE being solved. The lambda function receives the following arguments:

* `t` is the current value of the time variable \f$ t \in (0,T)\f$,
* `tau` is the current integration time step,
* `u` is the current value of the solution \f$ u(t)\f$,
* `fu` is a reference on a variable into which we evaluate the right-hand side \f$ f(u,t) \f$ on the ODE.

The lambda function is supposed to compute just the value of `fu`. It is `fu=t*sin(t)` in our case. Finally we call the ODE solver (`solver.solve(u,f)`). As parameters, we pass the variable `u` representing the solution \f$ u(t)\f$ and a lambda function representing the right-hand side of the ODE. At the end, we print values of the solution at given checkpoints.

The complete example looks as:

\includelineno Solvers/ODE/StaticODESolver-SineExample.h

The output is as follows:

\include StaticODESolver-SineExample.out

These results can be visualized using several methods. One option is to use [Gnuplot](http://www.gnuplot.info/). The Gnuplot command to plot the data is:

```
plot 'StaticODESolver-SineExample.out' with lines
```

Alternatively, the data can be processed and visualized using the following [Python](https://www.python.org/) script, which employs [Matplotlib](https://matplotlib.org/) for graphing.

\includelineno Solvers/ODE/StaticODESolver-SineExample.py

The graph depicting the solution of the scalar ODE problem is illustrated below:

\image{inline} html StaticODESolver-SineExample.png "Solution of the scalar ODE problem"

### Lorenz system

In this example, we demonstrate the application of the static ODE solver in solving a system of ODEs, specifically the [Lorenz system](https://en.wikipedia.org/wiki/Lorenz_system). The Lorenz system is a set of three coupled, nonlinear differential equations defined as follows:

\f[ \frac{\mathrm{d}x}{\mathrm{d}t} = \sigma( x - y), \text{ on } (0,T), \f]
\f[ \frac{\mathrm{d}y}{\mathrm{d}t} = x(\rho - z ) - y, \text{ on } (0,T),  \f]
\f[ \frac{\mathrm{d}z}{\mathrm{d}t} = xy - \beta z, \text{ on } (0,T), \f]

with the initial condition

\f[ \vec u(0) = (x(0),y(0),z(0)) = \vec u_{ini}. \f]

Here, \f$ \sigma, \rho \f$ and \f$ \beta \f$ are given constants. The solution of the system,
\f$ \vec u(t) = (x(t), y(t), z(t)) \in R^3 \f$ is represented by three-dimensional static vector (\ref TNL::Containers::StaticVector).

The implementation of the solver for the Lorenz system is outlined below:

\includelineno Solvers/ODE/StaticODESolver-LorenzExample.h

This code shares similarities with the previous example, with the following key differences:

1. We define the type `Vector` of the variable `u` representing the solution \f$ \vec u(t) \f$ as \ref TNL::Containers::StaticVector< 3, Real >, i.e. static vector with size of three.
\snippetlineno Solvers/ODE/StaticODESolver-LorenzExample.h Types definition
2. Alongside the solver parameters (`final_t`, `tau` and `output_time_step`) we define the Lorenz system's parameters (`sigma`, `rho` and `beta`).
\snippetlineno Solvers/ODE/StaticODESolver-LorenzExample.h Problem parameters
3. The initial condition is \f$ \vec u(0) = (1,2,3) \f$.
\snippetlineno Solvers/ODE/StaticODESolver-LorenzExample.h Initial condition
4. In the lambda function, which represents the right-hand side of the Lorenz system, auxiliary aliases `x`, `y`, and `z` are defined for readability. The main computation for the right-hand side of the system is implemented subsequently.
\snippetlineno Solvers/ODE/StaticODESolver-LorenzExample.h Lambda function
5. In the remaining part of the time loop, we simply execute the solver, allowing it to evolve the solution until the next snapshot time. At this point, we also print the current state of the solution to the terminal.

The solver generates a file containing the solution values \f$ (\sigma(i \tau), \rho( i \tau), \beta( i \tau )) \f$ for each time step, where \f$ i = 0, 1, \ldots N \f$. These values are recorded on separate lines. The content of the output file is structured as follows:

```
sigma[ 0 ] rho[ 0 ] beta[ 0 ]
sigma[ 1 ] rho[ 1 ] beta[ 1 ]
sigma[ 2 ] rho[ 2 ] beta[ 2 ]
...
```

The output file generated by the solver can be visualized in various ways. One effective method is to use [Gnuplot](http://www.gnuplot.info), which allows for interactive 3D visualization. The [Gnuplot](http://www.gnuplot.info) command for plotting the data in 3D is:


```
splot 'StaticODESolver-LorenzExample.out' with lines
```

Alternatively, the data can also be processed using the following [Python](https://www.python.org) script:

\includelineno Solvers/ODE/StaticODESolver-LorenzExample.py

This script is structured similarly to the one in the previous example. It processes the output data and creates a visual representation of the solution.

The resultant visualization of the Lorenz problem is shown below:

\image{inline} html StaticODESolver-LorenzExample.png "Solution of the Lorenz problem"

## Combining static ODE solvers with parallel for

Static solvers can be effectively utilized within lambda functions in conjunction with \ref TNL::Algorithms::parallelFor. This approach is particularly beneficial when there's a need to solve a large number of independent ODE problems, such as in parametric analysis scenarios. We will demonstrate this application using the two examples previously described.

### Solving scalar problems in parallel

The first example addresses an ODE defined by the following equation

\f[ \frac{\mathrm{d}u}{\mathrm{d}t} = t \sin ( c t ) \text{ on } (0,T), \f]

and the initial condition

\f[ u( 0 )  = 0, \f]

where \f$ c \f$ is a constant. We aim to solve this ODE in parallel for a range of values \f$ c \in \langle c_{min}, c_{max} \rangle \f$. The exact solution for this equation is available [here](https://www.wolframalpha.com/input?i=y%27%28t%29+%3D++t+sin%28+a+t%29). The implementation for this parallel computation is detailed in the following code:

\includelineno Solvers/ODE/StaticODESolver-SineParallelExample.h

In this example, we demonstrate how to execute the ODE solver on a GPU. To facilitate this, the main solver logic has been encapsulated within a separate function, `solveParallelODEs`, which accepts a template parameter `Device` indicating the target device for execution. The results from individual ODE solutions are stored in memory and eventually written to a file named `file_name`. The variable \f$ u \f$, being scalar, is represented by the type `TNL::Containers::StaticVector<1,Real>` within the solver.

\snippetlineno Solvers/ODE/StaticODESolver-SineParallelExample.h Types definition

Next, we define the parameters of the ODE solver (`final_t`, `tau` and `output_time_step`) as shown:

\snippetlineno Solvers/ODE/StaticODESolver-SineParallelExample.h Time variables

The interval for the ODE parameter \f$ c \in \langle c_{min}, c_{max} \rangle \f$ ( `c_min`, `c_max`, )
is established, along with the number of values `c_vals` distributed equidistantly in the interval, determined by the step size `c_step`.

\snippetlineno Solvers/ODE/StaticODESolver-SineParallelExample.h Problem parameters

We use the range of different `c values` as the range for the `parallelFor` loop. This loop processes the lambda function `solve`. Before diving into the main lambda function, we allocate the vector `results` for storing the ODE problem results at different time levels. As data cannot be directly written from GPU to an output file, `results` serves as intermediary storage. Additionally, to enable GPU access, we prepare the vector view `results_view`.

\snippetlineno Solvers/ODE/StaticODESolver-SineParallelExample.h Vector for results

Proceeding to the main lambda function `solve`:

\snippetlineno Solvers/ODE/StaticODESolver-SineParallelExample.h Lambda function for solving ODE

This function receives `idx`, the index of the value of the parameter `c`. After calculating `c`, we create the ODE solver `solver` and set its parameters using `setTau` and `setTime`. We also set the initial condition of the ODE and define the variable `time_step` to count checkpoints, which are stored in memory using `results_view`.

\snippetlineno Solvers/ODE/StaticODESolver-SineParallelExample.h Solver setup

In the time loop, we iterate over the interval \f$ (0, T) \f$, setting the solver’s stop time with `setStopTime` and running the solver with `solve`.

\snippetlineno Solvers/ODE/StaticODESolver-SineParallelExample.h Time loop

Each checkpoint's result is then stored in the `results_view`.

\snippetlineno Solvers/ODE/StaticODESolver-SineParallelExample.h Write results to file

It's important to note how the parameter `c` is passed to the lambda function `f`. The `solve` method of ODE solvers accepts user-defined parameters through variadic templates. This means additional parameters like `c` can be included alongside `u` and the right-hand side `f`, and are accessible within the lambda function `f`.

\snippetlineno Solvers/ODE/StaticODESolver-SineParallelExample.h Lambda function for ODE

Due to limitations of the `nvcc` compiler, which does not accept lambda functions defined within another lambda function, the `f` lambda function cannot be defined inside the `solve` lambda function. Therefore, `c`, defined in `solve`, cannot be captured by `f`.

The solver outputs a file in the following format:

```
# c = c[ 0 ]
x[ 0 ] u( c[ 0 ], x[ 0 ] )
x[ 1 ] u( c[ 0 ], x[ 1 ] )
....

# c = c[ 1 ]
x[ 0 ] u( c[ 1 ], x[ 0 ] )
x[ 1 ] u( c[ 1 ], x[ 1 ] )
...
```

The file can be visuallized using [Gnuplot](http://www.gnuplot.info) as follows

```
splot 'StaticODESolver-SineParallelExample-result.out' with lines
```

or it can be processed by the following [Python](https://www.python.org) script:

\includelineno Solvers/ODE/StaticODESolver-SineParallelExample.py

The result of this example looks as follows:

\image{inline} html StaticODESolver-SineParallelExample.png ""

### Solving the Lorenz system in parallel

The second example is a parametric analysis of the Lorenz model

\f[ \frac{\mathrm{d}x}{\mathrm{d}t} = \sigma( x - y), \text{ on } (0,T) \f]
\f[ \frac{\mathrm{d}y}{\mathrm{d}t} = x(\rho - z ) - y, \text{ on } (0,T) \f]
\f[ \frac{\mathrm{d}z}{\mathrm{d}t} = xy - \beta z, \text{ on } (0,T) \f]

with the initial condition

\f[ \vec u(0) = (x(0),y(0),z(0)) = \vec u_{ini}. \f]

We solve it for different values of the model parameters:

\f[ \sigma_i = \sigma_{min} + i  \Delta \sigma, \f]
\f[ \rho_j = \rho_{min} + j  \Delta \rho, \f]
\f[ \beta_k = \beta_{min} + k \Delta \beta, \f]

where we set \f$( \Delta \sigma = \Delta \rho = \Delta \beta = l / (p-1) \f$) and \f$( i,j,k = 0, 1, \ldots, p - 1 \f$). The code of the solver looks as follows:

\includelineno Solvers/ODE/StaticODESolver-LorenzParallelExample.h

It is very similar to the previous one. There are just the following changes:

1. Since we are analysing dependence on three parameters, we involve a three-dimensional `parallelFor`. For this, we introduce a type for a three-dimensional multiindex.

   \snippetlineno Solvers/ODE/StaticODESolver-LorenzParallelExample.h MultiIndex and Real definition
2. We define minimal values for the parameters \f$ \sigma \in [10,40], \beta \in [15, 36] \f$ and \f$ \rho \in [1,16]\f$ and set the number of different values for each parameter by `parametric_steps`. The size of equidistant steps (`sigma_steps`, `rho_steps` and `beta_steps`) in the parameter variations is also set.

   \snippetlineno Solvers/ODE/StaticODESolver-LorenzParallelExample.h Problem parameters
3. Next, we allocate vector `results` for storing the solution of the Lorenz problem for various parameters.

   \snippetlineno Solvers/ODE/StaticODESolver-LorenzParallelExample.h Vector for results
4. We define the lambda function `f` for the right-hand side of the Lorenz problem and the lambda function `solve` for the ODE solver, with a specific setup of the parameters.

   \snippetlineno Solvers/ODE/StaticODESolver-LorenzParallelExample.h Lambda function for ODE
and the lambda function `solve` representing the ODE solver for the Lorenz problem

   \snippetlineno Solvers/ODE/StaticODESolver-LorenzParallelExample.h Lambda function for solving ODE
with setup of the parameters

   \snippetlineno Solvers/ODE/StaticODESolver-LorenzParallelExample.h Solver setup
5. The `solve` lambda function is executed using a three-dimensional `parallelFor` (\ref TNL::Algorithms::parallelFor). We define multi-indexes `begin` and `end` for this purpose:

   \snippetlineno Solvers/ODE/StaticODESolver-LorenzParallelExample.h Parallel for
6. The lambda function `solve` takes a multi-index `idx`, which is used to compute specific values for \f$ \sigma_i, \rho_j, \beta_k \f$, denoted as sigma_i, rho_j, and beta_k:
   \snippetlineno Solvers/ODE/StaticODESolver-LorenzParallelExample.h Parameters
These parameters must be explicitly passed to the lambda function `f`. This necessity arises due to the `nvcc` compiler's limitation of not accepting a lambda function defined within another lambda function, as mentioned before:
   \snippetlineno Solvers/ODE/StaticODESolver-LorenzParallelExample.h Parameters passing
7. The initial condition for the Lorenz problem is set to vector \f$ (1,1,1) \f$:

   \snippetlineno Solvers/ODE/StaticODESolver-LorenzParallelExample.h Solver setup
Subsequently, we initiate the time loop. Within this loop, we store the state of the solution in the vector view `results_view` at intervals defined by `output_time_step`:

   \snippetlineno Solvers/ODE/StaticODESolver-LorenzParallelExample.h Time loop
Upon solving all ODEs, we transfer all solutions from the vector `results` to an output file:

   \snippetlineno Solvers/ODE/StaticODESolver-LorenzParallelExample.h Write results to file

The output file has the following format:

```
# sigma = c[ 0 ] rho = rho[ 0 ] beta = beta[ 0 ]
x[ 0 ] u( sigma[ 0 ], rho[ 0 ], beta[ 0 ], x[ 0 ] )
x[ 1 ] u( sigma[ 0 ], rho[ 0 ], beta[ 0 ], x[ 1 ] )
....

# sigma = c[ 1 ] rho = rho[ 1 ] beta = beta[ 1 ]
x[ 0 ] u( sigma[ 1 ], rho[ 1 ], beta[ 1 ], x[ 0 ] )
x[ 1 ] u( sigma[ 1 ], rho[ 1 ], beta[ 1 ], x[ 1 ] )
...
```

It can be processed by the following [Python](https://www.python.org) script:

\includelineno Solvers/ODE/StaticODESolver-LorenzParallelExample.py

The results are visualized in the following images:

\image{inline} html StaticODESolver-LorenzParallelExample-1.png ""
\image{inline} html StaticODESolver-LorenzParallelExample-2.png ""

\image{inline} html StaticODESolver-LorenzParallelExample-3.png ""
\image{inline} html StaticODESolver-LorenzParallelExample-4.png ""

## Dynamic ODE Solvers

In this section, we demonstrate how to solve the simple 1D [heat equation](https://en.wikipedia.org/wiki/Heat_equation), a [parabolic partial differential equation](https://en.wikipedia.org/wiki/Parabolic_partial_differential_equation) expressed as:

\f[
\frac{\partial u(t,x)}{\partial t} - \frac{\partial^2 u(t,x)}{\partial^2 x} = 0 \text{ on } (0,T) \times (0,1),
\f]

with boundary conditions

\f[
u(t,0) = 0,
\f]

\f[
u(t,0) = 1,
\f]

and initial condition

\f[
u(0,x) = u_{ini}(x) \text{ on } [0,1].
\f]

We discretize the equation by the [finite difference method](https://en.wikipedia.org/wiki/Finite_difference_method) for numerical approximation. First, we define set of nodes \f$ x_i = ih \f$ for \f$i=0,\ldots n-1 \f$ where \f$h = 1 / (n-1) \f$ (adopting C++ indexing for consistency). Employing the [method of lines](https://en.wikipedia.org/wiki/Method_of_lines) and approximating the second derivative by the central finite difference

\f[
\frac{\partial^2 u(t,x)}{\partial^2 x} \approx \frac{u_{i-1} - 2 u_i + u_{i+1}}{h^2},
\f]

we derive system of ODEs:

\f[
\frac{\mathrm{d} u_i(t)}{\mathrm{d}t} = \frac{u_{i-1} - 2 u_i + u_{i+1}}{h^2} \text{ for } i = 1, \ldots, n-2,
\f]

where \f$ u_i(t) = u(t,ih) \f$ represents the function value at node \f$ i \f$ and \f$ h \f$ is the spatial step between two adjacent nodes of the numerical mesh. The boundary conditions are set as:

\f[
u_0(t) = u_{n-1}(t) = 0 \text{ on } [0,T].
\f]


What are the main differences compared to the Lorenz model?


1. **System Size and Vector Representation:**
   - The Lorenz model has a fixed size of three, representing the parameters \f$ (x, y, z) \f$ in \f$ R^3 \f$. This small, fixed-size vector can be efficiently represented using a static vector, specifically \ref TNL::Containers::StaticVector< 3, Real >.
   - In contrast, the size of the ODE system for the heat equation, as determined by the [method of lines](https://en.wikipedia.org/wiki/Method_of_lines), varies based on the desired accuracy. The greater the value of \f$ n \f$, the more accurate the numerical approximation. The number of nodes \f$ n \f$ for spatial discretization defines the number of parameters or [degrees of freedom, DOFs](https://en.wikipedia.org/wiki/Degrees_of_freedom). Consequently, the system size can be large, necessitating the use of a dynamic vector, \ref TNL::Containers::Vector, for solutions.
2. **Evaluation Approach and Parallelization:**
   - Due to its small size, the Lorenz model's right-hand side can be evaluated sequentially by a single thread.
   - However, the ODE system for the heat equation can be very large, requiring parallel computation to efficiently evaluate its right-hand side.
3. **Data Allocation and Execution Environment:**
   - The dynamic vector \ref TNL::Containers::Vector allocates data dynamically, which precludes its creation within a GPU kernel. As a result, ODE solvers cannot be instantiated within a GPU kernel.
   - Therefore, the lambda function `f`, which evaluates the right-hand side of the ODE system, is executed on the host. It utilizes \ref TNL::Algorithms::parallelFor to facilitate the parallel evaluation of the system's right-hand side.

### Basic setup

The implementation of the solver for the heat equation is detailed in the following way:

\includelineno Solvers/ODE/ODESolver-HeatEquationExample.h

The solver is encapsulated within the function `solveHeatEquation`, which includes a template parameter `Device`. This parameter specifies the device (e.g., CPU or GPU) on which the solver will execute. The implementation begins with defining necessary types:

\snippetlineno Solvers/ODE/ODESolver-HeatEquationExample.h Types definition

1. `Vector` is alias for \ref TNL::Containers::Vector. The choice of a dynamic vector (rather than a static vector like \ref TNL::Containers::StaticVector) is due to the potentially large number of degrees of freedom (DOFs) that need to be stored in a resizable and dynamically allocated vector.
2. `VectorView` is designated for creating vector view, which is essential for accessing data within lambda functions, especially when these functions are executed on the GPU.
3. `Method` defines the numerical method for time integration.
4. `ODESolver` is the type of the ODE solver that will be used for calculating the time evolution in the time-dependent heat equation.

After defining these types, the next step in the implementation is to establish the parameters of the discretization:

\snippetlineno Solvers/ODE/ODESolver-HeatEquationExample.h Parameters of the discretisation

1. `final_t` represents the length of the time interval \f$ [0,T] \f$.
2. `output_time_step` defines the time intervals at which the solution \f$ u \f$ will be written into a file.
3. `n` stands for the number of DOFs, i.e. number of nodes used for the [finite difference method](https://en.wikipedia.org/wiki/Finite_difference_method).
4. `h` is the space step, meaning the distance between two consecutive nodes.
5. `tau` represents the time step used for integration by the ODE solver. For a second order parabolic problem like this, the time step should be proportional to \f$ h^2 \f$.
6. `h_sqr_inv` is an auxiliary constant equal to \f$ 1/h^2 \f$. It is used later in the finite difference method for approximating the second derivative.

The initial condition \f$ u_{ini} \f$ is set as:

\f[
u_{ini}(x) = \left\{
   \begin{array}{rl}
   0 & \text{ for } x < 0.4, \\
   1 & \text{ for } 0.4 \leq x \leq 0.6, \\
   0 & \text{ for } x > 0. \\
   \end{array}
\right.
\f]

This initial condition defines the state of the system at the beginning of the simulation. It specifies a region (between 0.4 and 0.6) where the temperature (or the value of \f$ u \f$) is set to 1, and outside this region, the temperature is 0.

\snippetlineno Solvers/ODE/ODESolver-HeatEquationExample.h Initial condition

After setting the initial condition, the next step is to write it to a file. This is done using the `write` function, which is detailed later.

Next, we create an instance of the ODE solver `solver` and set the integration time step `tau` of the solver (\ref TNL::Solvers::ODE::ExplicitSolver::setTau ) The initial time is set to zero with (\ref TNL::Solvers::ODE::ExplicitSolver::setTime).

\snippetlineno Solvers/ODE/ODESolver-HeatEquationExample.h Solver setup

Finally, we proceed to the time loop:

\snippetlineno Solvers/ODE/ODESolver-HeatEquationExample.h Time loop

The time loop is executed, utilizing the time variable from the ODE solver (\ref  TNL::Solvers::ODE::ExplicitSolver::getTime). The loop iterates until reaching the end of the specified time interval \f$ [0, T] \f$, defined by the variable `final_t`. The stop time of the ODE solver is set with \ref TNL::Solvers::ODE::ExplicitSolver::setStopTime, either to the next checkpoint for storing the state of the system or to the end of the time interval, depending on which comes first.

The lambda function `f` is defined to express the discretization of the second derivative of \f$ u \f$ using the central finite difference and to incorporate the boundary conditions.

\snippetlineno Solvers/ODE/ODESolver-HeatEquationExample.h Lambda function f

The function receives the following parameters:

1. `i` represents the index of the node, corresponding to the individual ODE derived from the method of lines. It is essential for evaluating the update of \f$ u_i^k \f$ to reach the next time level \f$ u_i^{k+1} \f$.
2. `u` is a vector view representing the state \f$ u_i^k \f$ of the heat equation on the \f$ k-\f$th time level.
3. `fu` is a vector that stores updates or time derivatives in the method of lines which will transition \f$ u \f$ to the next time level.

In this implementation, the lambda function `f` plays a crucial role in applying the finite difference method to approximate the second derivative and in enforcing the boundary conditions, thereby driving the evolution of the system in time according to the heat equation.

In the lambda function `f`, the solution \f$ u \f$ remains unchanged on the boundary nodes. Therefore, the function returns zero for these boundary nodes.
For the interior nodes, `f` evaluates the central difference to approximate the second derivative.

The lambda function `time_stepping` is responsible for computing updates for all nodes \f$ i = 0, \ldots, n-1 \f$. This computation is performed using \ref TNL::Algorithms::parallelFor, which iterates over all the nodes and calls the function `f` for each one. Since the `nvcc` compiler does not support lambda functions defined within another lambda function, `f` is defined separately, and the parameters `u` and `fu` are explicitly passed to it.

\snippetlineno Solvers/ODE/ODESolver-HeatEquationExample.h Parallel for call

The ODE solver is executed with `solve`. The current state of the heat equation `u` and the lambda function `f` (controlling the time evolution) are passed to the `solve` method. After each iteration, the current state is saved to a file using the `write` function.

\includelineno Solvers/ODE/write.h

The `write` function is used to write the solution of the heat equation to a file. The specifics of this function, including its parameters and functionality, are detailed in the following:

1. `file` specifies the file into which the solution will be stored.
2. `u` represents the solution of the heat equation at a given time. It can be a vector or vector view.
3. `n` indicates the number of nodes used for approximating the solution.
4. `h` refers to the space step, i.e., the distance between two consecutive nodes.
5. `time` represents the current time of the evolution being computed.


The solver writes the results in a structured format, making it convenient for visualization and analysis:

```
# time = t[ 0 ]
x[ 0 ] u( t[ 0 ], x[ 0 ] )
x[ 1 ] u( t[ 0 ], x[ 1 ] )
x[ 2 ] u( t[ 0 ], x[ 2 ] )
...

# time = t[ 1 ]
x[ 0 ] u( t[ 1 ], x[ 0 ] )
x[ 1 ] u( t[ 1 ], x[ 1 ] )
x[ 2 ] u( t[ 1 ], x[ 2 ] )
...
```

This format records the solution of the heat equation at various time steps and spatial nodes, providing a comprehensive view of the system's evolution over time.

The solution can be visualised with [Gnuplot](http://www.gnuplot.info/) using the command:

```
plot 'ODESolver-HeatEquationExample-result.out' with lines
```

This command plots the solution as a line graph, offering a visual representation of how the heat equation's solution evolves. The solution can also be parsed and processed in [Python](https://www.python.org/), using [Matplotlib](https://matplotlib.org/) for visualization. The specifics of this process are detailed in the following script:


\includelineno Solvers/ODE/ODESolver-HeatEquationExample.py

The outcome of the solver, once visualized, is shown as follows:

\image{inline} html ODESolver-HeatEquationExample.png "Heat equation"

### Setup with a solver monitor

In this section, we'll discuss how to integrate an ODE solver with a solver monitor, as demonstrated in the example:

\includelineno Solvers/ODE/ODESolver-HeatEquationWithMonitorExample.h

This setup incorporates a solver monitor into the ODE solver framework, which differs from the previous example in several key ways:

The first difference is the inclusion of a header file `TNL/Solvers/IterativeSolverMonitor.h` for the iterative solver monitor. This step is essential for enabling monitoring capabilities within the solver.
We have to setup the solver monitor:

\snippetlineno Solvers/ODE/ODESolver-HeatEquationWithMonitorExample.h Monitor setup

 First, we define the monitor type `IterativeSolverMonitorType ` and we create an instance of the monitor. A separate thread (`monitorThread`) is created for the monitor. The refresh rate of the monitor is set to 10 milliseconds with `setRefreshRate` and verbose mode is enabled with `setVerbose` for detailed monitoring. The solver stage name is specified with `setStage`. The monitor is connected to the solver using \ref TNL::Solvers::IterativeSolver::setSolverMonitor. Subsequently, the numerical computation is performed and after it finishes, the monitor is stopped by calling \ref TNL::Solvers::IterativeSolverMonitor::stopMainLoop.

## Use of the iterate method

The ODE solvers in TNL provide an `iterate` method for performing just one iteration. This is particularly useful when there is a need for enhanced control over the time loop, or when developing a hybrid solver that combines multiple integration methods. The usage of this method is demonstrated in the following example:

\includelineno Solvers/ODE/StaticODESolver-SineExample_iterate.h

For simplicity, we demonstrate the use of `iterate` with a static solver, but the process is similar for dynamic solvers. There are two main differences compared to using the solve method:

1. **Initialization:** Before calling `iterate`, it's necessary to initialize the solver using the `init` method. This step sets up auxiliary vectors within the solver. For ODE solvers with dynamic vectors, the internal vectors of the solver are allocated based on the size of the vector `u`.

\snippetlineno Solvers/ODE/StaticODESolver-SineExample_iterate.h Solver setup
2. **Time Loop:** Within the time loop, the `iterate` method is called. It requires the vector `u`, the right-hand side function `f` of the ODE, and also the variables `time` and `tau`, representing the current time \f$ t \f$ and the integration time step, respectively. The variable `time` is incremented by `tau` with each iteration. Additionally, `tau` can be adjusted if the solver performs adaptive time step selection. It's important to adjust `tau` to ensure that the `next_output_time` is reached exactly.

\snippetlineno Solvers/ODE/StaticODESolver-SineExample_iterate.h Time loop

## User defined methods

The Runge-Kutta methods used for solving ODEs can generally be expressed as follows:

\f[ k_1 = f(t, \vec u) \f]
\f[ k_2 = f(t + c_2, \vec u + \tau(a_{21} k_1)) \f]
\f[ k_3 = f(t + c_3, \vec u + \tau(a_{31} k_1 + a_{32} k_2)) \f]
\f[ \vdots \f]
\f[ k_s = f(t + c_s, \vec u + \tau( \sum_{j=1}^{s-1} a_{si} k_i ) )\f]


\f[ \vec u_{n+1} = \vec u_n + \tau \sum_{i=1}^s b_i k_i\f]
\f[ \vec u^\ast_{n+1} = \vec u_n + \tau \sum_{i=1}^s b^\ast_i k_i\f]
\f[ \vec e_{n+1} = \vec u_{n+1} - \vec u^\ast_{n+1} = \vec u_n + \tau \sum_{i=1}^s (b_i - b^\ast_i) k_i \f]

where \f$s\f$ denotes the number of stages, the vector \f$\vec e_{n+1} \f$ is an error estimate and a basis for the adaptive choice of the integration step. Each such method can be expressed in the form of a [Butcher tableau](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods#Explicit_Runge.E2.80.93Kutta_methods) having the following form:


|                |                  |                  |                |                   |
| -------------- | ---------------- | ---------------- | -------------- | ----------------- |
| \f$ c_1 \f$    |                  |                  |                |                   |
| \f$ c_2 \f$    | \f$ a_{21} \f$   |                  |                |                   |
| \f$ c_3 \f$    | \f$ a_{31} \f$   | \f$ a_{32} \f$   |                |                   |
| \f$ \vdots \f$ | \f$ \vdots \f$   | \f$ \vdots \f$   | \f$ \ddots \f$ |                   |
| \f$ c_s \f$    | \f$ a_{s1} \f$   | \f$ a_{s2} \f$   | \f$ \ldots \f$ | \f$ a_{s,s-1} \f$ |
|                | \f$ b_1 \f$      | \f$ b_2 \f$      | \f$ \ldots \f$ | \f$ b_s \f$       |
|                | \f$ b^\ast_1 \f$ | \f$ b^\ast_2 \f$ | \f$ \ldots \f$ | \f$ b^\ast_s \f$  |


 For example, the the Fehlberg RK1(2) method can be expressed as:

|             |               |                 |               |
| ----------- | ------------- | --------------- | ------------- |
| \f$ 0 \f$   |               |                 |               |
| \f$ 1/2 \f$ | \f$ 1/2 \f$   |                 |               |
| \f$ 1 \f$   | \f$ 1/256 \f$ | \f$ 1/256 \f$   |               |
|             | \f$ 1/512 \f$ | \f$ 255/256 \f$ | \f$ 1/521 \f$ |
|             | \f$ 1/256 \f$ | \f$ 255/256 \f$ |               |

TNL allows the implementation of new Runge-Kutta methods simply by specifying the Butcher tableau. The following is an example of the implementation of the Fehlberg RK1(2):

\includelineno Solvers/ODE/Fehlberg2.h

The method is a templated structure accepting a template parameter `Value` indicating the numeric type of the method. To implement a new method, we need to do the following:

1. **Set the Stages of the Method:**
   This is \f$ s \f$ in the definition of the Runge-Kutta method and equals the number of vectors \f$ \vec k_i \f$.

   \snippetlineno Solvers/ODE/Fehlberg2.h Stages

2. **Determine Adaptivity:**
   If the Runge-Kutta method allows an adaptive choice of the time step, the method `isAdaptive` shall return `true`, and `false` otherwise.

   \snippetlineno Solvers/ODE/Fehlberg2.h Adaptivity
3. **Define Method for Error Estimation:**
   Adaptive methods need coefficients for error estimation. Typically, these are the differences between coefficients for updates of higher and lower order of accuracy, as seen in `getErrorCoefficients`. However, this method can be altered if there's a different formula for error estimation.

   \snippetlineno Solvers/ODE/Fehlberg2.h Error coefficients
4. **Define Coefficients for Evaluating Vectors** \f$ \vec k_i \f$:
   The coefficients for the evaluation of the vectors \f$ \vec k_i \f$ need to be defined.

   \snippetlineno Solvers/ODE/Fehlberg2.h k coefficients definition

   and

   \snippetlineno Solvers/ODE/Fehlberg2.h Time coefficients definition

   Zero coefficients are omitted in the generated formulas at compile time, ensuring no performance drop.
5. **Define Update Coefficients:**
   Finally, the update coefficients must be defined.

   \snippetlineno Solvers/ODE/Fehlberg2.h Update coefficients definition

Such structure can be substituted to the \ref TNL::Solvers::ODE::ODESolver as the template parameter `Method`.
