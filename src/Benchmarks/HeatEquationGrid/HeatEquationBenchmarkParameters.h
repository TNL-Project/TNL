
template< typename Real,
          typename Index >
struct HeatEquationBenchmarkParameters
{
public:

   HeatEquationBenchmarkParameters() = default;

   HeatEquationBenchmarkParameters(
      const Index xSize, const Index ySize,
      const Real xDomainSize, const Real yDomainSize,
      const Real alpha,  const Real beta,  const Real gamma,
      const Real timeStep, const Real finalTime,
      const bool outputData,
      const bool verbose)
   : xSize( xSize ), ySize( ySize ),
     xDomainSize( xDomainSize ), yDomainSize( yDomainSize ),
     alpha( alpha ), beta( beta ), gamma( gamma ),
     timeStep( timeStep ), finalTime( finalTime ),
     outputData( outputData ), verbose (verbose ) {};

   static void setupConfig( TNL::Config::ConfigDescription& config )
   {
      config.addDelimiter( "Benchmark settings:" );
      config.addEntry<TNL::String>("id", "Identifier of the run", "unknown");
      config.addEntry< std::string >( "implementation", "Implementation of the heat equation solver.", "grid" );
      config.addEntryEnum< std::string >( "parallel-for" );
      config.addEntryEnum< std::string >( "grid" );
      config.addEntryEnum< std::string >( "nd-grid" );
      config.addEntry<TNL::String>("log-file", "Log file name.", "tnl-benchmark-heat-equation.log");
      config.addEntry<TNL::String>("output-mode", "Mode for opening the log file.", "overwrite");
      config.addEntryEnum("append");
      config.addEntryEnum("overwrite");

      config.addEntry<TNL::String>("device", "Device the computation will run on.", "cuda");
      config.addEntryEnum<TNL::String>("all");
      config.addEntryEnum<TNL::String>("host");

   #ifdef HAVE_CUDA
      config.addEntryEnum<TNL::String>("cuda");
   #endif

      config.addEntry<TNL::String>("precision", "Precision of the arithmetics.", "double");
      config.addEntryEnum("float");
      config.addEntryEnum("double");
      config.addEntryEnum("all");

      config.addEntry<int>("min-x-dimension", "Minimum dimension over x axis used in the benchmark.", 100);
      config.addEntry<int>("max-x-dimension", "Maximum dimension over x axis used in the benchmark.", 200);
      config.addEntry<int>("x-size-step-factor", "Factor determining the dimension grows over x axis. First size is min-x-dimension and each following size is stepFactor*previousSize, up to max-x-dimension.", 2);

      config.addEntry<int>("min-y-dimension", "Minimum dimension over x axis used in the benchmark.", 100);
      config.addEntry<int>("max-y-dimension", "Maximum dimension over x axis used in the benchmark.", 200);
      config.addEntry<int>("y-size-step-factor", "Factor determining the dimension grows over y axis. First size is min-y-dimension and each following size is stepFactor*previousSize, up to max-y-dimension.", 2);

      config.addEntry<int>("loops", "Number of iterations for every computation.", 10);

      config.addEntry<int>("verbose", "Verbose mode.", 1);

      config.addDelimiter("Problem settings:");
      config.addEntry<double>("domain-x-size", "Domain size along x-axis.", 2.0);
      config.addEntry<double>("domain-y-size", "Domain size along y-axis.", 2.0);

      config.addDelimiter( "Initial condition settings ( (x^2/alpha + y^2/beta) + gamma)):" );
      config.addEntry< double >( "alpha", "Alpha value in initial condition", -0.05 );
      config.addEntry< double >( "beta", "Beta value in initial condition", -0.05 );
      config.addEntry< double >( "gamma", "Gamma key in initial condition", 15 );

      config.addEntry<double>("sigma", "Sigma in exponential initial condition.", 1.0);

      config.addEntry<double>("time-step", "Time step. By default it is proportional to one over space step square.", 0.000001);
      config.addEntry<double>("final-time", "Final time of the simulation.", 0.01);

      config.addDelimiter("Device settings:");
      TNL::Devices::Host::configSetup( config );
      TNL::Devices::Cuda::configSetup( config );
   };

   void setup( TNL::Config::ParameterContainer& parameters )
   {
      this->xSize =       parameters.getParameter< int  >( "grid-x-size" );
      this->ySize =       parameters.getParameter< int  >( "grid-y-size" );
      this->xDomainSize = parameters.getParameter< Real >( "domain-x-size" );
      this->yDomainSize = parameters.getParameter< Real >( "domain-y-size" );
      this->alpha =       parameters.getParameter< Real >( "alpha" );
      this->beta =        parameters.getParameter< Real >( "beta" );
      this->gamma =       parameters.getParameter< Real >( "gamma" );
      this->timeStep =    parameters.getParameter< Real >( "time-step" );
      this->finalTime =   parameters.getParameter< Real >( "final-time" );
      this->outputData =  parameters.getParameter< bool >( "outputData" );
      this->verbose =     parameters.getParameter< bool >( "verbose" );
   };

protected:
   Index xSize = 0, ySize = 0;
   Real xDomainSize = 0.0, yDomainSize = 0.0;
   Real alpha = 0.0, beta = 0.0, gamma = 0.0;
   Real timeStep = 0.0, finalTime = 0.0;
   bool outputData = false;
   bool verbose = false;
};
