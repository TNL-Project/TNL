//#include <TNL/tnlConfig.h>
#include <TNL/Solvers/Solver.h>
#include <TNL/Solvers/BuildConfigTags.h>
#include <TNL/Operators/DirichletBoundaryConditions.h>
#include <TNL/Operators/NeumannBoundaryConditions.h>
#include <TNL/Functions/Analytic/Constant.h>
#include "twoEquationTurbulenceModelProblem.h"
#include "DifferentialOperators/Two-Equation-turbulence-model/AUSM+/AUSMPlus.h"
#include "flowsRhs.h"
#include "flowsBuildConfigTag.h"

#include "RiemannProblemInitialCondition.h"
#include "BoundaryConditions/Cavity/BoundaryConditionsCavity.h"
#include "BoundaryConditions/Boiler/BoundaryConditionsBoiler.h"
#include "BoundaryConditions/BoilerModel/BoundaryConditionsBoilerModel.h"
#include "BoundaryConditions/Dirichlet/BoundaryConditionsDirichlet.h"
#include "BoundaryConditions/Neumann/BoundaryConditionsNeumann.h"
#include "DifferentialOperatorsRightHandSide/KEpsilonRightHandSide/KEPsilonOperatorRightHandSide.h"

using namespace TNL;

typedef flowsBuildConfigTag BuildConfig;

/****
 * Uncomment the following (and comment the previous line) for the complete build.
 * This will include support for all floating point precisions, all indexing types
 * and more solvers. You may then choose between them from the command line.
 * The compile time may, however, take tens of minutes or even several hours,
 * especially if CUDA is enabled. Use this, if you want, only for the final build,
 * not in the development phase.
 */
//typedef tnlDefaultConfigTag BuildConfig;

template< typename ConfigTag >class twoEquationTurbulenceModelConfig
{
   public:
      static void configSetup( Config::ConfigDescription & config )
      {
         config.addDelimiter( "Inviscid flow settings:" );
         config.addEntry< String >( "boundary-conditions-type", "Choose the boundary conditions type.", "cavity");
            config.addEntryEnum< String >( "boiler" );
            config.addEntryEnum< String >( "boiler-model" );
            config.addEntryEnum< String >( "cavity" );
            config.addEntryEnum< String >( "dirichlet" );
            config.addEntryEnum< String >( "neumann" );
         config.addEntry< double >( "boundary-conditions-constant", "This sets a value in case of the constant boundary conditions." );
         config.addEntry< double >( "speed-increment", "This sets increment of input speed.", 0.0 );
         config.addEntry< double >( "speed-increment-until", "This sets time until input speed will rose", -0.1 );
         config.addEntry< double >( "start-speed", "This sets throttle speed at begining", 0.0 );
         config.addEntry< double >( "final-speed", "This sets speed at destined time", 0.0 );
         config.addEntry< double >( "speed-increment-until-h-throttle", "This sets time until input speed will rose for horizontal throttle", -0.1 );
         config.addEntry< double >( "start-speed-h-throttle", "This sets throttle speed at begining for horizontal throttle", 0.0 );
         config.addEntry< double >( "final-speed-h-throttle", "This sets speed at destined time for horizontal throttle", 0.0 );
         config.addEntry< double >( "cavity-speed", "This sets speed parameter of cavity", 0.0 );
         config.addEntry< double >( "turbulence-constant", "Value of turbulence constant", 1.0 );
         config.addEntry< double >( "viscosity-constant-1", "Value of viscosity constant C_epsilon_1", 1.0 );
         config.addEntry< double >( "viscosity-constant-2", "Value of viscosity constant C_epsilon_2", 1.0 );
         config.addEntry< double >( "sigma-k", "Value of sigma constant", 1.0 );
         config.addEntry< double >( "sigma-epsilon", "Value of sigma constant", 1.0 );
         config.addEntry< double >( "turbulence-intensity", "meassure of turbulence intensity", 1.0 );
         config.addEntry< double >( "turbulence-length-scale", "meassure of turbulence intensity", 1.0 );
         typedef Meshes::Grid< 3 > Mesh;
         AUSMPlus< Mesh >::configSetup( config, "inviscid-operators-" );
         RiemannProblemInitialCondition< Mesh >::configSetup( config );
         typedef Functions::Analytic::Constant< 3, double > Constant;
         BoundaryConditionsBoilerModel< Mesh, Constant >::configSetup( config, "boundary-conditions-" );
         /****
          * Add definition of your solver command line arguments.
          */

      }
};

template< typename Real,
          typename Device,
          typename Index,
          typename MeshType,
          typename ConfigTag,
          typename SolverStarter,
          typename CommunicatorType >
class twoEquationTurbulenceModelSetter
{
   public:

      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;

      static bool run( const Config::ParameterContainer & parameters )
      {
          enum { Dimension = MeshType::getMeshDimension() };

          typedef flowsRhs< MeshType, Real > RightHandSide;
          typedef Containers::StaticVector < MeshType::getMeshDimension(), Real > Point;
	  typedef KEpsilonOperatorRightHandSide< MeshType, Real, Index > OperatorRightHandSide;
          typedef AUSMPlus< MeshType, OperatorRightHandSide, Real, Index > ApproximateOperator;
          typedef Functions::Analytic::Constant< Dimension, Real > Constant;
          String boundaryConditionsType = parameters.getParameter< String >( "boundary-conditions-type" );
          if( boundaryConditionsType == "cavity" )
          {
             typedef BoundaryConditionsCavity< MeshType, Constant, Real, Index > BoundaryConditions;
             typedef twoEquationTurbulenceModelProblem< MeshType, BoundaryConditions, RightHandSide, CommunicatorType, ApproximateOperator > Problem;
             SolverStarter solverStarter;
             return solverStarter.template run< Problem >( parameters );
          }
          if( boundaryConditionsType == "boiler" )
          {
             typedef BoundaryConditionsBoiler< MeshType, Constant, Real, Index > BoundaryConditions;
             typedef twoEquationTurbulenceModelProblem< MeshType, BoundaryConditions, RightHandSide, CommunicatorType, ApproximateOperator > Problem;
             SolverStarter solverStarter;
             return solverStarter.template run< Problem >( parameters );
          } 
          if( boundaryConditionsType == "boiler-model" )
          {
             typedef BoundaryConditionsBoilerModel< MeshType, Constant, Real, Index > BoundaryConditions;
             typedef twoEquationTurbulenceModelProblem< MeshType, BoundaryConditions, RightHandSide, CommunicatorType, ApproximateOperator > Problem;
             SolverStarter solverStarter;
             return solverStarter.template run< Problem >( parameters );
          } 
          typedef Functions::MeshFunction< MeshType > MeshFunction;
          if( boundaryConditionsType == "dirichlet" )
          {
             typedef BoundaryConditionsDirichlet< MeshType, Constant, MeshType::getMeshDimension(), Real, Index > BoundaryConditions;
             typedef twoEquationTurbulenceModelProblem< MeshType, BoundaryConditions, RightHandSide, CommunicatorType, ApproximateOperator> Problem;
             SolverStarter solverStarter;
             return solverStarter.template run< Problem >( parameters );
          }
          if( boundaryConditionsType == "neumann" )
          {
             typedef BoundaryConditionsNeumann< MeshType, Constant, Real, Index > BoundaryConditions;
             typedef twoEquationTurbulenceModelProblem< MeshType, BoundaryConditions, RightHandSide, CommunicatorType, ApproximateOperator > Problem;
             SolverStarter solverStarter;
             return solverStarter.template run< Problem >( parameters );
          }              
      return true;}

};

int main( int argc, char* argv[] )
{
   Solvers::Solver< twoEquationTurbulenceModelSetter, twoEquationTurbulenceModelConfig, BuildConfig > solver;
   if( ! solver. run( argc, argv ) )
      return EXIT_FAILURE;
   return EXIT_SUCCESS;
};
