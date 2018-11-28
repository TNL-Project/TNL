#include <TNL/tnlConfig.h>
#include <TNL/Solvers/Solver.h>
#include <TNL/Solvers/BuildConfigTags.h>
#include <TNL/Operators/DirichletBoundaryConditions.h>
#include <TNL/Operators/NeumannBoundaryConditions.h>
#include <TNL/Functions/Analytic/Constant.h>
#include "navierStokesProblem.h"
#include "DifferentialOperators/Navier-Stokes/Lax-Friedrichs/LaxFridrichs.h"
#include "DifferentialOperators/Navier-Stokes/Steger-Warming/StegerWarming.h"
#include "DifferentialOperators/Navier-Stokes/VanLeer/VanLeer.h"
#include "DifferentialOperators/Navier-Stokes/AUSM+/AUSMPlus.h"
#include "flowsRhs.h"
#include "flowsBuildConfigTag.h"

#include "RiemannProblemInitialCondition.h"
#include "BoundaryConditions/Cavity/BoundaryConditionsCavity.h"
#include "BoundaryConditions/Boiler/BoundaryConditionsBoiler.h"
#include "BoundaryConditions/BoilerModel/BoundaryConditionsBoilerModel.h"
#include "BoundaryConditions/Dirichlet/BoundaryConditionsDirichlet.h"
#include "BoundaryConditions/Neumann/BoundaryConditionsNeumann.h"
#include "DifferentialOperatorsRightHandSide/NavierStokesRightHandSide/NavierStokesOperatorRightHandSide.h"
#include "DifferentialOperatorsRightHandSide/nullRightHandSide/nullOperatorRightHandSide.h"

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

template< typename ConfigTag >class navierStokesConfig
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
         config.addEntry< String >( "differential-operator", "Choose the differential operator.", "Lax-Friedrichs");
            config.addEntryEnum< String >( "Lax-Friedrichs" );
            config.addEntryEnum< String >( "Steger-Warming" );
            config.addEntryEnum< String >( "VanLeer" );
            config.addEntryEnum< String >( "AUSMPlus" );
         config.addEntry< String >( "operator-right-hand-side", "Choose equation type.", "Euler");
            config.addEntryEnum< String >( "Euler" );
            config.addEntryEnum< String >( "Navier-Stokes" );
         config.addEntry< double >( "boundary-conditions-constant", "This sets a value in case of the constant boundary conditions." );
         config.addEntry< double >( "speed-increment", "This sets increment of input speed.", 0.0 );
         config.addEntry< double >( "speed-increment-until", "This sets time until input speed will rose", -0.1 );
         config.addEntry< double >( "cavity-speed", "This sets speed parameter of cavity", 0.0 );
         typedef Meshes::Grid< 3 > Mesh;
         LaxFridrichs< Mesh >::configSetup( config, "inviscid-operators-" );
         RiemannProblemInitialCondition< Mesh >::configSetup( config );

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
class navierStokesSetter
{
   public:

      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;

      static bool run( const Config::ParameterContainer & parameters )
      {
          enum { Dimension = MeshType::getMeshDimension() };
	  typedef NullOperatorRightHandSide< MeshType, Real, Index > OperatorRightHandSide;
          typedef LaxFridrichs< MeshType, OperatorRightHandSide, Real, Index > ApproximateOperator;
          typedef flowsRhs< MeshType, Real > RightHandSide;
          typedef Containers::StaticVector < MeshType::getMeshDimension(), Real > Point;
	  String operatorRightHandSideType = parameters.getParameter< String >( "operator-right-hand-side");
	  if( operatorRightHandSideType == "Euler" )
	     typedef NullOperatorRightHandSide< MeshType, Real, Index > OperatorRightHandSide;
 	  else if( operatorRightHandSideType == "Navier-Stokes" )
	     typedef NavierStokesOperatorRightHandSide< MeshType, Real, Index > OperatorRightHandSide;
	  String differentialOperatorType = parameters.getParameter< String >( "differential-operator");
	  if( differentialOperatorType == "Lax-Friedrichs" )
	     typedef LaxFridrichs< MeshType, OperatorRightHandSide, Real, Index > ApproximateOperator;
          else if( differentialOperatorType == "Steger-Warming" )
	     typedef StegerWarming< MeshType, OperatorRightHandSide, Real, Index > ApproximateOperator;
          else if( differentialOperatorType == "VanLeer" )
	     typedef VanLeer< MeshType, OperatorRightHandSide, Real, Index > ApproximateOperator;
          else if( differentialOperatorType == "AUSMPlus" )
	     typedef AUSMPlus< MeshType, OperatorRightHandSide, Real, Index > ApproximateOperator;

         /****
          * Resolve the template arguments of your solver here.
          * The following code is for the Dirichlet and the Neumann boundary conditions.
          * Both can be constant or defined as descrete values of Vector.
          */

          typedef Functions::Analytic::Constant< Dimension, Real > Constant;
          String boundaryConditionsType = parameters.getParameter< String >( "boundary-conditions-type" );
          if( boundaryConditionsType == "cavity" )
             {
                typedef BoundaryConditionsCavity< MeshType, Constant, Real, Index > BoundaryConditions;
                typedef navierStokesProblem< MeshType, BoundaryConditions, RightHandSide, CommunicatorType, ApproximateOperator > Problem;
                SolverStarter solverStarter;
                return solverStarter.template run< Problem >( parameters );
             }
           if( boundaryConditionsType == "boiler" )
             {
                typedef BoundaryConditionsBoiler< MeshType, Constant, Real, Index > BoundaryConditions;
                typedef navierStokesProblem< MeshType, BoundaryConditions, RightHandSide, CommunicatorType, ApproximateOperator > Problem;
                SolverStarter solverStarter;
                return solverStarter.template run< Problem >( parameters );
             } 
           if( boundaryConditionsType == "boiler-model" )
             {
                typedef BoundaryConditionsBoilerModel< MeshType, Constant, Real, Index > BoundaryConditions;
                typedef navierStokesProblem< MeshType, BoundaryConditions, RightHandSide, CommunicatorType, ApproximateOperator > Problem;
                SolverStarter solverStarter;
                return solverStarter.template run< Problem >( parameters );
             } 
          typedef Functions::MeshFunction< MeshType > MeshFunction;
          if( boundaryConditionsType == "dirichlet" )
          {
             typedef BoundaryConditionsDirichlet< MeshType, Constant, MeshType::getMeshDimension(), Real, Index > BoundaryConditions;
             typedef navierStokesProblem< MeshType, BoundaryConditions, RightHandSide, CommunicatorType, ApproximateOperator> Problem;
             SolverStarter solverStarter;
             return solverStarter.template run< Problem >( parameters );
          }
          if( boundaryConditionsType == "neumann" )
          {
             typedef BoundaryConditionsNeumann< MeshType, Constant, Real, Index > BoundaryConditions;
             typedef navierStokesProblem< MeshType, BoundaryConditions, RightHandSide, CommunicatorType, ApproximateOperator > Problem;
             SolverStarter solverStarter;
             return solverStarter.template run< Problem >( parameters );
          }      

      return true;}

};

int main( int argc, char* argv[] )
{
   Solvers::Solver< navierStokesSetter, navierStokesConfig, BuildConfig > solver;
   if( ! solver. run( argc, argv ) )
      return EXIT_FAILURE;
   return EXIT_SUCCESS;
};
