/***************************************************************************
                          twoEquationTurbulenceModelProblem_impl.h  -  description
                             -------------------
    begin                : Feb 13, 2017
    copyright            : (C) 2017 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/FileName.h>
#include <TNL/Matrices/MatrixSetter.h>
#include <TNL/Solvers/PDE/ExplicitUpdater.h>
#include <TNL/Solvers/PDE/LinearSystemAssembler.h>
#include <TNL/Solvers/PDE/BackwardTimeDiscretisation.h>
#include <TNL/Functions/Analytic/VectorNorm.h>

#include "RiemannProblemInitialCondition.h"
#include "CompressibleConservativeVariables.h"
#include "PhysicalVariablesGetter.h"
#include "twoEquationTurbulenceModelProblem.h"

namespace TNL {

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename Communicator,
          typename InviscidOperators >
String
twoEquationTurbulenceModelProblem< Mesh, BoundaryCondition, RightHandSide, Communicator, InviscidOperators >::
getType()
{
   return String( "twoEquationTurbulenceModelProblem< " ) + Mesh :: getType() + " >";
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename Communicator,
          typename InviscidOperators >
String
twoEquationTurbulenceModelProblem< Mesh, BoundaryCondition, RightHandSide, Communicator, InviscidOperators >::
getPrologHeader() const
{
   return String( "flow solver" );
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename Communicator,
          typename InviscidOperators >
void
twoEquationTurbulenceModelProblem< Mesh, BoundaryCondition, RightHandSide, Communicator, InviscidOperators >::
writeProlog( Logger& logger, const Config::ParameterContainer& parameters ) const
{
   /****
    * Add data you want to have in the computation report (log) as follows:
    * logger.writeParameter< double >( "Parameter description", parameter );
    */
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename Communicator,
          typename InviscidOperators >
bool
twoEquationTurbulenceModelProblem< Mesh, BoundaryCondition, RightHandSide, Communicator, InviscidOperators >::
setup( const Config::ParameterContainer& parameters,
       const String& prefix )
{
   if( ! this->inviscidOperatorsPointer->setup( this->getMesh(), parameters, prefix + "inviscid-operators-" ) ||
       ! this->boundaryConditionPointer->setup( this->getMesh(), parameters, prefix + "boundary-conditions-" ) ||
       ! this->rightHandSidePointer->setup( parameters, prefix + "right-hand-side-" ) )
      return false;
   this->gamma = parameters.getParameter< double >( "gamma" );
   this->startSpeed = parameters.getParameter< double >( "start-speed" );
   this->finalSpeed = parameters.getParameter< double >( "final-speed" );
   this->speedIncrementUntil = parameters.getParameter< RealType >( "speed-increment-until" );
   this->startSpeedHThrottle = parameters.getParameter< double >( "start-speed-h-throttle" );
   this->finalSpeedHThrottle = parameters.getParameter< double >( "final-speed-h-throttle" );
   this->speedIncrementUntilHThrottle = parameters.getParameter< RealType >( "speed-increment-until-h-throttle" );
   this->turbulenceConstant = parameters.getParameter< double >( "turbulence-constant" );
   this->beta = parameters.getParameter< double >( "beta" );
   this->betaStar = parameters.getParameter< double >( "beta-star" );
   this->alpha = parameters.getParameter< double >( "aplha" );
   this->sigmaK = parameters.getParameter< double >( "sigma-k" );
   this->sigmaEpsilon = parameters.getParameter< double >( "sigma-epsilon" );
   this->intensity = parameters.getParameter< double >( "turbulence-intensity" );
   this->lengthScale = parameters.getParameter< double >( "length-scale" );
   velocity->setMesh( this->getMesh() );
   pressure->setMesh( this->getMesh() );

   /****
    * Set-up operators
    */

   this->inviscidOperatorsPointer->setSigmaK( this->sigmaK );
   this->inviscidOperatorsPointer->setSigmaEpsilon( this->sigmaEpsilon );
   this->inviscidOperatorsPointer->setBetaStar( this->betaStar );
   this->inviscidOperatorsPointer->setDisipation( this->disipation );        
   this->inviscidOperatorsPointer->setVelocity( this->velocity );
   this->inviscidOperatorsPointer->setPressure( this->pressure );
   this->inviscidOperatorsPointer->setDensity( this->conservativeVariables->getDensity() );
   this->inviscidOperatorsPointer->setGamma( this->gamma );
   this->inviscidOperatorsPointer->setTurbulentViscosity( this->turbulentViscosity );
   this->inviscidOperatorsPointer->setTurbulentEnergy( this->turbulentEnergy );

   /****
    * Continuity equation
    */ 
   this->explicitUpdaterContinuity.setDifferentialOperator( this->inviscidOperatorsPointer->getContinuityOperator() );
   this->explicitUpdaterContinuity.setBoundaryConditions( this->boundaryConditionPointer->getDensityBoundaryCondition() );
   this->explicitUpdaterContinuity.setRightHandSide( this->rightHandSidePointer );

   /****
    * Momentum equations
    */
   this->explicitUpdaterMomentumX.setDifferentialOperator( this->inviscidOperatorsPointer->getMomentumXOperator() );
   this->explicitUpdaterMomentumX.setBoundaryConditions( this->boundaryConditionPointer->getMomentumXBoundaryCondition() );
   this->explicitUpdaterMomentumX.setRightHandSide( this->rightHandSidePointer );   

   if( Dimensions > 1 )
   {
      this->explicitUpdaterMomentumY.setDifferentialOperator( this->inviscidOperatorsPointer->getMomentumYOperator() );
      this->explicitUpdaterMomentumY.setBoundaryConditions( this->boundaryConditionPointer->getMomentumYBoundaryCondition() );
      this->explicitUpdaterMomentumY.setRightHandSide( this->rightHandSidePointer ); 
   }

   if( Dimensions > 2 )
   {
      this->explicitUpdaterMomentumZ.setDifferentialOperator( this->inviscidOperatorsPointer->getMomentumZOperator() );
      this->explicitUpdaterMomentumZ.setBoundaryConditions( this->boundaryConditionPointer->getMomentumZBoundaryCondition() );
      this->explicitUpdaterMomentumZ.setRightHandSide( this->rightHandSidePointer ); 
   }

   /****
    * Energy equation
    */
   this->explicitUpdaterEnergy.setDifferentialOperator( this->inviscidOperatorsPointer->getEnergyOperator() );
   this->explicitUpdaterEnergy.setBoundaryConditions( this->boundaryConditionPointer->getEnergyBoundaryCondition() );
   this->explicitUpdaterEnergy.setRightHandSide( this->rightHandSidePointer );            

    /****
    * Turbulent energy equation
    */
   this->explicitUpdaterTurbulentEnergy.setDifferentialOperator( this->inviscidOperatorsPointer->getTurbulentEnergyOperator() );
   this->explicitUpdaterTurbulentEnergy.setBoundaryConditions( this->boundaryConditionPointer->getTurbulentEnergyBoundaryCondition() );
   this->explicitUpdaterTurbulentEnergy.setRightHandSide( this->rightHandSidePointer );            

    /****
    * Disipation equation
    */
   this->explicitUpdaterDisipation.setDifferentialOperator( this->inviscidOperatorsPointer->getDisipationOperator() );
   this->explicitUpdaterDisipation.setBoundaryConditions( this->boundaryConditionPointer->getDisipationBoundaryCondition() );
   this->explicitUpdaterDisipation.setRightHandSide( this->rightHandSidePointer );            

   return true;
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename Communicator,
          typename InviscidOperators >
typename twoEquationTurbulenceModelProblem< Mesh, BoundaryCondition, RightHandSide, Communicator, InviscidOperators >::IndexType
twoEquationTurbulenceModelProblem< Mesh, BoundaryCondition, RightHandSide, Communicator, InviscidOperators >::
getDofs() const
{
   /****
    * Return number of  DOFs (degrees of freedom) i.e. number
    * of unknowns to be resolved by the main solver.
    */
   return this->conservativeVariables->getDofs( this->getMesh() );
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename Communicator,
          typename InviscidOperators >
void
twoEquationTurbulenceModelProblem< Mesh, BoundaryCondition, RightHandSide, Communicator, InviscidOperators >::
bindDofs( DofVectorPointer& dofVector )
{
   this->conservativeVariables->bind( this->getMesh(), dofVector );
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename Communicator,
          typename InviscidOperators >
bool
twoEquationTurbulenceModelProblem< Mesh, BoundaryCondition, RightHandSide, Communicator, InviscidOperators >::
setInitialCondition( const Config::ParameterContainer& parameters,
                     DofVectorPointer& dofs )
{
   CompressibleConservativeVariables< MeshType > conservativeVariables;
   conservativeVariables.bind( this->getMesh(), dofs );
   const String& initialConditionType = parameters.getParameter< String >( "initial-condition" );
   this->speedIncrementUntil = parameters.getParameter< RealType >( "speed-increment-until" );
   this->speedIncrement = parameters.getParameter< RealType >( "speed-increment" );
   this->cavitySpeed = parameters.getParameter< RealType >( "cavity-speed" );
   if( initialConditionType == "riemann-problem" )
   {
      RiemannProblemInitialCondition< MeshType > initialCondition;
      if( ! initialCondition.setup( parameters ) )
         return false;
      initialCondition.setInitialCondition( conservativeVariables );
      return true;
   }
   std::cerr << "Unknown initial condition " << initialConditionType << std::endl;
   return false;
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename Communicator,
          typename InviscidOperators >
   template< typename Matrix >
bool
twoEquationTurbulenceModelProblem< Mesh, BoundaryCondition, RightHandSide, Communicator, InviscidOperators >::
setupLinearSystem( Matrix& matrix )
{
/*   const IndexType dofs = this->getDofs();
   typedef typename Matrix::CompressedRowLengthsVector CompressedRowLengthsVectorType;
   CompressedRowLengthsVectorType rowLengths;
   if( ! rowLengths.setSize( dofs ) )
      return false;
   MatrixSetter< MeshType, DifferentialOperator, BoundaryCondition, CompressedRowLengthsVectorType > matrixSetter;
   matrixSetter.template getCompressedRowLengths< typename Mesh::Cell >( mesh,
                                                                          differentialOperator,
                                                                          boundaryCondition,
                                                                          rowLengths );
   matrix.setDimensions( dofs, dofs );
   if( ! matrix.setCompressedRowLengths( rowLengths ) )
      return false;*/
   return true;
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename Communicator,
          typename InviscidOperators >
bool
twoEquationTurbulenceModelProblem< Mesh, BoundaryCondition, RightHandSide, Communicator, InviscidOperators >::
makeSnapshot( const RealType& time,
              const IndexType& step,
              DofVectorPointer& dofs )
{
  std::cout << std::endl << "Writing output at time " << time << " step " << step << "." << std::endl;
  
  this->bindDofs( dofs );
  PhysicalVariablesGetter< MeshType > physicalVariablesGetter;
  physicalVariablesGetter.getVelocity( this->conservativeVariables, this->velocity );
  physicalVariablesGetter.getPressure( this->conservativeVariables, this->gamma, this->pressure );
  
   FileName fileName;
   fileName.setExtension( "tnl" );
   fileName.setIndex( step );
   fileName.setFileNameBase( "density-" );
//   if( ! this->conservativeVariables->getDensity()->save( fileName.getFileName() ) )
   this->conservativeVariables->getDensity()->save( fileName.getFileName() );
//      return false;
   
   fileName.setFileNameBase( "velocity-" );
//   if( ! this->velocity->save( fileName.getFileName() ) )
   this->velocity->save( fileName.getFileName() );
//      return false;

   fileName.setFileNameBase( "pressure-" );
//   if( ! this->pressure->save( fileName.getFileName() ) )
   this->pressure->save( fileName.getFileName() );
//      return false;

   fileName.setFileNameBase( "energy-" );
//   if( ! this->conservativeVariables->getEnergy()->save( fileName.getFileName() ) )
   this->conservativeVariables->getEnergy()->save( fileName.getFileName() );
//      return false;
   return true;
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename Communicator,
          typename InviscidOperators >
void
twoEquationTurbulenceModelProblem< Mesh, BoundaryCondition, RightHandSide, Communicator, InviscidOperators >::
getExplicitUpdate( const RealType& time,
                   const RealType& tau,
                   DofVectorPointer& _u,
                   DofVectorPointer& _fu )
{
    typedef typename MeshType::Cell Cell;
    
    /****
     * Bind DOFs
     */
    this->conservativeVariables->bind( this->getMesh(), _u );
    this->conservativeVariablesRHS->bind( this->getMesh(), _fu );
    
    /****
     * Resolve the physical variables
     */
    PhysicalVariablesGetter< typename MeshPointer::ObjectType > physicalVariables;
    physicalVariables.getVelocity( this->conservativeVariables, this->velocity );
    physicalVariables.getPressure( this->conservativeVariables, this->gamma, this->pressure );
    physicalVariables.getTurbulentEnergy( this->conservativeVariables, this->turbulentEnergy );
    physicalVariables.getDisipation( this->conservativeVariables, this->disipation );
//    physicalVariables.getTurbulentViscosity( this->conservativeVariables, this->turbulentEnergy, this->disipation, this->turbulenceConstant, this->turbulentViscosity );

   /****
    * Set-up operators
    */
       
   this->inviscidOperatorsPointer->setTau( tau );
    
   /****
    * Continuity equation
    */ 
   this->explicitUpdaterContinuity.template update< typename Mesh::Cell, Communicator >( time, tau, this->getMesh(), 
                                                                     this->conservativeVariables->getDensity(),
                                                                     this->conservativeVariablesRHS->getDensity() );

//   this->explicitUpdaterContinuity.template applyBoundaryConditions< typename Mesh::Cell >( this->getMesh(), time, 
//                                                                                this->conservativeVariables->getDensity() );

   /****
    * Momentum equations
    */ 
   this->explicitUpdaterMomentumX.template update< typename Mesh::Cell, Communicator >( time, tau, this->getMesh(),
                                                           ( *this->conservativeVariables->getMomentum() )[ 0 ], // uRhoVelocityX,
                                                           ( *this->conservativeVariablesRHS->getMomentum() )[ 0 ] ); //, fuRhoVelocityX );
   if( Dimensions > 1 )
   {    
      this->explicitUpdaterMomentumY.template update< typename Mesh::Cell, Communicator >( time, tau, this->getMesh(),
                                                              ( *this->conservativeVariables->getMomentum() )[ 1 ], // uRhoVelocityX,
                                                              ( *this->conservativeVariablesRHS->getMomentum() )[ 1 ] ); //, fuRhoVelocityX );
   }

   if( Dimensions > 2 )
   {     
      this->explicitUpdaterMomentumZ.template update< typename Mesh::Cell, Communicator >( time, tau, this->getMesh(),
                                                              ( *this->conservativeVariables->getMomentum() )[ 2 ], // uRhoVelocityX,
                                                              ( *this->conservativeVariablesRHS->getMomentum() )[ 2 ] ); //, fuRhoVelocityX );
   }
   
   /****
    * Energy equation
    */               
   this->explicitUpdaterEnergy.template update< typename Mesh::Cell, Communicator >( time, tau, this->getMesh(),
                                                           this->conservativeVariables->getEnergy(), // uRhoVelocityX,
                                                           this->conservativeVariablesRHS->getEnergy() ); //, fuRhoVelocityX );


   /****
    * Turbulent energy equation
    */               
   this->explicitUpdaterTurbulentEnergy.template update< typename Mesh::Cell, Communicator >( time, tau, this->getMesh(),
                                                           this->conservativeVariables->getTurbulentEnergy(), // uRhoVelocityX,
                                                           this->conservativeVariablesRHS->getTurbulentEnergy() ); //, fuRhoVelocityX );
   
   /****
    * Disipation equation
    */               
   this->explicitUpdaterTurbulentEnergy.template update< typename Mesh::Cell, Communicator >( time, tau, this->getMesh(),
                                                           this->conservativeVariables->getDisipation(), // uRhoVelocityX,
                                                           this->conservativeVariablesRHS->getDisipation() ); //, fuRhoVelocityX );
   
   /*this->conservativeVariablesRHS->getDensity()->write( "density", "gnuplot" );
   this->conservativeVariablesRHS->getEnergy()->write( "energy", "gnuplot" );
   this->conservativeVariablesRHS->getMomentum()->write( "momentum", "gnuplot", 0.05 );
   getchar();*/

}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename Communicator,
          typename InviscidOperators >
void
twoEquationTurbulenceModelProblem< Mesh, BoundaryCondition, RightHandSide, Communicator, InviscidOperators >::
applyBoundaryConditions( const RealType& time,
                         DofVectorPointer& dofs )
{
    /****
     * Update Boundary Conditions
     */
    if(this->speedIncrementUntil > time )
    {
       this->boundaryConditionPointer->setTimestep(this->speedIncrement);
    }
    else
    {
       this->boundaryConditionPointer->setTimestep(0);
    }
    this->boundaryConditionPointer->setIntensity(this->intensity);
    this->boundaryConditionPointer->setLengthScale(this->lengthScale);
    this->boundaryConditionPointer->setTurbulenceConstant(this->turbulenceConstant);
    this->boundaryConditionPointer->setSpeed(this->cavitySpeed);
    this->boundaryConditionPointer->setCompressibleConservativeVariables(this->conservativeVariables);
    this->boundaryConditionPointer->setGamma(this->gamma);
    this->boundaryConditionPointer->setPressure(this->pressure);
    this->boundaryConditionPointer->setVerticalThrottleSpeed( startSpeed, finalSpeed, time, speedIncrementUntil );
    this->boundaryConditionPointer->setHorizontalThrottleSpeed( startSpeedHThrottle, finalSpeedHThrottle, time, speedIncrementUntilHThrottle );
    /****
     * Bind DOFs
     */
    this->conservativeVariables->bind( this->getMesh(), dofs );
//   this->conservativeVariables->getDensity()->write( "density", "gnuplot" );
//   this->conservativeVariables->getEnergy()->write( "energy", "gnuplot" );
//   this->conservativeVariables->getMomentum()->write( "momentum", "gnuplot", 0.05 );
//   dofs->save("dofs.tnl");
//   getchar();
//   std::cout <<"applyBCC" << std::endl;
   /****
    * Continuity equation
    */ 
   this->explicitUpdaterContinuity.template applyBoundaryConditions< typename Mesh::Cell >( this->getMesh(), time, 
                                                                                this->conservativeVariables->getDensity() );
   /****
    * Momentum equations
    */ 
   this->explicitUpdaterMomentumX.template applyBoundaryConditions< typename Mesh::Cell >( this->getMesh(), time,
                                                                               ( *this->conservativeVariables->getMomentum() )[ 0 ] ); // uRhoVelocityX,
   if( Dimensions > 1 )
   {   

      this->explicitUpdaterMomentumY.template applyBoundaryConditions< typename Mesh::Cell >( this->getMesh(), time,
                                                              ( *this->conservativeVariables->getMomentum() )[ 1 ] ); // uRhoVelocityX,
   }

   if( Dimensions > 2 )
   {           
      this->explicitUpdaterMomentumZ.template applyBoundaryConditions< typename Mesh::Cell >( this->getMesh(), time,
                                                              ( *this->conservativeVariables->getMomentum() )[ 2 ] ); // uRhoVelocityX,
   }
   
   /****
    * Energy equation
    */               
   this->explicitUpdaterEnergy.template applyBoundaryConditions< typename Mesh::Cell >( this->getMesh(), time,
                                                           this->conservativeVariables->getEnergy() ); // uRhoVelocityX,
   
   /****
    * Turbulent energy equation
    */               
   this->explicitUpdaterTurbulentEnergy.template applyBoundaryConditions< typename Mesh::Cell >( this->getMesh(), time,
                                                           this->conservativeVariables->getTurbulentEnergy() ); // uRhoVelocityX,
   /****
    * Turbulent energy equation
    */               
   this->explicitUpdaterDisipation.template applyBoundaryConditions< typename Mesh::Cell >( this->getMesh(), time,
                                                           this->conservativeVariables->getDisipation() ); // uRhoVelocityX,


} 

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename Communicator,
          typename InviscidOperators >
   template< typename Matrix >
void
twoEquationTurbulenceModelProblem< Mesh, BoundaryCondition, RightHandSide, Communicator, InviscidOperators >::
assemblyLinearSystem( const RealType& time,
                      const RealType& tau,
                      DofVectorPointer& _u,
                      Matrix& matrix,
                      DofVectorPointer& b )
{
/*   LinearSystemAssembler< Mesh,
                             MeshFunctionType,
                             InviscidOperators,
                             BoundaryCondition,
                             RightHandSide,
                             BackwardTimeDiscretisation,
                             Matrix,
                             DofVectorType > systemAssembler;

   MeshFunction< Mesh > u( mesh, _u );
   systemAssembler.template assembly< typename Mesh::Cell >( time,
                                                             tau,
                                                             this->differentialOperator,
                                                             this->boundaryCondition,
                                                             this->rightHandSide,
                                                             u,
                                                             matrix,
                                                             b );*/
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename Communicator,
          typename InviscidOperators >
bool
twoEquationTurbulenceModelProblem< Mesh, BoundaryCondition, RightHandSide, Communicator, InviscidOperators >::
postIterate( const RealType& time,
             const RealType& tau,
             DofVectorPointer& dofs )
{
   /*
    typedef typename MeshType::Cell Cell;
    int count = mesh->template getEntitiesCount< Cell >()/4;
	//bind _u
    this->_uRho.bind( *dofs, 0, count);
    this->_uRhoVelocityX.bind( *dofs, count, count);
    this->_uRhoVelocityY.bind( *dofs, 2 * count, count);
    this->_uEnergy.bind( *dofs, 3 * count, count);

   MeshFunctionType velocity( mesh, this->velocity );
   MeshFunctionType velocityX( mesh, this->velocityX );
   MeshFunctionType velocityY( mesh, this->velocityY );
   MeshFunctionType pressure( mesh, this->pressure );
   MeshFunctionType uRho( mesh, _uRho ); 
   MeshFunctionType uRhoVelocityX( mesh, _uRhoVelocityX ); 
   MeshFunctionType uRhoVelocityY( mesh, _uRhoVelocityY ); 
   MeshFunctionType uEnergy( mesh, _uEnergy ); 
   //Generating differential operators
   Velocity euler2DVelocity;
   VelocityX euler2DVelocityX;
   VelocityY euler2DVelocityY;
   Pressure euler2DPressure;

   //velocityX
   euler2DVelocityX.setRhoVelX(uRhoVelocityX);
   euler2DVelocityX.setRho(uRho);
//   OperatorFunction< VelocityX, MeshFunction, void, true > OFVelocityX;
//   velocityX = OFVelocityX;

   //velocityY
   euler2DVelocityY.setRhoVelY(uRhoVelocityY);
   euler2DVelocityY.setRho(uRho);
//   OperatorFunction< VelocityY, MeshFunction, void, time > OFVelocityY;
//   velocityY = OFVelocityY;

   //velocity
   euler2DVelocity.setVelX(velocityX);
   euler2DVelocity.setVelY(velocityY);
//   OperatorFunction< Velocity, MeshFunction, void, time > OFVelocity;
//   velocity = OFVelocity;

   //pressure
   euler2DPressure.setGamma(gamma);
   euler2DPressure.setVelocity(velocity);
   euler2DPressure.setEnergy(uEnergy);
   euler2DPressure.setRho(uRho);
//   OperatorFunction< euler2DPressure, MeshFunction, void, time > OFPressure;
//   pressure = OFPressure;
    */
   return true;
}

} // namespace TNL

