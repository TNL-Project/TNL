/***************************************************************************
                          transportEquationProblem.h  -  description
                             -------------------
    begin                : Feb 10, 2017
    copyright            : (C) 2017 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Problems/PDEProblem.h>
#include <TNL/Functions/MeshFunction.h>
#include <TNL/Pointers/SharedPointer.h>

using namespace TNL::Problems;

namespace TNL {

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename Communicator,
          typename DifferentialOperator >
class transportEquationProblem:
public PDEProblem< Mesh,
                   Communicator,
                   typename DifferentialOperator::RealType,
                   typename Mesh::DeviceType,
                   typename DifferentialOperator::IndexType >
{
   public:

      typedef typename DifferentialOperator::RealType RealType;
      typedef typename Mesh::DeviceType DeviceType;
      typedef typename DifferentialOperator::IndexType IndexType;
      typedef Functions::MeshFunction< Mesh > MeshFunctionType;
      typedef PDEProblem< Mesh, Communicator, RealType, DeviceType, IndexType > BaseType;
      typedef Pointers::SharedPointer<  MeshFunctionType, DeviceType > MeshFunctionPointer;
      typedef Pointers::SharedPointer<  DifferentialOperator > DifferentialOperatorPointer;
      typedef Pointers::SharedPointer<  BoundaryCondition > BoundaryConditionPointer;
      typedef Pointers::SharedPointer<  RightHandSide, DeviceType > RightHandSidePointer;
      typedef typename DifferentialOperator::VelocityFieldType VelocityFieldType;
      typedef Pointers::SharedPointer<  VelocityFieldType, DeviceType > VelocityFieldPointer;

      typedef Communicator CommunicatorType;

      using typename BaseType::MeshType;
      using typename BaseType::MeshPointer;
      using typename BaseType::DofVectorType;
      using typename BaseType::DofVectorPointer;

      static String getType();

      String getPrologHeader() const;

      void writeProlog( Logger& logger,
                        const Config::ParameterContainer& parameters ) const;

      bool setup( const Config::ParameterContainer& parameters,
                  const String& prefix = "" );

      bool setInitialCondition( const Config::ParameterContainer& parameters,
                                DofVectorPointer& dofs );

      template< typename Matrix >
      bool setupLinearSystem( Matrix& matrix );

      bool makeSnapshot( const RealType& time,
                         const IndexType& step,
                         DofVectorPointer& dofs );

      IndexType getDofs() const;

      void bindDofs( DofVectorPointer& dofs );

      void getExplicitUpdate( const RealType& time,
                              const RealType& tau,
                              DofVectorPointer& _u,
                              DofVectorPointer& _fu );
      
      void applyBoundaryConditions( const RealType& time,
                                       DofVectorPointer& dofs );      

      template< typename Matrix >
      void assemblyLinearSystem( const RealType& time,
                                 const RealType& tau,
                                 DofVectorPointer& dofs,
                                 Matrix& matrix,
                                 DofVectorPointer& rightHandSide );

   protected:

      MeshFunctionPointer uPointer, velocityX, velocityY, velocityZ;

      DifferentialOperatorPointer differentialOperatorPointer;

      BoundaryConditionPointer boundaryConditionPointer;

      RightHandSidePointer rightHandSidePointer;
      
      VelocityFieldPointer velocityField;
      
      int dimension;
      String choice;
      RealType size;
      long step = 0;
      MeshFunctionType analyt;
      RealType speedX;
      RealType speedY;
      RealType speedZ;
      RealType schemeSize;      
};

} // namespace TNL

#include "transportEquationProblem_impl.h"
