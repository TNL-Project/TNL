// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Containers/Vector.h>

namespace TNL::Solvers::Optimization {

/**
 * \brief This structure represents a linear programming problem.
 *
 * The problem has the following form:
 *
 * min c^T x
 * s.t. A_1 x >= b_1
 *      A_2 x = b_2
 *      l <= x <= u
 * The constraint matrix A is given as:
 *
 * A = | A_1 |
 *     | A_2 |
 *
 * The constraint vector b is given as:
 * b = | b_1 |
 *     | b_2 |
 * \tparam Matrix The type of the constraint matrix.
 */
template< typename Matrix >
struct LPProblem
{
   using MatrixType = Matrix;
   using RealType = typename MatrixType::RealType;
   using DeviceType = typename MatrixType::DeviceType;
   using IndexType = typename MatrixType::IndexType;
   using VectorType = Containers::Vector< RealType, DeviceType, IndexType >;

   LPProblem() = default;

   /**
    * \brief Constructor of a new LPProblem object with all parameters.
    *
    * \param constraintMatrix is the constraint matrix A.
    * \param constraintVector is the constraint vector b.
    * \param inequalityCount is the number of inequality constraints and defines the number of rows of the matrix A_1.
    * \param objectiveFunction is the vector c. Its size must be equal to the number of columns of the constraint matrix.
    * \param lowerBounds is the vector l. Its size must be equal to the size of the objective function.
    * \param upperBounds is the vector u. Its size must be equal to the size of the objective function.
    */
   LPProblem( const MatrixType& constraintMatrix,
              const VectorType& constraintVector,
              const IndexType inequalityCount,
              const VectorType& objectiveFunction,
              const VectorType& lowerBounds,
              const VectorType& upperBounds )
   : constraintMatrix( constraintMatrix ), constraintVector( constraintVector ), objectiveFunction( objectiveFunction ),
     lowerBounds( lowerBounds ), upperBounds( upperBounds ), inequalityCount( inequalityCount )
   {
      TNL_ASSERT_EQ( constraintMatrix.getRows(), constraintVector.getSize(), "" );
      TNL_ASSERT_EQ( constraintMatrix.getColumns(), objectiveFunction.getSize(), "" );
      TNL_ASSERT_EQ( lowerBounds.getSize(), objectiveFunction.getSize(), "" );
      TNL_ASSERT_EQ( upperBounds.getSize(), objectiveFunction.getSize(), "" );
      TNL_ASSERT_GE( inequalityCount, 0, "" );
      TNL_ASSERT_LE( inequalityCount, constraintMatrix.getRows(), "" );
   }

   /**
    * \brief Constructor of a new LPProblem object with only the constraint matrix and vector.
    *
    * The objective function, lower and upper bounds are set to zero.
    *
    * \param constraintMatrix is the constraint matrix A.
    * \param constraintVector is the constraint vector b.
    * \param inequalityCount is the number of inequality constraints and defines the number of rows of the matrix A_1.
    */
   LPProblem( const MatrixType& constraintMatrix,
              const VectorType& constraintVector,
              const IndexType inequalityCount,
              const VectorType& objectiveFunction )
   : LPProblem( constraintMatrix,
                constraintVector,
                inequalityCount,
                objectiveFunction,
                VectorType( constraintMatrix.getColumns(), -std::numeric_limits< RealType >::infinity() ),
                VectorType( constraintMatrix.getColumns(), std::numeric_limits< RealType >::infinity() ) )
   {}

   MatrixType&
   getConstraintMatrix()
   {
      return constraintMatrix;
   }

   const MatrixType&
   getConstraintMatrix() const
   {
      return constraintMatrix;
   }

   VectorType&
   getConstraintVector()
   {
      return constraintVector;
   }

   const VectorType&
   getConstraintVector() const
   {
      return constraintVector;
   }

   VectorType&
   getObjectiveFunction()
   {
      return objectiveFunction;
   }

   const VectorType&
   getObjectiveFunction() const
   {
      return objectiveFunction;
   }

   VectorType&
   getLowerBounds()
   {
      return lowerBounds;
   }

   const VectorType&
   getLowerBounds() const
   {
      return lowerBounds;
   }

   VectorType&
   getUpperBounds()
   {
      return upperBounds;
   }

   const VectorType&
   getUpperBounds() const
   {
      return upperBounds;
   }

   IndexType
   getInequalityCount() const
   {
      return inequalityCount;
   }

   IndexType
   getVariableCount() const
   {
      return constraintMatrix.getColumns();
   }

protected:
   MatrixType constraintMatrix;

   VectorType constraintVector, objectiveFunction, lowerBounds, upperBounds;

   IndexType inequalityCount;
};

}  // namespace TNL::Solvers::Optimization
