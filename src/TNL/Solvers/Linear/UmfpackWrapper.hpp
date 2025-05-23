// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include "UmfpackWrapper.h"

#include <TNL/Solvers/Linear/Utils/LinearResidueGetter.h>

namespace TNL::Solvers::Linear {

template< typename Matrix, typename SolverMonitor >
void
UmfpackWrapper< Matrix, SolverMonitor >::setMatrix( const MatrixPointer& matrix )
{
#ifdef HAVE_UMFPACK
   if( matrix->getRows() != matrix->getColumns() )
      throw std::invalid_argument( "UmfpackWrapper::solve: matrix must be square" );

   this->matrix = matrix;

   const IndexType size = this->matrix->getRows();

   int status = UMFPACK_OK;

   // The solver does not work without calling umfpack_di_defaults
   umfpack_di_defaults( Control );

   bool symbolic_fail = false;
   bool numeric_fail = false;
   this->factorized = false;
   // symbolic reordering of the sparse matrix
   status = umfpack_di_symbolic( size,
                                 size,
                                 this->matrix->getSegments().getOffsets().getData(),
                                 this->matrix->getColumnIndexes().getData(),
                                 this->matrix->getValues().getData(),
                                 &Symbolic,
                                 Control,
                                 Info );
   if( status != UMFPACK_OK ) {
      symbolic_fail = true;
   }

   // numeric factorization
   if( ! symbolic_fail ) {
      status = umfpack_di_numeric( this->matrix->getSegments().getOffsets().getData(),
                                   this->matrix->getColumnIndexes().getData(),
                                   this->matrix->getValues().getData(),
                                   Symbolic,
                                   &Numeric,
                                   Control,
                                   Info );
      if( status != UMFPACK_OK ) {
         numeric_fail = true;
      }
   }

   if( status != UMFPACK_OK ) {
      // increase print level for reports
      Control[ UMFPACK_PRL ] = 2;
      umfpack_di_report_status( Control, status );
      //        umfpack_di_report_control( Control );
      //        umfpack_di_report_info( Control, Info );
      if( symbolic_fail )
         throw std::runtime_error( "Umfpack symbolic factorization failed." );
      if( numeric_fail )
         throw std::runtime_error( "Umfpack numeric factorization failed." );
      if( this->Symbolic )
         umfpack_di_free_symbolic( &Symbolic );
      if( this->Numeric )
         umfpack_di_free_numeric( &Numeric );
   }
   this->factorized = true;
#else
   std::cerr << "Umfpack is not available." << std::endl;
#endif
}

template< typename Matrix, typename SolverMonitor >
bool
UmfpackWrapper< Matrix, SolverMonitor >::solve( ConstVectorViewType b, VectorViewType x )
{
#ifdef HAVE_UMFPACK
   if( this->matrix->getColumns() != x.getSize() )
      throw std::invalid_argument( "UmfpackWrapper::solve: wrong size of the solution vector" );
   if( this->matrix->getColumns() != b.getSize() )
      throw std::invalid_argument( "UmfpackWrapper::solve: wrong size of the right hand side" );

   if( ! this->factorized ) {
      std::cerr << "The solver is not ready for solving." << std::endl;
      return false;
   }

   int status = UMFPACK_OK;

   this->setResidue( std::numeric_limits< RealType >::max() );

   RealType bNorm = lpNorm( b, (RealType) 2.0 );

   // umfpack expects Compressed Sparse Column format, we have Compressed Sparse Row
   // so we need to solve  A^T * x = rhs
   int system_type = UMFPACK_Aat;

   // solve with specified right-hand-side
   status = umfpack_di_solve( system_type,
                              this->matrix->getSegments().getOffsets().getData(),
                              this->matrix->getColumnIndexes().getData(),
                              this->matrix->getValues().getData(),
                              x.getData(),
                              b.getData(),
                              Numeric,
                              Control,
                              Info );
   if( status != UMFPACK_OK ) {
      std::cerr << "Umfpack solver failed." << std::endl;
      return false;
   }
   this->solver_success = true;

   this->setResidue( LinearResidueGetter::getResidue( *this->matrix, x, b, bNorm ) );
   return true;
#else
   std::cerr << "Umfpack is not available." << std::endl;
   return false;
#endif
}

template< typename Matrix, typename SolverMonitor >
bool
UmfpackWrapper< Matrix, SolverMonitor >::succeeded() const
{
   return this->solver_success;
}

template< typename Matrix, typename SolverMonitor >
UmfpackWrapper< Matrix, SolverMonitor >::~UmfpackWrapper()
{
#ifdef HAVE_UMFPACK
   if( this->Symbolic )
      umfpack_di_free_symbolic( &Symbolic );
   if( this->Numeric )
      umfpack_di_free_numeric( &Numeric );
#endif
}

}  // namespace TNL::Solvers::Linear
