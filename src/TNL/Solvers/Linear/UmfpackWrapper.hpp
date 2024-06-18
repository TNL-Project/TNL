// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#ifdef HAVE_UMFPACK

   #include "UmfpackWrapper.h"

   #include <TNL/Solvers/Linear/Utils/LinearResidueGetter.h>

namespace TNL::Solvers::Linear {

bool
UmfpackWrapper< CSRMatrix< double, Devices::Host, int > >::solve( ConstVectorViewType b, VectorViewType x )
{
   if( this->matrix->getRows() != this->matrix->getColumns() )
      throw std::invalid_argument( "UmfpackWrapper::solve: matrix must be square" );
   if( this->matrix->getColumns() != x.getSize() )
      throw std::invalid_argument( "UmfpackWrapper::solve: wrong size of the solution vector" );
   if( this->matrix->getColumns() != b.getSize() )
      throw std::invalid_argument( "UmfpackWrapper::solve: wrong size of the right hand side" );

   const IndexType size = this->matrix->getRows();

   this->resetIterations();
   this->setResidue( this->getConvergenceResidue() + 1.0 );

   RealType bNorm = lpNorm( b, (RealType) 2.0 );

   // UMFPACK objects
   void* Symbolic = nullptr;
   void* Numeric = nullptr;

   int status = UMFPACK_OK;
   double Control[ UMFPACK_CONTROL ];
   double Info[ UMFPACK_INFO ];

   // The solver does not work without calling umfpack_di_defaults
   umfpack_di_defaults( Control );

   // umfpack expects Compressed Sparse Column format, we have Compressed Sparse Row
   // so we need to solve  A^T * x = rhs
   int system_type = UMFPACK_Aat;

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
      std::cerr << "error: symbolic reordering failed" << std::endl;
      goto finished;
   }

   // numeric factorization
   status = umfpack_di_numeric( this->matrix->getSegments().getOffsets().getData(),
                                this->matrix->getColumnIndexes().getData(),
                                this->matrix->getValues().getData(),
                                Symbolic,
                                &Numeric,
                                Control,
                                Info );
   if( status != UMFPACK_OK ) {
      std::cerr << "error: numeric factorization failed" << std::endl;
      goto finished;
   }

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
      std::cerr << "error: umfpack_di_solve failed" << std::endl;
      goto finished;
   }

finished:
   if( status != UMFPACK_OK ) {
      // increase print level for reports
      Control[ UMFPACK_PRL ] = 2;
      umfpack_di_report_status( Control, status );
      //        umfpack_di_report_control( Control );
      //        umfpack_di_report_info( Control, Info );
   }

   if( Symbolic )
      umfpack_di_free_symbolic( &Symbolic );
   if( Numeric )
      umfpack_di_free_numeric( &Numeric );

   this->setResidue( LinearResidueGetter::getResidue( *this->matrix, x, b, bNorm ) );
   this->refreshSolverMonitor( true );
   return true;
}

UmfpackWrapper< CSRMatrix< double, Devices::Host, int > >::~UmfpackWrapper()
{
   if( this->Symbolic )
      umfpack_di_free_symbolic( &Symbolic );
   if( this->Numeric )
      umfpack_di_free_numeric( &Numeric );
}

}  // namespace TNL::Solvers::Linear

#endif
