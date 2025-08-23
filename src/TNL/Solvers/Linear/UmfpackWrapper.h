// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#ifdef HAVE_UMFPACK
   #include <umfpack.h>
#endif

#include <TNL/Solvers/Linear/LinearSolver.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Matrices/TypeTraits.h>
#include <TNL/Algorithms/Segments/CSR.h>

namespace TNL::Solvers::Linear {

template< typename Matrix, typename SolverMonitor = IterativeSolverMonitor< double > >
class UmfpackWrapper : public LinearSolver< Matrix >
{
   static_assert( Matrices::is_sparse_csr_matrix_v< Matrix >, "Umfpack works only with CSR format." );
   static_assert( std::is_same_v< typename Matrix::DeviceType, TNL::Devices::Host >
                     || std::is_same_v< typename Matrix::DeviceType, TNL::Devices::Sequential >,
                  "Umfpack is only available on the host." );
   static_assert( std::is_same_v< typename Matrix::RealType, double >, "Umfpack is only available for double precision." );
   static_assert( std::is_same_v< typename Matrix::IndexType, int >, "Umfpack is only available for int indexing." );

   using Base = LinearSolver< Matrix >;

public:
   using RealType = typename Base::RealType;
   using IndexType = typename Base::IndexType;
   using MatrixPointer = typename Base::MatrixPointer;
   using VectorViewType = typename Base::VectorViewType;
   using ConstVectorViewType = typename Base::ConstVectorViewType;

   UmfpackWrapper() = default;

   void
   setMatrix( const MatrixPointer& matrix ) override;

   bool
   solve( ConstVectorViewType b, VectorViewType x ) override;

   bool
   succeeded() const;

   ~UmfpackWrapper();

protected:
#ifdef HAVE_UMFPACK
   // UMFPACK objects
   void* Symbolic = nullptr;
   void* Numeric = nullptr;
   double Control[ UMFPACK_CONTROL ];
   double Info[ UMFPACK_INFO ];
#endif

   MatrixPointer matrix;

   bool factorized = false;
   bool solver_success = false;
};

}  // namespace TNL::Solvers::Linear

#include "UmfpackWrapper.hpp"
