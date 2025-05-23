// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#ifdef HAVE_UMFPACK
   #include <umfpack.h>
#endif

#include <TNL/Solvers/DirectSolver.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Matrices/MatrixInfo.h>
#include <TNL/Algorithms/Segments/CSR.h>

namespace TNL::Solvers::Linear {

template< typename Matrix, typename SolverMonitor = DirectSolverMonitor< double, int > >
class UmfpackWrapper : public DirectSolver< typename Matrix::RealType, typename Matrix::IndexType, SolverMonitor >
{
   static_assert( Matrices::is_csr_matrix< Matrix >::value, "Umfpack works only with CSR format." );
   static_assert( std::is_same_v< typename Matrix::DeviceType, TNL::Devices::Host >
                     || std::is_same_v< typename Matrix::DeviceType, TNL::Devices::Sequential >,
                  "Umfpack is only available on the host." );
   static_assert( std::is_same_v< typename Matrix::RealType, double >, "Umfpack is only available for double precision." );
   static_assert( std::is_same_v< typename Matrix::IndexType, int >, "Umfpack is only available for int indexing." );

   using Base = DirectSolver< typename Matrix::RealType, typename Matrix::IndexType, SolverMonitor >;

public:
   /**
    * \brief Type for floating point numbers.
    */
   using RealType = typename Matrix::RealType;

   /**
    * \brief Device where the solver will run on and auxiliary data will be allocated on.
    */
   using DeviceType = typename Matrix::DeviceType;

   /**
    * \brief Indexing type.
    */
   using IndexType = typename Matrix::IndexType;

   /**
    * \brief Type of the matrix representing the linear system.
    */
   using MatrixType = Matrix;

   /**
    * \brief Type of shared pointer to the matrix.
    */
   using MatrixPointer = std::shared_ptr< std::add_const_t< MatrixType > >;

   /**
    * \brief Type for vector view.
    */
   using VectorViewType = Containers::VectorView< RealType, DeviceType, IndexType >;

   /**
    * \brief Type for constant vector view.
    */
   using ConstVectorViewType = typename VectorViewType::ConstViewType;

   UmfpackWrapper() = default;

   void
   setMatrix( const MatrixPointer& matrix );

   bool
   solve( ConstVectorViewType b, VectorViewType x );

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
