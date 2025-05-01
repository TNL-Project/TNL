// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#ifdef HAVE_UMFPACK
   #include <umfpack.h>
#endif

#include <TNL/Solvers/DirectSolver.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Algorithms/Segments/CSR.h>

namespace TNL::Solvers::Linear {

template< typename Matrix >
struct is_csr_matrix
{
   static constexpr bool value = false;
};

template< typename Real, typename Device, typename Index, typename ComputeReal, typename RealAllocator, typename IndexAllocator >
struct is_csr_matrix< Matrices::SparseMatrix< Real,
                                              Device,
                                              Index,
                                              Matrices::GeneralMatrix,
                                              Algorithms::Segments::CSR,
                                              ComputeReal,
                                              RealAllocator,
                                              IndexAllocator > >
{
   static constexpr bool value = true;
};

template< typename Real, typename Device, typename Index >
using CSRMatrix = Matrices::SparseMatrix< Real, Device, Index, Matrices::GeneralMatrix, Algorithms::Segments::CSR >;

template< typename Matrix, typename SolverMonitor = DirectSolverMonitor< double, int > >
class UmfpackWrapper : public DirectSolver< typename Matrix::RealType, typename Matrix::IndexType, SolverMonitor >
{
   using Base = DirectSolver< typename Matrix::RealType, typename Matrix::IndexType, SolverMonitor >;

public:
   /**
    * \brief Type for floating point numbers.
    */
   using RealType = typename Matrix::RealType;

   /**
    * \brief Device where the solver will run on and auxillary data will alloacted on.
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

   UmfpackWrapper();

   void
   setMatrix( const MatrixPointer& matrix );

   bool
   solve( ConstVectorViewType b, VectorViewType x );

   bool
   solved() const;
};

template< typename SolverMonitor >
class UmfpackWrapper< CSRMatrix< double, Devices::Host, int >, SolverMonitor >
: public DirectSolver< double, int, SolverMonitor >
{
   using Base = DirectSolver< double, int, SolverMonitor >;

public:
   /**
    * \brief Type for floating point numbers.
    */
   using RealType = double;

   /**
    * \brief Device where the solver will run on and auxillary data will alloacted on.
    */
   using DeviceType = Devices::Host;

   /**
    * \brief Indexing type.
    */
   using IndexType = int;

   /**
    * \brief Type of the matrix representing the linear system.
    */
   using MatrixType = CSRMatrix< double, Devices::Host, int >;

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

   void
   setMatrix( const MatrixPointer& matrix );

   bool
   solve( ConstVectorViewType b, VectorViewType x );

   bool
   solved() const;

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
