// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#ifdef HAVE_UMFPACK

   #include <umfpack.h>

   #include "LinearSolver.h"
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

template< typename Matrix >
class UmfpackWrapper : public LinearSolver< Matrix >
{
   using Base = LinearSolver< Matrix >;

public:
   using RealType = typename Base::RealType;
   using DeviceType = typename Base::DeviceType;
   using IndexType = typename Base::IndexType;
   using VectorViewType = typename Base::VectorViewType;
   using ConstVectorViewType = typename Base::ConstVectorViewType;

   UmfpackWrapper()
   {
      if( ! is_csr_matrix< Matrix >::value )
         std::cerr << "The UmfpackWrapper solver is available only for CSR matrices." << std::endl;
      if( std::is_same_v< typename Matrix::DeviceType, Devices::Cuda > )
         std::cerr << "The UmfpackWrapper solver is not available on CUDA." << std::endl;
      if( ! std::is_same_v< RealType, double > )
         std::cerr << "The UmfpackWrapper solver is available only for double precision." << std::endl;
      if( ! std::is_same_v< IndexType, int > )
         std::cerr << "The UmfpackWrapper solver is available only for 'int' index type." << std::endl;
   }

   bool
   solve( ConstVectorViewType b, VectorViewType x ) override
   {
      return false;
   }
};

template<>
class UmfpackWrapper< CSRMatrix< double, Devices::Host, int > > : public LinearSolver< CSRMatrix< double, Devices::Host, int > >
{
   using Base = LinearSolver< CSRMatrix< double, Devices::Host, int > >;

public:
   using RealType = typename Base::RealType;
   using DeviceType = typename Base::DeviceType;
   using IndexType = typename Base::IndexType;
   using VectorViewType = typename Base::VectorViewType;
   using ConstVectorViewType = typename Base::ConstVectorViewType;

   void
   setMatrix( const MatrixPointer& matrix ) override;

   bool
   solve( ConstVectorViewType b, VectorViewType x ) override;

   ~UmfpackWrapper();

protected:
   // UMFPACK objects
   void* Symbolic = nullptr;
   void* Numeric = nullptr;
   double Control[ UMFPACK_CONTROL ];
   double Info[ UMFPACK_INFO ];

   bool factorized = false;
};

}  // namespace TNL::Solvers::Linear

   #include "UmfpackWrapper.hpp"

#endif
