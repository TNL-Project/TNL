// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#ifdef HAVE_TRILINOS
   #include <Kokkos_Core.hpp>
   #include <Kokkos_Core_fwd.hpp>
   #include <Tacho.hpp>
   #include <Tacho_Solver.hpp>
   #include <Tacho_Internal.hpp>
#endif

#include <TNL/Solvers/Linear/LinearSolver.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Algorithms/Segments/CSR.h>
#include <TNL/Matrices/GinkgoOperator.h>
#include <TNL/Containers/GinkgoVector.h>

template< typename Device >
struct KokkosDevice
{};

#ifdef HAVE_TRILINOS
template<>
struct KokkosDevice< TNL::Devices::Sequential >
{
   using type = Kokkos::Serial;
};

template<>
struct KokkosDevice< TNL::Devices::Host >
{
   using type = Kokkos::DefaultHostExecutionSpace;
};

template<>
struct KokkosDevice< TNL::Devices::Cuda >
{
   using type = Kokkos::DefaultExecutionSpace;
};

/*template<>
struct KokkosDevice< TNL::Devices::Hip >
{
   using type = Kokkos::Hip;
};*/
#endif

template< typename Matrix >
class TachoWrapper : public TNL::Solvers::Linear::LinearSolver< Matrix >
{
   using Base = TNL::Solvers::Linear::LinearSolver< Matrix >;

public:
   using RealType = typename Base::RealType;
   using DeviceType = typename Base::DeviceType;
   using IndexType = typename Base::IndexType;
   using VectorViewType = typename Base::VectorViewType;
   using ConstVectorViewType = typename Base::ConstVectorViewType;
   using MatrixPointer = typename Base::MatrixPointer;

#ifdef HAVE_TRILINOS
   using device_type = typename Tacho::UseThisDevice< typename KokkosDevice< DeviceType >::type >::type;
   using DenseMultiVectorType = Kokkos::View< RealType**, Kokkos::LayoutLeft, device_type >;
#endif

   TachoWrapper()
   {
#ifdef HAVE_TRILINOS
#endif
   }

   void
   setMatrix( const MatrixPointer& matrix )
   {
#ifdef HAVE_TRILINOS
      const long unsigned int nnz = matrix->getValues().getSize();
      this->rowPtrs = matrix->getSegments().getOffsets();
      this->values = matrix->getValues();
      this->columns = matrix->getColumnIndexes();

      Kokkos::View< unsigned long int*, device_type > k_rowPtr( rowPtrs.getData(), rowPtrs.getSize() );
      Kokkos::View< IndexType*, device_type > k_columns( this->columns.getData(), nnz );
      Kokkos::View< RealType*, device_type > k_values( this->values.getData(), nnz );

      this->solver.setSolutionMethod( 3 );  // SymLU method
      this->solver.analyze( matrix->getRows(), k_rowPtr, k_columns );
      this->solver.initialize();
      this->solver.factorize( k_values );
#endif
   }

   bool
   solve( ConstVectorViewType b, VectorViewType x ) override
   {
#ifdef HAVE_TRILINOS
      DenseMultiVectorType kokkos_b( "b", b.getSize(), 1 ),  // rhs multivector
         kokkos_x( "x", b.getSize(), 1 ),                    // solution multivector
         kokkos_t( "t", b.getSize(), 1 );                    // temp workspace (store permuted rhs)

      TNL::Algorithms::parallelFor< DeviceType >( 0,
                                                  b.getSize(),
                                                  [ = ] __cuda_callable__( IndexType i )
                                                  {
                                                     kokkos_b( i, 0 ) = b[ i ];
                                                     kokkos_x( i, 0 ) = x[ i ];
                                                  } );
      this->solver.solve( kokkos_x, kokkos_b, kokkos_t );
      return true;
#else
      std::cerr << "Tacho is not available." << std::endl;
      return false;
#endif
   }

protected:
   TNL::Containers::Vector< long unsigned int, DeviceType > rowPtrs;
   TNL::Containers::Vector< IndexType, DeviceType > columns;
   TNL::Containers::Vector< RealType, DeviceType > values;

#ifdef HAVE_TRILINOS
   Kokkos::View< unsigned long int*, device_type > kokkos_rowPtr;
   Kokkos::View< IndexType*, device_type > kokkos_columns;
   Kokkos::View< RealType*, device_type > kokkos_values;

   Tacho::Solver< RealType, device_type > solver;
#endif
};
