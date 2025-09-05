// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <set>
#include <sstream>
#include <string>
#include <random>

#ifndef NDEBUG
   #include <TNL/Debugging/FPE.h>
#endif

#include <TNL/Config/parseCommandLine.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/MPI/ScopedInitializer.h>
#include <TNL/MPI/Config.h>
#include <TNL/Containers/BlockPartitioning.h>
#include <TNL/Containers/DistributedVector.h>
#include <TNL/Matrices/DistributedMatrix.h>
#include <TNL/Matrices/SparseOperations.h>
#include <TNL/Matrices/MatrixReader.h>
#include <TNL/Solvers/Linear/Preconditioners/Diagonal.h>
#include <TNL/Solvers/Linear/Preconditioners/ILU0.h>
#include <TNL/Solvers/Linear/Preconditioners/ILUT.h>
#include <TNL/Solvers/Linear/GMRES.h>
#include <TNL/Solvers/Linear/TFQMR.h>
#include <TNL/Solvers/Linear/BICGStab.h>
#include <TNL/Solvers/Linear/BICGStabL.h>
#include <TNL/Solvers/Linear/IDRs.h>
#include <TNL/Solvers/Linear/CuSolverWrapper.h>
#include <TNL/Solvers/Linear/UmfpackWrapper.h>
#include <TNL/Solvers/Linear/GinkgoDirectSolver.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Algorithms/Segments/CSR.h>
#include <TNL/Algorithms/Segments/SlicedEllpack.h>
#include <TNL/Benchmarks/Benchmarks.h>
#include "ordering.h"
#include "benchmarks.h"
#include "BenchmarkResults.h"
#include "StrumpackWrapper.h"
#include "TachoWrapper.h"
#include "CuDSSWrapper.h"

// FIXME: nvcc 8.0 fails when cusolverSp.h is included (works fine with clang):
// /opt/cuda/include/cuda_fp16.h(3068): error: more than one instance of overloaded function "isinf" matches the argument list:
//             function "isinf(float)"
//             function "std::isinf(float)"
//             argument types are: (float)
//#if defined(__CUDACC__) && !defined(__NVCC__)
// FIXME: CuSolverWrapper does not work at all now
#if 0
   #include "CuSolverWrapper.h"
   #define HAVE_CUSOLVER
#endif

template< typename _Device, typename _Index, typename _IndexAlocator >
//using SegmentsType = TNL::Algorithms::Segments::SlicedEllpack< _Device, _Index, _IndexAlocator >;
using SegmentsType = TNL::Algorithms::Segments::CSR< _Device, _Index, _IndexAlocator >;

static const std::set< std::string > valid_solvers = {
   "gmres", "tfqmr", "bicgstab", "bicgstab-ell", "idrs",
};

static const std::set< std::string > valid_gmres_variants = {
   "CGS", "CGSR", "MGS", "MGSR", "CWY",
};

static const std::set< std::string > valid_preconditioners = {
   "jacobi",
   "ilu0",
   "ilut",
};

std::set< std::string >
parse_comma_list( const TNL::Config::ParameterContainer& parameters,
                  const char* parameter,
                  const std::set< std::string >& options )
{
   const TNL::String param = parameters.getParameter< TNL::String >( parameter );

   if( param == "all" )
      return options;

   std::stringstream ss( param.getString() );
   std::string s;
   std::set< std::string > set;

   while( std::getline( ss, s, ',' ) ) {
      if( ! options.count( s ) )
         throw std::logic_error( std::string( "Invalid value in the comma-separated list for the parameter '" ) + parameter
                                 + "': '" + s + "'. The list contains: '" + param.getString() + "'." );

      set.insert( s );

      if( ss.peek() == ',' )
         ss.ignore();
   }

   return set;
}

// initialize all vector entries with a uniformly distributed random value from the interval [a, b]
template< typename Vector >
void
set_random_vector( Vector& v, typename Vector::RealType a, typename Vector::RealType b )
{
   using RealType = typename Vector::RealType;
   using IndexType = typename Vector::IndexType;
   // random device will be used to obtain a seed for the random number engine
   std::random_device rd;
   // initialize the standard mersenne_twister_engine with rd() as the seed
   std::mt19937 gen( rd() );
   // create uniform distribution
   std::uniform_real_distribution< RealType > dis( a, b );

   // create host vector
   typename Vector::template Self< RealType, TNL::Devices::Host > host_v;
   host_v.setSize( v.getSize() );

   // initialize the host vector
   auto kernel = [ & ]( IndexType i )
   {
      host_v[ i ] = dis( gen );
   };
   TNL::Algorithms::parallelFor< TNL::Devices::Host >( 0, host_v.getSize(), kernel );

   // copy the data to the device vector
   v = host_v;
}

template< typename Matrix, typename Vector >
void
benchmarkIterativeSolvers( TNL::Benchmarks::Benchmark<>& benchmark,
                           TNL::Config::ParameterContainer parameters,
                           const std::shared_ptr< Matrix >& matrixPointer,
                           const Vector& x0,
                           const Vector& b )
{
#ifdef __CUDACC__
   using CudaMatrix = typename Matrix::template Self< typename Matrix::RealType, TNL::Devices::Cuda >;
   using CudaVector = typename Vector::template Self< typename Vector::RealType, TNL::Devices::Cuda >;

   CudaVector cuda_x0, cuda_b;
   cuda_x0 = x0;
   cuda_b = b;

   auto cudaMatrixPointer = std::make_shared< CudaMatrix >();
   *cudaMatrixPointer = *matrixPointer;
#endif

   using namespace TNL::Solvers::Linear;
   using namespace TNL::Solvers::Linear::Preconditioners;

   const int ell_max = 2;
   const std::set< std::string > solvers = parse_comma_list( parameters, "solvers", valid_solvers );
   const std::set< std::string > gmresVariants = parse_comma_list( parameters, "gmres-variants", valid_gmres_variants );
   const std::set< std::string > preconditioners = parse_comma_list( parameters, "preconditioners", valid_preconditioners );
   const bool with_preconditioner_update = parameters.getParameter< bool >( "with-preconditioner-update" );

   if( preconditioners.count( "jacobi" ) ) {
      if( with_preconditioner_update ) {
         benchmark.setOperation( "preconditioner update (Jacobi)" );
         benchmarkPreconditionerUpdate< Diagonal >( benchmark, parameters, matrixPointer );
#ifdef __CUDACC__
         benchmarkPreconditionerUpdate< Diagonal >( benchmark, parameters, cudaMatrixPointer );
#endif
      }

      if( solvers.count( "gmres" ) ) {
         for( auto variant : gmresVariants ) {
            benchmark.setOperation( variant + "-GMRES (Jacobi)" );
            parameters.template setParameter< TNL::String >( "gmres-variant", variant );
            benchmarkSolver< GMRES, Diagonal >( benchmark, parameters, matrixPointer, x0, b );
#ifdef __CUDACC__
            benchmarkSolver< GMRES, Diagonal >( benchmark, parameters, cudaMatrixPointer, cuda_x0, cuda_b );
#endif
         }
      }

      if( solvers.count( "tfqmr" ) ) {
         benchmark.setOperation( "TFQMR (Jacobi)" );
         benchmarkSolver< TFQMR, Diagonal >( benchmark, parameters, matrixPointer, x0, b );
#ifdef __CUDACC__
         benchmarkSolver< TFQMR, Diagonal >( benchmark, parameters, cudaMatrixPointer, cuda_x0, cuda_b );
#endif
      }

      if( solvers.count( "bicgstab" ) ) {
         benchmark.setOperation( "BiCGstab (Jacobi)" );
         benchmarkSolver< BICGStab, Diagonal >( benchmark, parameters, matrixPointer, x0, b );
#ifdef __CUDACC__
         benchmarkSolver< BICGStab, Diagonal >( benchmark, parameters, cudaMatrixPointer, cuda_x0, cuda_b );
#endif
      }

      if( solvers.count( "bicgstab-ell" ) ) {
         benchmark.setOperation( "BiCGstab (Jacobi)" );
         for( int ell = 1; ell <= ell_max; ell++ ) {
            parameters.template setParameter< int >( "bicgstab-ell", ell );
            benchmark.setOperation( "BiCGstab(" + TNL::convertToString( ell ) + ") (Jacobi)" );
            benchmarkSolver< BICGStabL, Diagonal >( benchmark, parameters, matrixPointer, x0, b );
#ifdef __CUDACC__
            benchmarkSolver< BICGStabL, Diagonal >( benchmark, parameters, cudaMatrixPointer, cuda_x0, cuda_b );
#endif
         }
      }

      if( solvers.count( "idrs" ) ) {
         benchmark.setOperation( "IDRs (Jacobi)" );
         benchmarkSolver< IDRs, Diagonal >( benchmark, parameters, matrixPointer, x0, b );
#ifdef __CUDACC__
         benchmarkSolver< IDRs, Diagonal >( benchmark, parameters, cudaMatrixPointer, cuda_x0, cuda_b );
#endif
      }
   }

   if( preconditioners.count( "ilu0" ) ) {
      if( with_preconditioner_update ) {
         benchmark.setOperation( "preconditioner update (ILU0)" );
         benchmarkPreconditionerUpdate< ILU0 >( benchmark, parameters, matrixPointer );
#ifdef __CUDACC__
         benchmarkPreconditionerUpdate< ILU0 >( benchmark, parameters, cudaMatrixPointer );
#endif
      }

      if( solvers.count( "gmres" ) ) {
         for( auto variant : gmresVariants ) {
            benchmark.setOperation( variant + "-GMRES (ILU0)" );
            parameters.template setParameter< TNL::String >( "gmres-variant", variant );
            benchmarkSolver< GMRES, ILU0 >( benchmark, parameters, matrixPointer, x0, b );
#ifdef __CUDACC__
            benchmarkSolver< GMRES, ILU0 >( benchmark, parameters, cudaMatrixPointer, cuda_x0, cuda_b );
#endif
         }
      }

      if( solvers.count( "tfqmr" ) ) {
         benchmark.setOperation( "TFQMR (ILU0)" );
         benchmarkSolver< TFQMR, ILU0 >( benchmark, parameters, matrixPointer, x0, b );
#ifdef __CUDACC__
         benchmarkSolver< TFQMR, ILU0 >( benchmark, parameters, cudaMatrixPointer, cuda_x0, cuda_b );
#endif
      }

      if( solvers.count( "bicgstab" ) ) {
         benchmark.setOperation( "BiCGstab (ILU0)" );
         benchmarkSolver< BICGStab, ILU0 >( benchmark, parameters, matrixPointer, x0, b );
#ifdef __CUDACC__
         benchmarkSolver< BICGStab, ILU0 >( benchmark, parameters, cudaMatrixPointer, cuda_x0, cuda_b );
#endif
      }

      if( solvers.count( "bicgstab-ell" ) ) {
         for( int ell = 1; ell <= ell_max; ell++ ) {
            parameters.template setParameter< int >( "bicgstab-ell", ell );
            benchmark.setOperation( "BiCGstab(" + TNL::convertToString( ell ) + ") (ILU0)" );
            benchmarkSolver< BICGStabL, ILU0 >( benchmark, parameters, matrixPointer, x0, b );
#ifdef __CUDACC__
            benchmarkSolver< BICGStabL, ILU0 >( benchmark, parameters, cudaMatrixPointer, cuda_x0, cuda_b );
#endif
         }
      }

      if( solvers.count( "idrs" ) ) {
         benchmark.setOperation( "IDRs (ILU0)" );
         benchmarkSolver< IDRs, ILU0 >( benchmark, parameters, matrixPointer, x0, b );
#ifdef __CUDACC__
         benchmarkSolver< IDRs, ILU0 >( benchmark, parameters, cudaMatrixPointer, cuda_x0, cuda_b );
#endif
      }
   }

   if( preconditioners.count( "ilut" ) ) {
      if( with_preconditioner_update ) {
         benchmark.setOperation( "preconditioner update (ILUT)" );
         benchmarkPreconditionerUpdate< ILUT >( benchmark, parameters, matrixPointer );
#ifdef __CUDACC__
         benchmarkPreconditionerUpdate< ILUT >( benchmark, parameters, cudaMatrixPointer );
#endif
      }

      if( solvers.count( "gmres" ) ) {
         for( auto variant : gmresVariants ) {
            benchmark.setOperation( variant + "-GMRES (ILUT)" );
            parameters.template setParameter< TNL::String >( "gmres-variant", variant );
            benchmarkSolver< GMRES, ILUT >( benchmark, parameters, matrixPointer, x0, b );
#ifdef __CUDACC__
            benchmarkSolver< GMRES, ILUT >( benchmark, parameters, cudaMatrixPointer, cuda_x0, cuda_b );
#endif
         }
      }

      if( solvers.count( "tfqmr" ) ) {
         benchmark.setOperation( "TFQMR (ILUT)" );
         benchmarkSolver< TFQMR, ILUT >( benchmark, parameters, matrixPointer, x0, b );
#ifdef __CUDACC__
         benchmarkSolver< TFQMR, ILUT >( benchmark, parameters, cudaMatrixPointer, cuda_x0, cuda_b );
#endif
      }

      if( solvers.count( "bicgstab" ) ) {
         benchmark.setOperation( "BiCGstab (ILUT)" );
         benchmarkSolver< BICGStab, ILUT >( benchmark, parameters, matrixPointer, x0, b );
#ifdef __CUDACC__
         benchmarkSolver< BICGStab, ILUT >( benchmark, parameters, cudaMatrixPointer, cuda_x0, cuda_b );
#endif
      }

      if( solvers.count( "bicgstab-ell" ) ) {
         for( int ell = 1; ell <= ell_max; ell++ ) {
            parameters.template setParameter< int >( "bicgstab-ell", ell );
            benchmark.setOperation( "BiCGstab(" + TNL::convertToString( ell ) + ") (ILUT)" );
            benchmarkSolver< BICGStabL, ILUT >( benchmark, parameters, matrixPointer, x0, b );
#ifdef __CUDACC__
            benchmarkSolver< BICGStabL, ILUT >( benchmark, parameters, cudaMatrixPointer, cuda_x0, cuda_b );
#endif
         }
      }

      if( solvers.count( "idrs" ) ) {
         benchmark.setOperation( "IDRs (ILUT)" );
         benchmarkSolver< IDRs, ILUT >( benchmark, parameters, matrixPointer, x0, b );
#ifdef __CUDACC__
         benchmarkSolver< IDRs, ILUT >( benchmark, parameters, cudaMatrixPointer, cuda_x0, cuda_b );
#endif
      }
   }
}

template< typename MatrixType >
struct LinearSolversBenchmark
{
   using RealType = typename MatrixType::RealType;
   using DeviceType = typename MatrixType::DeviceType;
   using IndexType = typename MatrixType::IndexType;
   using VectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;

   using DistributedMatrix = TNL::Matrices::DistributedMatrix< MatrixType >;
   using DistributedVector = TNL::Containers::DistributedVector< RealType, DeviceType, IndexType >;
   using DistributedRowLengths = typename DistributedMatrix::RowCapacitiesType;

   static bool
   run( TNL::Benchmarks::Benchmark<>& benchmark, const TNL::Config::ParameterContainer& parameters )
   {
      const TNL::String file_matrix = parameters.getParameter< TNL::String >( "input-matrix" );
      const TNL::String file_dof = parameters.getParameter< TNL::String >( "input-dof" );
      const TNL::String file_rhs = parameters.getParameter< TNL::String >( "input-rhs" );

      auto matrixPointer = std::make_shared< MatrixType >();
      VectorType x0, b;

      // load the matrix
      if( file_matrix.endsWith( ".mtx" ) ) {
         TNL::Matrices::MatrixReader< MatrixType > reader;
         reader.readMtx( file_matrix, *matrixPointer );
      }
      else {
         matrixPointer->load( file_matrix );
      }
      TNL::Matrices::compressSparseMatrix( *matrixPointer );
      matrixPointer->sortColumnIndexes();

      // load the vectors
      if( file_dof && file_rhs ) {
         TNL::File( file_dof, std::ios_base::in ) >> x0;
         TNL::File( file_rhs, std::ios_base::in ) >> b;
      }
      else {
         // set x0 := 0
         x0.setSize( matrixPointer->getColumns() );
         x0 = 0;

         // generate random vector x
         VectorType x;
         x.setSize( matrixPointer->getColumns() );
         if( parameters.getParameter< TNL::String >( "set-rhs" ) == "random" )
            set_random_vector( x, 1e2, 1e3 );
         else
            x = 1;

         // set b := A*x
         b.setSize( matrixPointer->getRows() );
         matrixPointer->vectorProduct( x, b );
      }

      typename MatrixType::RowCapacitiesType rowLengths;
      matrixPointer->getCompressedRowLengths( rowLengths );
      const IndexType maxRowLength = max( rowLengths );

      const TNL::String title = ( TNL::MPI::GetSize() > 1 ) ? "Distributed linear solvers" : "Linear solvers";
      std::cout << "\n== " << title << " ==\n" << std::endl;

      benchmark.setMetadataColumns( TNL::Benchmarks::Benchmark<>::MetadataColumns( {
         { "matrix name", parameters.getParameter< TNL::String >( "name" ) },
         // TODO: strip the device
         //         { "matrix type", matrixPointer->getType() },
         { "rows", TNL::convertToString( matrixPointer->getRows() ) },
         { "columns", TNL::convertToString( matrixPointer->getColumns() ) },
         // FIXME: getMaxRowLengths() returns 0 for matrices loaded from file
         //         { "max elements per row", matrixPointer->getMaxRowLength() },
         { "max elements per row", TNL::convertToString( maxRowLength ) },
      } ) );

      if( parameters.getParameter< bool >( "with-iterative" ) ) {
         std::cout << "Iterative solvers:" << std::endl;

         if( parameters.getParameter< bool >( "reorder-dofs" ) ) {
            using PermutationVector = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;
            PermutationVector perm, iperm;
            getTrivialOrdering( *matrixPointer, perm, iperm );
            auto matrix_perm = std::make_shared< MatrixType >();
            VectorType x0_perm, b_perm;
            x0_perm.setLike( x0 );
            b_perm.setLike( b );
            TNL::Matrices::reorderSparseMatrix( *matrixPointer, *matrix_perm, perm, iperm );
            TNL::Matrices::reorderArray( x0, x0_perm, perm );
            TNL::Matrices::reorderArray( b, b_perm, perm );
            if( TNL::MPI::GetSize() > 1 )
               runDistributed( benchmark, parameters, matrix_perm, x0_perm, b_perm );
            else
               runNonDistributed( benchmark, parameters, matrix_perm, x0_perm, b_perm );
         }
         else {
            if( TNL::MPI::GetSize() > 1 )
               runDistributed( benchmark, parameters, matrixPointer, x0, b );
            else
               runNonDistributed( benchmark, parameters, matrixPointer, x0, b );
         }
      }
      if( parameters.getParameter< bool >( "with-direct" ) ) {
         std::cout << "Direct solvers:" << std::endl;
         runDirect( benchmark, parameters, matrixPointer, x0, b );
      }
      return true;
   }

   static void
   runDistributed( TNL::Benchmarks::Benchmark<>& benchmark,
                   const TNL::Config::ParameterContainer& parameters,
                   const std::shared_ptr< MatrixType >& matrixPointer,
                   const VectorType& x0,
                   const VectorType& b )
   {
      // set up the distributed matrix
      const auto communicator = MPI_COMM_WORLD;
      const auto localRange = TNL::Containers::splitRange( matrixPointer->getRows(), communicator );
      auto distMatrixPointer = std::make_shared< DistributedMatrix >(
         localRange, matrixPointer->getRows(), matrixPointer->getColumns(), communicator );
      DistributedVector dist_x0( localRange, 0, matrixPointer->getRows(), communicator );
      DistributedVector dist_b( localRange, 0, matrixPointer->getRows(), communicator );

      // copy the row capacities from the global matrix to the distributed matrix
      DistributedRowLengths distributedRowLengths( localRange, 0, matrixPointer->getRows(), communicator );
      for( IndexType i = 0; i < distMatrixPointer->getLocalMatrix().getRows(); i++ ) {
         const auto gi = distMatrixPointer->getLocalRowRange().getGlobalIndex( i );
         distributedRowLengths[ gi ] = matrixPointer->getRowCapacity( gi );
      }
      distMatrixPointer->setRowCapacities( distributedRowLengths );

      // copy data from the global matrix/vector into the distributed matrix/vector
      for( IndexType i = 0; i < distMatrixPointer->getLocalMatrix().getRows(); i++ ) {
         const auto gi = distMatrixPointer->getLocalRowRange().getGlobalIndex( i );
         dist_x0[ gi ] = x0[ gi ];
         dist_b[ gi ] = b[ gi ];

         //         const IndexType rowLength = matrixPointer->getRowLength( i );
         //         IndexType columns[ rowLength ];
         //         RealType values[ rowLength ];
         //         matrixPointer->getRowFast( gi, columns, values );
         //         distMatrixPointer->setRowFast( gi, columns, values, rowLength );
         const auto global_row = matrixPointer->getRow( gi );
         auto local_row = distMatrixPointer->getRow( gi );
         for( IndexType j = 0; j < global_row.getSize(); j++ )
            local_row.setElement( j, global_row.getColumnIndex( j ), global_row.getValue( j ) );
      }
      benchmarkIterativeSolvers( benchmark, parameters, distMatrixPointer, dist_x0, dist_b );
   }

   static void
   runNonDistributed( TNL::Benchmarks::Benchmark<>& benchmark,
                      const TNL::Config::ParameterContainer& parameters,
                      const std::shared_ptr< MatrixType >& matrixPointer,
                      const VectorType& x0,
                      const VectorType& b )
   {
      benchmarkIterativeSolvers( benchmark, parameters, matrixPointer, x0, b );

#ifdef HAVE_CUSOLVER
      std::cout << "CuSOLVER:" << std::endl;
      {
         using CSR = TNL::Matrices::
            SparseMatrix< RealType, DeviceType, IndexType, TNL::Matrices::GeneralMatrix, Algorithms::Segments::CSR >;
         auto matrixCopy = std::make_shared< CSR >();
         Matrices::copySparseMatrix( *matrixCopy, *matrixPointer );

         using CudaCSR = TNL::Matrices::
            SparseMatrix< RealType, Devices::Cuda, IndexType, TNL::Matrices::GeneralMatrix, Algorithms::Segments::CSR >;
         using CudaVector = typename VectorType::template Self< RealType, Devices::Cuda >;
         auto cuda_matrixCopy = std::make_shared< CudaCSR >();
         *cuda_matrixCopy = *matrixCopy;
         CudaVector cuda_x0, cuda_b;
         cuda_x0.setLike( x0 );
         cuda_b.setLike( b );
         cuda_x0 = x0;
         cuda_b = b;

         using namespace Solvers::Linear;
         using namespace Solvers::Linear::Preconditioners;
         benchmarkSolver< CuSolverWrapper, Preconditioner >( benchmark, parameters, cuda_matrixCopy, cuda_x0, cuda_b );
      }
#endif
   }

   static void
   runDirect( TNL::Benchmarks::Benchmark<>& benchmark,
              const TNL::Config::ParameterContainer& parameters,
              const std::shared_ptr< MatrixType >& matrixPointer,
              const VectorType& x0,
              const VectorType& b )
   {
      using CSR = TNL::Matrices::
         SparseMatrix< RealType, DeviceType, IndexType, TNL::Matrices::GeneralMatrix, TNL::Algorithms::Segments::CSR >;
      auto matrixCopy = std::make_shared< CSR >();
      TNL::Matrices::copySparseMatrix( *matrixCopy, *matrixPointer );

#ifdef HAVE_UMFPACK
      if constexpr( ( std::is_same_v< DeviceType, TNL::Devices::Host >
                      || std::is_same_v< DeviceType, TNL::Devices::Sequential > )
                    && std::is_same_v< RealType, double > && std::is_same_v< IndexType, int > )
         benchmarkDirectSolver< TNL::Solvers::Linear::UmfpackWrapper >( "UMFPACK", benchmark, parameters, matrixCopy, x0, b );
#endif

#ifdef HAVE_TRILINOS
      benchmarkDirectSolver< TachoWrapper >( "Tacho CPU", benchmark, parameters, matrixCopy, x0, b );
#endif

#ifdef HAVE_GINKGO
      benchmarkDirectSolver< TNL::Solvers::Linear::GinkgoDirectSolver >(
         "Ginkgo CPU", benchmark, parameters, matrixCopy, x0, b );
#endif

#ifdef __CUDACC__
      const std::string performer = "CPU/GPU";
      auto compute = [ & ]()
      {
         TNL::Matrices::copySparseMatrix( *matrixCopy, *matrixPointer );
      };
      TNL::Benchmarks::BenchmarkResult benchmarkResult;
      benchmark.setOperation( "Copy" );
      benchmark.time< TNL::Devices::Host >( performer, compute, benchmarkResult );

      using CudaCSR = TNL::Matrices::
         SparseMatrix< RealType, TNL::Devices::Cuda, IndexType, TNL::Matrices::GeneralMatrix, TNL::Algorithms::Segments::CSR >;
      auto cudaMatrix = std::make_shared< CudaCSR >();
      TNL::Containers::Vector< RealType, TNL::Devices::Cuda, IndexType > cuda_x0( x0 ), cuda_b( b );
      TNL::Containers::Vector< RealType, TNL::Devices::Host, IndexType > cuda_x0_copy;
      *cudaMatrix = *matrixPointer;

      benchmarkDirectSolver< TNL::Solvers::Linear::CuSolverWrapper >(
         "CuSolver", benchmark, parameters, cudaMatrix, cuda_x0, cuda_b );
      cuda_x0_copy = cuda_x0;
      if( l2Norm( cuda_x0_copy - x0 ) > 1e-10 )
         std::cout << "Warning: the result of the CuSolver solver is not equal to the result of the CPU solver." << std::endl;

   #ifdef HAVE_CUDSS
      benchmarkDirectSolver< TNL::Solvers::Linear::CuDSSWrapper >(
         "CuDSS", benchmark, parameters, cudaMatrix, cuda_x0, cuda_b );
      cuda_x0_copy = cuda_x0;
      if( l2Norm( cuda_x0_copy - x0 ) > 1e-10 )
         std::cout << "Warning: the result of the CuDSS solver is not equal to the result of the CPU solver." << std::endl;
   #endif

   #ifdef HAVE_GINKGO
      benchmarkDirectSolver< TNL::Solvers::Linear::GinkgoDirectSolver >(
         "Ginkgo GPU", benchmark, parameters, cudaMatrix, cuda_x0, cuda_b );
      cuda_x0_copy = cuda_x0;
      if( l2Norm( cuda_x0_copy - x0 ) > 1e-10 )
         std::cout << "Warning: the result of the Ginkgo GPU solver is not equal to the result of the CPU solver." << std::endl;
   #endif

   #ifdef HAVE_TRILINOS
      if( ! std::is_same_v< Kokkos::DefaultHostExecutionSpace, Kokkos::DefaultExecutionSpace > ) {
         benchmarkDirectSolver< TachoWrapper >( "Tacho GPU", benchmark, parameters, cudaMatrix, cuda_x0, cuda_b );
         cuda_x0_copy = cuda_x0;
         if( l2Norm( cuda_x0_copy - x0 ) > 1e-10 )
            std::cout << "Warning: the result of the Tacho GPU solver is not equal to the result of the CPU solver."
                      << std::endl;
      }
   #endif
#endif  // __CUDACC__

#ifdef HAVE_STRUMPACK
      // Strumpack currently support only GPU offloading - https://github.com/pghysels/STRUMPACK/issues/113
      benchmarkDirectSolver< StrumpackWrapper >( "Strumpack", benchmark, parameters, matrixCopy, x0, b );
#endif

#ifdef HAVE_ARMADILLO
      std::cout << "Armadillo wrapper (which wraps SuperLU):" << std::endl;
      benchmarkArmadillo( parameters, matrixCopy, x0, b );
#endif
   }
};

void
configSetup( TNL::Config::ConfigDescription& config )
{
   config.addDelimiter( "Benchmark settings:" );
   config.addEntry< TNL::String >( "log-file", "Log file name.", "tnl-benchmark-linear-solvers.log" );
   config.addEntry< TNL::String >( "output-mode", "Mode for opening the log file.", "overwrite" );
   config.addEntryEnum( "append" );
   config.addEntryEnum( "overwrite" );
   config.addEntry< int >( "loops", "Number of repetitions of the benchmark.", 10 );
   config.addRequiredEntry< TNL::String >( "input-matrix",
                                           "File name of the input matrix (in binary TNL format or textual MTX format)." );
   config.addEntry< TNL::String >( "input-dof", "File name of the input DOF vector (in binary TNL format).", "" );
   config.addEntry< TNL::String >( "input-rhs", "File name of the input right-hand-side vector (in binary TNL format).", "" );
   config.addEntry< TNL::String >( "set-rhs", "Saya how to set the right-hand-side vector if no input file is given.", "ones" );
   config.addEntryEnum( "ones" );
   config.addEntryEnum( "random" );
   config.addEntry< TNL::String >( "name", "Name of the matrix in the benchmark.", "" );
   config.addEntry< int >( "verbose", "Verbose mode.", 1 );
   config.addEntry< bool >( "reorder-dofs", "Reorder matrix entries corresponding to the same DOF together.", false );
   config.addEntry< bool >( "with-iterative", "Includes the iterative solvers in the benchmark.", true );
   config.addEntry< bool >( "with-direct", "Includes the 3rd party direct solvers in the benchmark.", true );
   config.addEntry< TNL::String >(
      "solvers",
      "Comma-separated list of solvers to run benchmarks for. Options: gmres, tfqmr, bicgstab, bicgstab-ell, idrs.",
      "all" );
   config.addEntry< TNL::String >(
      "gmres-variants",
      "Comma-separated list of GMRES variants to run benchmarks for. Options: CGS, CGSR, MGS, MGSR, CWY.",
      "all" );
   config.addEntry< TNL::String >(
      "preconditioners", "Comma-separated list of preconditioners to run benchmarks for. Options: jacobi, ilu0, ilut.", "all" );
   config.addEntry< bool >( "with-preconditioner-update", "Run benchmark for the preconditioner update.", true );
   config.addEntry< TNL::String >( "devices", "Run benchmarks on these devices.", "all" );
   config.addEntryEnum( "all" );
   config.addEntryEnum( "host" );
#ifdef __CUDACC__
   config.addEntryEnum( "cuda" );
#endif

   config.addDelimiter( "Device settings:" );
   TNL::Devices::Host::configSetup( config );
   TNL::Devices::Cuda::configSetup( config );
   TNL::MPI::configSetup( config );

   config.addDelimiter( "Linear solver settings:" );
   TNL::Solvers::IterativeSolver< double, int >::configSetup( config );
   using Matrix = TNL::Matrices::SparseMatrix< double >;
   using GMRES = TNL::Solvers::Linear::GMRES< Matrix >;
   GMRES::configSetup( config );
   using BiCGstabL = TNL::Solvers::Linear::BICGStabL< Matrix >;
   BiCGstabL::configSetup( config );
   using ILUT = TNL::Solvers::Linear::Preconditioners::ILUT< Matrix >;
   ILUT::configSetup( config );

   config.addEntry< TNL::String >( "precision", "Precision of the solver.", "double" );
   config.addEntryEnum( "double" );
   config.addEntryEnum( "float" );
}

int
main( int argc, char* argv[] )
{
#ifndef NDEBUG
   TNL::Debugging::trackFloatingPointExceptions();
#endif

   TNL::Config::ParameterContainer parameters;
   TNL::Config::ConfigDescription conf_desc;

   configSetup( conf_desc );

   TNL::MPI::ScopedInitializer mpi( argc, argv );
   const int rank = TNL::MPI::GetRank();

#ifdef HAVE_TRILINOS
   Kokkos::initialize( argc, argv );
#endif

   if( ! parseCommandLine( argc, argv, conf_desc, parameters ) )
      return EXIT_FAILURE;
   if( ! TNL::Devices::Host::setup( parameters ) || ! TNL::Devices::Cuda::setup( parameters )
       || ! TNL::MPI::setup( parameters ) )
      return EXIT_FAILURE;

   const TNL::String& logFileName = parameters.getParameter< TNL::String >( "log-file" );
   const TNL::String& outputMode = parameters.getParameter< TNL::String >( "output-mode" );
   const int loops = parameters.getParameter< int >( "loops" );
   const int verbose = ( rank == 0 ) ? parameters.getParameter< int >( "verbose" ) : 0;

   // open log file
   auto mode = std::ios::out;
   if( outputMode == "append" )
      mode |= std::ios::app;
   std::ofstream logFile;
   if( rank == 0 )
      logFile.open( logFileName, mode );

   // init benchmark and set parameters
   TNL::Benchmarks::Benchmark<> benchmark( logFile, loops, verbose );

   // write global metadata into a separate file
   std::map< std::string, std::string > metadata = TNL::Benchmarks::getHardwareMetadata();
   TNL::Benchmarks::writeMapAsJson( metadata, logFileName, ".metadata.json" );

   // TODO: implement resolveMatrixType
   //   return ! Matrices::resolveMatrixType< MainConfig,
   //                                         Devices::Host,
   //                                         LinearSolversBenchmark >( benchmark, parameters );
   auto precision = parameters.getParameter< TNL::String >( "precision" );
   bool ret_code = false;
   if( precision == "float" ) {
      using MatrixType =
         TNL::Matrices::SparseMatrix< float, TNL::Devices::Host, int, TNL::Matrices::GeneralMatrix, SegmentsType >;
      ret_code = LinearSolversBenchmark< MatrixType >::run( benchmark, parameters );
   }

   if( precision == "double" ) {
      using MatrixType =
         TNL::Matrices::SparseMatrix< double, TNL::Devices::Host, int, TNL::Matrices::GeneralMatrix, SegmentsType >;
      ret_code = LinearSolversBenchmark< MatrixType >::run( benchmark, parameters );
   }

#ifdef HAVE_TRILINOS
   Kokkos::finalize();
#endif

   if( ret_code )
      return EXIT_SUCCESS;
   return EXIT_FAILURE;
}
