// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#ifdef HAVE_PETSC

   #include <set>
   #include <stdexcept>
   #include <string>
   #include <type_traits>
   #include <vector>

   #include <petscmat.h>

   #include <TNL/Benchmarks/Benchmark.h>
   #include <TNL/Containers/Vector.h>
   #include <TNL/Devices/GPU.h>
   #include <TNL/Devices/Host.h>

   #include "LaplaceProblem.h"
   #include "Utils.h"

namespace TNL::Benchmarks::MatrixAssembly {

//! \brief Throws an exception when a PETSc call fails.
inline void
checkPetsc( PetscErrorCode error, const char* call )
{
   if( error != PETSC_SUCCESS ) {
      const char* message = nullptr;
      PetscErrorMessage( error, &message, nullptr );
      throw std::runtime_error( std::string( call ) + " failed: " + ( message != nullptr ? message : "unknown error" ) );
   }
}

   #define TNL_PETSC_CALL( call ) checkPetsc( call, #call )

//! \brief Reads the matrix row by row and compares it with the exact matrix.
template< typename Real, typename Index >
void
verifyPetscMatrix( Benchmark& benchmark, const LaplaceProblem< Real, Index >& problem, Mat matrix )
{
   std::vector< PetscInt > rowPointers( 1, 0 );
   std::vector< PetscInt > columnIndexes;
   std::vector< PetscScalar > data;
   for( PetscInt row = 0; row < problem.getRows(); row++ ) {
      PetscInt size = 0;
      const PetscInt* columns = nullptr;
      const PetscScalar* rowValues = nullptr;
      TNL_PETSC_CALL( MatGetRow( matrix, row, &size, &columns, &rowValues ) );
      columnIndexes.insert( columnIndexes.end(), columns, columns + size );
      data.insert( data.end(), rowValues, rowValues + size );
      TNL_PETSC_CALL( MatRestoreRow( matrix, row, &size, &columns, &rowValues ) );
      rowPointers.push_back( columnIndexes.size() );
   }
   reportVerification( benchmark, "PETSc", problem.compare( rowPointers, columnIndexes, data ) );
}

//! \brief Assembles a MATSEQAIJ matrix on the host with the COO interface.
template< typename Real, typename Index >
Mat
assemblePetscHostMatrix( const LaplaceProblem< Real, Index >& problem )
{
   const Index rows = problem.getRows();
   const Index nonzeros = problem.getNonzeroElementsCount();
   Mat matrix = nullptr;
   TNL_PETSC_CALL( MatCreate( PETSC_COMM_SELF, &matrix ) );
   TNL_PETSC_CALL( MatSetSizes( matrix, rows, rows, rows, rows ) );
   TNL_PETSC_CALL( MatSetType( matrix, MATSEQAIJ ) );
   Containers::Vector< PetscInt, Devices::Host, Index > offsets;
   Containers::Vector< PetscInt, Devices::Host, Index > cooRows( nonzeros );
   Containers::Vector< PetscInt, Devices::Host, Index > cooColumns( nonzeros );
   Containers::Vector< PetscScalar, Devices::Host, Index > values( nonzeros );
   problem.computeRowOffsets( offsets );
   problem.writeCOO( offsets, cooRows.getData(), cooColumns.getData(), values.getData() );
   TNL_PETSC_CALL( MatSetPreallocationCOO( matrix, nonzeros, cooRows.getData(), cooColumns.getData() ) );
   TNL_PETSC_CALL( MatSetValuesCOO( matrix, values.getData(), INSERT_VALUES ) );
   return matrix;
}

/**
 * \brief Benchmarks the assembly of a CSR matrix on the host followed by the transfer to the GPU.
 *
 * The matrix is assembled as MATSEQAIJ with the COO interface and converted to the GPU type
 * (MATSEQAIJCUSPARSE or MATSEQAIJHIPSPARSE) with `MatConvert`. PETSc copies the matrix to the
 * GPU lazily, so the copy is forced by getting the array of values on the device.
 * See \ref timeHostAssemblyAndTransfer for the measured operations.
 */
template< typename Real, typename Device, typename Index >
void
benchmarkPetscHostToGPU( Benchmark& benchmark, const LaplaceProblem< Real, Index >& problem, bool verify )
{
   #if ( defined( __CUDACC__ ) && defined( PETSC_HAVE_CUDA ) ) || ( defined( __HIP__ ) && defined( PETSC_HAVE_HIP ) )
   benchmark.setMetadataElement( { "format", "CSR" } );

   Mat hostMatrix = nullptr;
   Mat matrix = nullptr;
   auto resetDevice = [ & ]()
   {
      if( matrix != nullptr )
         TNL_PETSC_CALL( MatDestroy( &matrix ) );
      matrix = nullptr;
   };
   auto resetAll = [ & ]()
   {
      resetDevice();
      if( hostMatrix != nullptr )
         TNL_PETSC_CALL( MatDestroy( &hostMatrix ) );
      hostMatrix = nullptr;
   };
   auto hostAssemble = [ & ]()
   {
      hostMatrix = assemblePetscHostMatrix( problem );
   };
   auto transfer = [ & ]()
   {
      const PetscScalar* deviceValues = nullptr;
      #if defined( __CUDACC__ )
      TNL_PETSC_CALL( MatConvert( hostMatrix, MATSEQAIJCUSPARSE, MAT_INITIAL_MATRIX, &matrix ) );
      TNL_PETSC_CALL( MatSeqAIJCUSPARSEGetArrayRead( matrix, &deviceValues ) );
      TNL_PETSC_CALL( MatSeqAIJCUSPARSERestoreArrayRead( matrix, &deviceValues ) );
      #else
      TNL_PETSC_CALL( MatConvert( hostMatrix, MATSEQAIJHIPSPARSE, MAT_INITIAL_MATRIX, &matrix ) );
      TNL_PETSC_CALL( MatSeqAIJHIPSPARSEGetArrayRead( matrix, &deviceValues ) );
      TNL_PETSC_CALL( MatSeqAIJHIPSPARSERestoreArrayRead( matrix, &deviceValues ) );
      #endif
   };
   timeHostAssemblyAndTransfer< Device >( benchmark, "PETSc", resetAll, resetDevice, hostAssemble, transfer );

   if( verify && matrix != nullptr )
      verifyPetscMatrix( benchmark, problem, matrix );
   resetAll();
   #endif
}

/**
 * \brief Benchmarks the assembly of a PETSc matrix of the given type on one MPI rank.
 *
 * The matrix is assembled with the COO interface which is the most efficient way of
 * matrix assembly in PETSc, especially on GPUs:
 *
 * - `allocation`: creating the matrix, generating the COO row and column indexes on the
 *   host and `MatSetPreallocationCOO`, allocating the array of values on the device,
 * - `filling`: generating the values on the device and `MatSetValuesCOO` (which also
 *   assembles the matrix); for the SELL formats, the values are generated on the host because
 *   PETSc has no native COO assembly for SELL and its generic implementation reads them on the host,
 * - `assembly`: allocation + filling.
 *
 * The benchmark runs only for the precision PETSc was built with (\e PetscScalar).
 */
template< typename Real, typename Device, typename Index >
void
benchmarkPetscType(
   Benchmark& benchmark,
   const LaplaceProblem< Real, Index >& problem,
   const std::string& format,
   MatType matrixType,
   bool verify )
{
   benchmark.setMetadataElement( { "format", format } );

   const Index rows = problem.getRows();
   const Index nonzeros = problem.getNonzeroElementsCount();

   const bool isSell = std::string( matrixType ).find( "sell" ) != std::string::npos;
   Mat matrix = nullptr;
   Containers::Vector< PetscInt, Devices::Host, Index > hostOffsets;
   Containers::Vector< PetscInt, Devices::Host, Index > cooRows;
   Containers::Vector< PetscInt, Devices::Host, Index > cooColumns;
   Containers::Vector< Index, Device, Index > offsets;
   Containers::Vector< PetscScalar, Device, Index > values;
   Containers::Vector< PetscScalar, Devices::Host, Index > hostValues;

   auto destroy = [ & ]()
   {
      if( matrix != nullptr )
         TNL_PETSC_CALL( MatDestroy( &matrix ) );
      matrix = nullptr;
      hostOffsets.reset();
      cooRows.reset();
      cooColumns.reset();
      offsets.reset();
      values.reset();
      hostValues.reset();
   };
   auto allocate = [ & ]()
   {
      TNL_PETSC_CALL( MatCreate( PETSC_COMM_SELF, &matrix ) );
      TNL_PETSC_CALL( MatSetSizes( matrix, rows, rows, rows, rows ) );
      TNL_PETSC_CALL( MatSetType( matrix, matrixType ) );
      problem.computeRowOffsets( hostOffsets );
      if( isSell ) {
         // SELL has no native COO assembly, the generic implementation inserts the elements
         // with MatSetValues which requires preallocation
         Containers::Vector< PetscInt, Devices::Host, Index > rowLengths( rows );
         for( Index row = 0; row < rows; row++ )
            rowLengths[ row ] = hostOffsets[ row + 1 ] - hostOffsets[ row ];
         TNL_PETSC_CALL( MatSeqSELLSetPreallocation( matrix, 0, rowLengths.getData() ) );
      }
      cooRows.setSize( nonzeros );
      cooColumns.setSize( nonzeros );
      problem.writeCOO( hostOffsets, cooRows.getData(), cooColumns.getData(), (PetscScalar*) nullptr );
      TNL_PETSC_CALL( MatSetPreallocationCOO( matrix, nonzeros, cooRows.getData(), cooColumns.getData() ) );
      if( isSell )
         hostValues.setSize( nonzeros );
      else
         values.setSize( nonzeros );
   };
   auto fill = [ & ]()
   {
      if( isSell ) {
         // the generic COO implementation used for SELL reads the values on the host
         problem.writeCOO( hostOffsets, (PetscInt*) nullptr, (PetscInt*) nullptr, hostValues.getData() );
         TNL_PETSC_CALL( MatSetValuesCOO( matrix, hostValues.getData(), INSERT_VALUES ) );
      }
      else {
         problem.computeRowOffsets( offsets );
         problem.writeCOO( offsets, (PetscInt*) nullptr, (PetscInt*) nullptr, values.getData() );
         TNL_PETSC_CALL( MatSetValuesCOO( matrix, values.getData(), INSERT_VALUES ) );
      }
   };
   auto assemble = [ & ]()
   {
      allocate();
      fill();
   };

   timeOperation< Device >( benchmark, "allocation", "PETSc", destroy, allocate );
   timeOperation< Device >( benchmark, "filling", "PETSc", noReset, fill );
   timeOperation< Device >( benchmark, "assembly", "PETSc", destroy, assemble );

   if( verify && matrix != nullptr )
      verifyPetscMatrix( benchmark, problem, matrix );
   destroy();
}

template< typename Real, typename Device, typename Index >
void
benchmarkPetsc(
   Benchmark& benchmark,
   const LaplaceProblem< Real, Index >& problem,
   const std::set< std::string >& formats,
   bool verify )
{
   if constexpr( std::is_same_v< Real, PetscScalar > && std::is_same_v< Index, PetscInt > ) {
      if constexpr( std::is_same_v< Device, Devices::Host > ) {
         if( formats.count( "CSR" ) > 0 )
            benchmarkPetscType< Real, Device, Index >( benchmark, problem, "CSR", MATSEQAIJ, verify );
         if( formats.count( "SlicedEllpack" ) > 0 )
            benchmarkPetscType< Real, Device, Index >( benchmark, problem, "SlicedEllpack", MATSEQSELL, verify );
      }
      else {
   #if defined( __CUDACC__ ) && defined( PETSC_HAVE_CUDA )
         if( formats.count( "CSR" ) > 0 )
            benchmarkPetscType< Real, Device, Index >( benchmark, problem, "CSR", MATSEQAIJCUSPARSE, verify );
         if( formats.count( "SlicedEllpack" ) > 0 )
            benchmarkPetscType< Real, Device, Index >( benchmark, problem, "SlicedEllpack", MATSEQSELLCUDA, verify );
   #elif defined( __HIP__ ) && defined( PETSC_HAVE_HIP )
         if( formats.count( "CSR" ) > 0 )
            benchmarkPetscType< Real, Device, Index >( benchmark, problem, "CSR", MATSEQAIJHIPSPARSE, verify );
         if( formats.count( "SlicedEllpack" ) > 0 )
            benchmarkPetscType< Real, Device, Index >( benchmark, problem, "SlicedEllpack", MATSEQSELLHIP, verify );
   #endif
         if( formats.count( "CSR" ) > 0 )
            benchmarkPetscHostToGPU< Real, Device, Index >( benchmark, problem, verify );
      }
   }
}

}  // namespace TNL::Benchmarks::MatrixAssembly

#endif  // HAVE_PETSC
