// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#ifdef HAVE_HYPRE

   #include <string>
   #include <type_traits>
   #include <vector>

   #include <TNL/Algorithms/parallelFor.h>
   #include <TNL/Benchmarks/Benchmark.h>
   #include <TNL/Containers/Vector.h>
   #include <TNL/Devices/GPU.h>
   #include <TNL/Devices/Host.h>
   #include <HYPRE.h>
   #include <TNL/Hypre.h>

   #include "LaplaceProblem.h"
   #include "Utils.h"

namespace TNL::Benchmarks::MatrixAssembly {

//! \brief Device on which the Hypre build operates (Hypre is either a CPU or a GPU library).
   #ifdef HYPRE_USING_GPU
using HypreDevice = Devices::GPU;
   #else
using HypreDevice = Devices::Host;
   #endif

/**
 * \brief Benchmarks the assembly of a Hypre ParCSR matrix via the IJ interface on one MPI rank.
 *
 * The matrix is filled with one bulk call of `HYPRE_IJMatrixSetValues2` for all rows,
 * i.e. not element by element:
 *
 * - `allocation`: creating the IJ matrix, setting the row sizes (CPU builds only, GPU
 *   builds of Hypre do not use them), `HYPRE_IJMatrixInitialize` and allocating the input
 *   arrays for `HYPRE_IJMatrixSetValues2`,
 * - `filling`: generating the input arrays on the device, `HYPRE_IJMatrixSetValues2`
 *   and `HYPRE_IJMatrixAssemble`,
 * - `assembly`: allocation + filling.
 *
 * The benchmark runs only for the precision Hypre was built with (\e HYPRE_Real) and on the
 * device matching the Hypre build.
 */
template< typename Real, typename Device, typename Index >
void
benchmarkHypre( Benchmark& benchmark, const LaplaceProblem< Real, Index >& problem, bool verify )
{
   if constexpr( std::is_same_v< Real, HYPRE_Real > && std::is_same_v< Index, HYPRE_BigInt >
                 && std::is_same_v< Device, HypreDevice > )
   {
      benchmark.setMetadataElement( { "format", "CSR" } );

   #ifdef HYPRE_USING_GPU
      const HYPRE_MemoryLocation memoryLocation = HYPRE_MEMORY_DEVICE;
      HYPRE_SetMemoryLocation( HYPRE_MEMORY_DEVICE );
      HYPRE_SetExecutionPolicy( HYPRE_EXEC_DEVICE );
   #else
      const HYPRE_MemoryLocation memoryLocation = HYPRE_MEMORY_HOST;
   #endif

      const Index rows = problem.getRows();
      const Index nonzeros = problem.getNonzeroElementsCount();

      HYPRE_IJMatrix matrix = nullptr;
      Containers::Vector< HYPRE_Int, Devices::Host, Index > diagSizes;
      Containers::Vector< HYPRE_Int, Devices::Host, Index > offdSizes;
      Containers::Vector< Index, Device, Index > offsets;
      Containers::Vector< HYPRE_Int, Device, Index > rowLengths;
      Containers::Vector< HYPRE_BigInt, Device, Index > rowIndexes;
      Containers::Vector< HYPRE_BigInt, Device, Index > columnIndexes;
      Containers::Vector< HYPRE_Real, Device, Index > values;

      auto destroy = [ & ]()
      {
         if( matrix != nullptr )
            HYPRE_IJMatrixDestroy( matrix );
         matrix = nullptr;
         diagSizes.reset();
         offdSizes.reset();
         offsets.reset();
         rowLengths.reset();
         rowIndexes.reset();
         columnIndexes.reset();
         values.reset();
      };
      auto allocate = [ & ]()
      {
         HYPRE_IJMatrixCreate( MPI_COMM_WORLD, 0, rows - 1, 0, rows - 1, &matrix );
         HYPRE_IJMatrixSetObjectType( matrix, HYPRE_PARCSR );
   #ifndef HYPRE_USING_GPU
         diagSizes.setSize( rows );
         offdSizes.setSize( rows );
         offdSizes.setValue( 0 );
         auto diagSizesView = diagSizes.getView();
         Algorithms::parallelFor< Devices::Host >(
            0,
            rows,
            [ = ]( Index row ) mutable
            {
               diagSizesView[ row ] = problem.getRowLength( row );
            } );
         HYPRE_IJMatrixSetDiagOffdSizes( matrix, diagSizes.getData(), offdSizes.getData() );
   #endif
         HYPRE_IJMatrixInitialize_v2( matrix, memoryLocation );
         rowLengths.setSize( rows );
         rowIndexes.setSize( rows );
         columnIndexes.setSize( nonzeros );
         values.setSize( nonzeros );
      };
      auto fill = [ & ]()
      {
         problem.computeRowOffsets( offsets );
         problem.writeCOO( offsets, (HYPRE_BigInt*) nullptr, columnIndexes.getData(), values.getData() );
         auto rowLengthsView = rowLengths.getView();
         auto rowIndexesView = rowIndexes.getView();
         Algorithms::parallelFor< Device >(
            0,
            rows,
            [ = ] __cuda_callable__( Index row ) mutable
            {
               rowLengthsView[ row ] = problem.getRowLength( row );
               rowIndexesView[ row ] = row;
            } );
         HYPRE_IJMatrixSetValues2(
            matrix,
            rows,
            rowLengths.getData(),
            rowIndexes.getData(),
            offsets.getData(),
            columnIndexes.getData(),
            values.getData() );
         HYPRE_IJMatrixAssemble( matrix );
      };
      auto assemble = [ & ]()
      {
         allocate();
         fill();
      };
      auto reallocate = [ & ]()
      {
         destroy();
         allocate();
      };

      timeOperation< Device >( benchmark, "allocation", "Hypre", destroy, allocate );
      // the IJ matrix must be initialized again before setting the values
      timeOperation< Device >( benchmark, "filling", "Hypre", reallocate, fill );
      timeOperation< Device >( benchmark, "assembly", "Hypre", destroy, assemble );

      if( verify ) {
         hypre_ParCSRMatrix* parcsr = nullptr;
         HYPRE_IJMatrixGetObject( matrix, (void**) &parcsr );
         hypre_CSRMatrix* diag = hypre_ParCSRMatrixDiag( parcsr );
         hypre_CSRMatrix* offd = hypre_ParCSRMatrixOffd( parcsr );
         const HYPRE_MemoryLocation location = hypre_CSRMatrixMemoryLocation( diag );
         const HYPRE_Int nnz = hypre_CSRMatrixNumNonzeros( diag );
         std::vector< HYPRE_Int > rowPointers( rows + 1 );
         std::vector< HYPRE_Int > columns( nnz );
         std::vector< HYPRE_Real > data( nnz );
         hypre_TMemcpy( rowPointers.data(), hypre_CSRMatrixI( diag ), HYPRE_Int, rows + 1, HYPRE_MEMORY_HOST, location );
         hypre_TMemcpy( columns.data(), hypre_CSRMatrixJ( diag ), HYPRE_Int, nnz, HYPRE_MEMORY_HOST, location );
         hypre_TMemcpy( data.data(), hypre_CSRMatrixData( diag ), HYPRE_Real, nnz, HYPRE_MEMORY_HOST, location );
         std::string error = problem.compare( rowPointers, columns, data );
         if( error.empty() && hypre_CSRMatrixNumNonzeros( offd ) != 0 )
            error = "the off-diagonal block is not empty";
         reportVerification( benchmark, "Hypre", error );
      }
      destroy();
   }
}

}  // namespace TNL::Benchmarks::MatrixAssembly

#endif  // HAVE_HYPRE
