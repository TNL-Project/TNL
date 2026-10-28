// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#ifdef HAVE_GINKGO

   #include <memory>
   #include <set>
   #include <string>
   #include <vector>

   #include <ginkgo/ginkgo.hpp>

   #include <TNL/Backend/Functions.h>
   #include <TNL/Benchmarks/Benchmark.h>
   #include <TNL/Containers/Vector.h>
   #include <TNL/Devices/GPU.h>
   #include <TNL/Devices/Host.h>

   #include "LaplaceProblem.h"
   #include "Utils.h"

namespace TNL::Benchmarks::MatrixAssembly {

//! \brief Creates the Ginkgo executor corresponding to the TNL device.
template< typename Device >
std::shared_ptr< gko::Executor >
getGinkgoExecutor()
{
   if constexpr( std::is_same_v< Device, Devices::Host > )
      return gko::OmpExecutor::create();
   #if defined( __CUDACC__ )
   else
      return gko::CudaExecutor::create( Backend::getDevice(), gko::OmpExecutor::create() );
   #elif defined( __HIP__ )
   else
      return gko::HipExecutor::create( Backend::getDevice(), gko::OmpExecutor::create() );
   #endif
}

//! \brief Converts a Ginkgo matrix to CSR on the host and compares it with the exact matrix.
template< typename Real, typename Index, typename Format >
void
verifyGinkgoMatrix( Benchmark& benchmark, const LaplaceProblem< Real, Index >& problem, const Format& matrix )
{
   using Csr = gko::matrix::Csr< Real, Index >;
   auto csr = Csr::create( matrix.get_executor()->get_master() );
   matrix.convert_to( csr );
   const auto rows = static_cast< Index >( csr->get_size()[ 0 ] );
   const auto nnz = static_cast< Index >( csr->get_num_stored_elements() );
   std::vector< Index > rowPointers( csr->get_const_row_ptrs(), csr->get_const_row_ptrs() + rows + 1 );
   std::vector< Index > columnIndexes( csr->get_const_col_idxs(), csr->get_const_col_idxs() + nnz );
   std::vector< Real > values( csr->get_const_values(), csr->get_const_values() + nnz );
   reportVerification( benchmark, "Ginkgo", problem.compare( rowPointers, columnIndexes, values ) );
}

//! \brief Assembles a Ginkgo matrix from COO triplets generated on \e Device.
template< typename Device, typename Format, typename Real, typename Index >
std::unique_ptr< Format >
assembleGinkgoMatrix( const std::shared_ptr< gko::Executor >& exec, const LaplaceProblem< Real, Index >& problem )
{
   gko::device_matrix_data< Real, Index > data(
      exec, gko::dim< 2 >( problem.getRows(), problem.getRows() ), problem.getNonzeroElementsCount() );
   Containers::Vector< Index, Device, Index > offsets;
   problem.computeRowOffsets( offsets );
   problem.writeCOO( offsets, data.get_row_idxs(), data.get_col_idxs(), data.get_values() );
   auto matrix = Format::create( exec );
   matrix->read( std::move( data ) );
   return matrix;
}

/**
 * \brief Benchmarks the assembly of a Ginkgo matrix (Csr, Ell or Sellp).
 *
 * Ginkgo matrices are assembled from \e gko::device_matrix_data, i.e. COO arrays
 * allocated on the executor:
 *
 * - `allocation`: allocation of the COO arrays and creation of an empty matrix,
 * - `filling`: generating the COO triplets on the device and reading them into the matrix
 *   (conversion to the target format including the allocation of the matrix storage),
 * - `assembly`: allocation + filling.
 *
 * The triplets are generated sorted in the row-major order, so no sorting is needed.
 *
 * For GPU devices, the assembly on the OpenMP executor followed by `copy_from` to the GPU
 * executor is measured in addition (see \ref timeHostAssemblyAndTransfer).
 */
template< typename Real, typename Device, typename Index, typename Format >
void
benchmarkGinkgoFormat(
   Benchmark& benchmark,
   const LaplaceProblem< Real, Index >& problem,
   const std::string& format,
   bool verify )
{
   using DeviceData = gko::device_matrix_data< Real, Index >;

   benchmark.setMetadataElement( { "format", format } );

   const auto exec = getGinkgoExecutor< Device >();
   const gko::dim< 2 > size( problem.getRows(), problem.getRows() );
   const auto nonzeros = problem.getNonzeroElementsCount();

   std::unique_ptr< DeviceData > data;
   std::unique_ptr< Format > matrix;
   Containers::Vector< Index, Device, Index > offsets;

   auto allocate = [ & ]()
   {
      data = std::make_unique< DeviceData >( exec, size, nonzeros );
      matrix = Format::create( exec );
   };
   auto fill = [ & ]()
   {
      problem.computeRowOffsets( offsets );
      problem.writeCOO( offsets, data->get_row_idxs(), data->get_col_idxs(), data->get_values() );
      matrix->read( std::move( *data ) );
   };
   auto assemble = [ & ]()
   {
      allocate();
      fill();
   };
   auto resetAll = [ & ]()
   {
      data.reset();
      matrix.reset();
      offsets.reset();
   };

   timeOperation< Device >( benchmark, "allocation", "Ginkgo", resetAll, allocate );
   // reading the device data consumes them, so they must be allocated again
   timeOperation< Device >( benchmark, "filling", "Ginkgo", allocate, fill );
   timeOperation< Device >( benchmark, "assembly", "Ginkgo", resetAll, assemble );

   if( verify )
      verifyGinkgoMatrix( benchmark, problem, *matrix );

   if constexpr( ! std::is_same_v< Device, Devices::Host > ) {
      const auto hostExec = getGinkgoExecutor< Devices::Host >();
      std::unique_ptr< Format > hostMatrix;
      auto resetAll = [ & ]()
      {
         hostMatrix.reset();
         matrix.reset();
      };
      auto resetDevice = [ & ]()
      {
         matrix.reset();
      };
      auto hostAssemble = [ & ]()
      {
         hostMatrix = assembleGinkgoMatrix< Devices::Host, Format >( hostExec, problem );
      };
      auto transfer = [ & ]()
      {
         matrix = Format::create( exec );
         matrix->copy_from( hostMatrix.get() );
      };
      timeHostAssemblyAndTransfer< Device >( benchmark, "Ginkgo", resetAll, resetDevice, hostAssemble, transfer );

      if( verify && matrix != nullptr )
         verifyGinkgoMatrix( benchmark, problem, *matrix );
   }
}

template< typename Real, typename Device, typename Index >
void
benchmarkGinkgo(
   Benchmark& benchmark,
   const LaplaceProblem< Real, Index >& problem,
   const std::set< std::string >& formats,
   bool verify )
{
   if( formats.count( "CSR" ) > 0 )
      benchmarkGinkgoFormat< Real, Device, Index, gko::matrix::Csr< Real, Index > >( benchmark, problem, "CSR", verify );
   if( formats.count( "Ellpack" ) > 0 )
      benchmarkGinkgoFormat< Real, Device, Index, gko::matrix::Ell< Real, Index > >( benchmark, problem, "Ellpack", verify );
   if( formats.count( "SlicedEllpack" ) > 0 )
      benchmarkGinkgoFormat< Real, Device, Index, gko::matrix::Sellp< Real, Index > >(
         benchmark, problem, "SlicedEllpack", verify );
}

}  // namespace TNL::Benchmarks::MatrixAssembly

#endif  // HAVE_GINKGO
