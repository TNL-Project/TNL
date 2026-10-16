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

   if( verify ) {
      using Csr = gko::matrix::Csr< Real, Index >;
      auto csr = Csr::create( exec->get_master() );
      matrix->convert_to( csr );
      const auto rows = static_cast< Index >( csr->get_size()[ 0 ] );
      const auto nnz = static_cast< Index >( csr->get_num_stored_elements() );
      std::vector< Index > rowPointers( csr->get_const_row_ptrs(), csr->get_const_row_ptrs() + rows + 1 );
      std::vector< Index > columnIndexes( csr->get_const_col_idxs(), csr->get_const_col_idxs() + nnz );
      std::vector< Real > values( csr->get_const_values(), csr->get_const_values() + nnz );
      reportVerification( benchmark, "Ginkgo", problem.compare( rowPointers, columnIndexes, values ) );
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
