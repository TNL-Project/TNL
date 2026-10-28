// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#if defined( __CUDACC__ )

   #include <stdexcept>
   #include <string>
   #include <type_traits>
   #include <vector>

   #include <cusparse.h>

   #include <TNL/Benchmarks/Benchmark.h>
   #include <TNL/Containers/Vector.h>
   #include <TNL/Devices/GPU.h>
   #include <TNL/Devices/Host.h>

   #include "LaplaceProblem.h"
   #include "Utils.h"

namespace TNL::Benchmarks::MatrixAssembly {

inline void
checkCusparse( cusparseStatus_t status, const char* call )
{
   if( status != CUSPARSE_STATUS_SUCCESS )
      throw std::runtime_error( std::string( call ) + " failed: " + cusparseGetErrorString( status ) );
}

   #define TNL_CUSPARSE_CALL( call ) checkCusparse( call, #call )

/**
 * \brief Benchmarks the assembly of a cuSPARSE CSR matrix.
 *
 * cuSPARSE has no assembly routine, the matrix is assembled in the COO format and
 * converted to CSR by `cusparseXcoo2csr`:
 *
 * - `allocation`: allocation of the COO and CSR arrays,
 * - `filling`: generating the COO triplets on the GPU (sorted in the row-major order,
 *   so no sorting is needed), `cusparseXcoo2csr` and creating the matrix descriptor,
 * - `assembly`: allocation + filling.
 *
 * The assembly of the CSR arrays on the host followed by their transfer to the GPU and
 * creating the matrix descriptor is measured in addition (see \ref timeHostAssemblyAndTransfer).
 */
template< typename Real, typename Index >
void
benchmarkCusparse( Benchmark& benchmark, const LaplaceProblem< Real, Index >& problem, bool verify )
{
   static_assert( std::is_same_v< Index, int >, "cusparseXcoo2csr supports only 32-bit indexes" );
   using Device = Devices::GPU;
   using IndexVector = Containers::Vector< Index, Device, Index >;
   using RealVector = Containers::Vector< Real, Device, Index >;

   benchmark.setMetadataElement( { "format", "CSR" } );

   const Index rows = problem.getRows();
   const Index nonzeros = problem.getNonzeroElementsCount();
   const cudaDataType valueType = std::is_same_v< Real, float > ? CUDA_R_32F : CUDA_R_64F;

   cusparseHandle_t handle = nullptr;
   TNL_CUSPARSE_CALL( cusparseCreate( &handle ) );

   cusparseSpMatDescr_t descriptor = nullptr;
   IndexVector offsets;
   IndexVector rowPointers;
   IndexVector cooRows;
   IndexVector columnIndexes;
   RealVector values;

   auto destroy = [ & ]()
   {
      if( descriptor != nullptr )
         TNL_CUSPARSE_CALL( cusparseDestroySpMat( descriptor ) );
      descriptor = nullptr;
      offsets.reset();
      rowPointers.reset();
      cooRows.reset();
      columnIndexes.reset();
      values.reset();
   };
   auto destroyDescriptor = [ & ]()
   {
      if( descriptor != nullptr )
         TNL_CUSPARSE_CALL( cusparseDestroySpMat( descriptor ) );
      descriptor = nullptr;
   };
   auto allocate = [ & ]()
   {
      rowPointers.setSize( rows + 1 );
      cooRows.setSize( nonzeros );
      columnIndexes.setSize( nonzeros );
      values.setSize( nonzeros );
   };
   auto fill = [ & ]()
   {
      problem.computeRowOffsets( offsets );
      problem.writeCOO( offsets, cooRows.getData(), columnIndexes.getData(), values.getData() );
      TNL_CUSPARSE_CALL(
         cusparseXcoo2csr( handle, cooRows.getData(), nonzeros, rows, rowPointers.getData(), CUSPARSE_INDEX_BASE_ZERO ) );
      TNL_CUSPARSE_CALL( cusparseCreateCsr(
         &descriptor,
         rows,
         rows,
         nonzeros,
         rowPointers.getData(),
         columnIndexes.getData(),
         values.getData(),
         CUSPARSE_INDEX_32I,
         CUSPARSE_INDEX_32I,
         CUSPARSE_INDEX_BASE_ZERO,
         valueType ) );
   };
   auto assemble = [ & ]()
   {
      allocate();
      fill();
   };

   timeOperation< Device >( benchmark, "allocation", "cuSPARSE", destroy, allocate );
   timeOperation< Device >( benchmark, "filling", "cuSPARSE", destroyDescriptor, fill );
   timeOperation< Device >( benchmark, "assembly", "cuSPARSE", destroy, assemble );

   auto verifyMatrix = [ & ]()
   {
      const Containers::Vector< Index, Devices::Host, Index > hostRowPointersVector( rowPointers );
      const Containers::Vector< Index, Devices::Host, Index > hostColumnIndexesVector( columnIndexes );
      const Containers::Vector< Real, Devices::Host, Index > hostValuesVector( values );
      const std::vector< Index > hostRowPointers(
         hostRowPointersVector.getData(), hostRowPointersVector.getData() + hostRowPointersVector.getSize() );
      const std::vector< Index > hostColumnIndexes(
         hostColumnIndexesVector.getData(), hostColumnIndexesVector.getData() + hostColumnIndexesVector.getSize() );
      const std::vector< Real > hostValues(
         hostValuesVector.getData(), hostValuesVector.getData() + hostValuesVector.getSize() );
      reportVerification( benchmark, "cuSPARSE", problem.compare( hostRowPointers, hostColumnIndexes, hostValues ) );
   };

   if( verify )
      verifyMatrix();
   destroy();

   // assembly of the CSR arrays on the host followed by the transfer to the GPU
   Containers::Vector< Index, Devices::Host, Index > hostRowPointers;
   Containers::Vector< Index, Devices::Host, Index > hostColumnIndexes;
   Containers::Vector< Real, Devices::Host, Index > hostValues;
   auto resetDevice = [ & ]()
   {
      destroyDescriptor();
      rowPointers.reset();
      columnIndexes.reset();
      values.reset();
   };
   auto resetAll = [ & ]()
   {
      resetDevice();
      hostRowPointers.reset();
      hostColumnIndexes.reset();
      hostValues.reset();
   };
   auto hostAssemble = [ & ]()
   {
      // the row offsets are the CSR row pointers
      problem.computeRowOffsets( hostRowPointers );
      hostColumnIndexes.setSize( nonzeros );
      hostValues.setSize( nonzeros );
      problem.writeCOO( hostRowPointers, (Index*) nullptr, hostColumnIndexes.getData(), hostValues.getData() );
   };
   auto transfer = [ & ]()
   {
      rowPointers = hostRowPointers;
      columnIndexes = hostColumnIndexes;
      values = hostValues;
      TNL_CUSPARSE_CALL( cusparseCreateCsr(
         &descriptor,
         rows,
         rows,
         nonzeros,
         rowPointers.getData(),
         columnIndexes.getData(),
         values.getData(),
         CUSPARSE_INDEX_32I,
         CUSPARSE_INDEX_32I,
         CUSPARSE_INDEX_BASE_ZERO,
         valueType ) );
   };
   timeHostAssemblyAndTransfer< Device >( benchmark, "cuSPARSE", resetAll, resetDevice, hostAssemble, transfer );

   if( verify )
      verifyMatrix();
   resetAll();

   TNL_CUSPARSE_CALL( cusparseDestroy( handle ) );
}

}  // namespace TNL::Benchmarks::MatrixAssembly

#endif  // __CUDACC__
