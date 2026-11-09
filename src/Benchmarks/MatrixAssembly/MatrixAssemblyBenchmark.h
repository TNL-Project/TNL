// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <exception>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <TNL/Algorithms/Segments/AdaptiveCSR.h>
#include <TNL/Algorithms/Segments/BiEllpack.h>
#include <TNL/Algorithms/Segments/CSR.h>
#include <TNL/Algorithms/Segments/ChunkedEllpack.h>
#include <TNL/Algorithms/Segments/Ellpack.h>
#include <TNL/Algorithms/Segments/SlicedEllpack.h>
#include <TNL/Benchmarks/Benchmark.h>
#include <TNL/Config/ParameterContainer.h>
#include <TNL/Devices/GPU.h>
#include <TNL/Devices/Host.h>
#include <TNL/TypeInfo.h>

#include "CusparseAssembly.h"
#include "GinkgoAssembly.h"
#include "GraphProblem.h"
#include "HypreAssembly.h"
#include "LaplaceProblem.h"
#include "PetscAssembly.h"
#include "TNLAssembly.h"

namespace TNL::Benchmarks::MatrixAssembly {

inline const std::vector< std::string > validLibraries = { "TNL", "Ginkgo", "Hypre", "PETSc", "cuSPARSE" };
inline const std::vector< std::string > validFormats = {
   "CSR",       "AdaptiveCSR",       "Ellpack",       "SlicedEllpack",       "ChunkedEllpack",       "BiEllpack",
   "SortedCSR", "SortedAdaptiveCSR", "SortedEllpack", "SortedSlicedEllpack", "SortedChunkedEllpack", "SortedBiEllpack"
};

//! \brief Resolves a list of values from the command line where "all" stands for all valid values.
inline std::set< std::string >
resolveList( const std::vector< std::string >& list, const std::vector< std::string >& validValues, const std::string& name )
{
   std::set< std::string > result;
   for( const auto& value : list ) {
      if( value == "all" )
         result.insert( validValues.begin(), validValues.end() );
      else if( std::find( validValues.begin(), validValues.end(), value ) != validValues.end() )
         result.insert( value );
      else
         throw std::invalid_argument( "Invalid value '" + value + "' of the parameter '" + name + "'." );
   }
   return result;
}

//! \brief Runs the benchmark of one library and reports exceptions as errors.
template< typename Function >
void
runSafely( Benchmark& benchmark, const std::string& library, Function&& function )
{
   try {
      function();
   }
   catch( const std::exception& e ) {
      benchmark.addErrorMessage( library + " benchmark failed: " + e.what() );
   }
}

//! \brief Benchmarks the TNL sparse matrix with the given segments if the format is selected.
template<
   typename Real,
   typename Device,
   typename Index,
   template< typename, typename, typename > class Segments,
   typename Problem >
void
runTNLFormat(
   Benchmark& benchmark,
   const Problem& problem,
   const std::set< std::string >& formats,
   const std::string& format,
   bool verify )
{
   if( formats.count( format ) == 0 )
      return;
   runSafely(
      benchmark,
      "TNL",
      [ & ]()
      {
         benchmarkTNL< Real, Device, Index, Segments >( benchmark, problem, format, verify );
      } );
}

template< typename Real, typename Device, typename Index, typename Problem >
void
runOnDevice(
   Benchmark& benchmark,
   const Problem& problem,
   const std::set< std::string >& libraries,
   const std::set< std::string >& formats,
   bool verify )
{
   if( libraries.count( "TNL" ) > 0 ) {
      using namespace Algorithms::Segments;
      runTNLFormat< Real, Device, Index, CSR >( benchmark, problem, formats, "CSR", verify );
      runTNLFormat< Real, Device, Index, AdaptiveCSR >( benchmark, problem, formats, "AdaptiveCSR", verify );
      runTNLFormat< Real, Device, Index, Ellpack >( benchmark, problem, formats, "Ellpack", verify );
      runTNLFormat< Real, Device, Index, SlicedEllpack >( benchmark, problem, formats, "SlicedEllpack", verify );
      runTNLFormat< Real, Device, Index, ChunkedEllpack >( benchmark, problem, formats, "ChunkedEllpack", verify );
      runTNLFormat< Real, Device, Index, BiEllpack >( benchmark, problem, formats, "BiEllpack", verify );
      runTNLFormat< Real, Device, Index, SortedCSR >( benchmark, problem, formats, "SortedCSR", verify );
      runTNLFormat< Real, Device, Index, SortedAdaptiveCSR >( benchmark, problem, formats, "SortedAdaptiveCSR", verify );
      runTNLFormat< Real, Device, Index, SortedEllpack >( benchmark, problem, formats, "SortedEllpack", verify );
      runTNLFormat< Real, Device, Index, SortedSlicedEllpack >( benchmark, problem, formats, "SortedSlicedEllpack", verify );
      runTNLFormat< Real, Device, Index, SortedChunkedEllpack >( benchmark, problem, formats, "SortedChunkedEllpack", verify );
      runTNLFormat< Real, Device, Index, SortedBiEllpack >( benchmark, problem, formats, "SortedBiEllpack", verify );
   }

#ifdef HAVE_GINKGO
   if( libraries.count( "Ginkgo" ) > 0 )
      runSafely(
         benchmark,
         "Ginkgo",
         [ & ]()
         {
            benchmarkGinkgo< Real, Device, Index >( benchmark, problem, formats, verify );
         } );
#endif

#ifdef HAVE_HYPRE
   if( libraries.count( "Hypre" ) > 0 && formats.count( "CSR" ) > 0 )
      runSafely(
         benchmark,
         "Hypre",
         [ & ]()
         {
            benchmarkHypre< Real, Device, Index >( benchmark, problem, verify );
         } );
#endif

#ifdef HAVE_PETSC
   if( libraries.count( "PETSc" ) > 0 )
      runSafely(
         benchmark,
         "PETSc",
         [ & ]()
         {
            benchmarkPetsc< Real, Device, Index >( benchmark, problem, formats, verify );
         } );
#endif

#if defined( __CUDACC__ )
   if constexpr( std::is_same_v< Device, Devices::GPU > ) {
      if( libraries.count( "cuSPARSE" ) > 0 && formats.count( "CSR" ) > 0 )
         runSafely(
            benchmark,
            "cuSPARSE",
            [ & ]()
            {
               benchmarkCusparse< Real, Index >( benchmark, problem, verify );
            } );
   }
#endif
}

//! \brief Returns the grid size along the given axis (the default size is used if the parameter is zero).
inline int
getAxisSize( const Config::ParameterContainer& parameters, const std::string& axis, int dimension )
{
   const int size = parameters.getParameter< int >( axis + "-size" );
   if( size > 0 )
      return size;
   // default sizes give matrices with about 10^6 rows
   switch( dimension ) {
      case 1:
         return 1000000;
      case 2:
         return 1000;
      default:
         return 100;
   }
}

//! \brief Sets the metadata of the problem and runs the benchmark on the selected devices.
template< typename Real, typename Index, typename Problem >
void
runProblem( Benchmark& benchmark, const Config::ParameterContainer& parameters, const Problem& problem )
{
   const auto& device = parameters.getParameter< std::string >( "device" );
   const bool verify = parameters.getParameter< bool >( "verify" );
   const auto libraries = resolveList( parameters.getList< std::string >( "libraries" ), validLibraries, "libraries" );
   const auto formats = resolveList( parameters.getList< std::string >( "formats" ), validFormats, "formats" );

   benchmark.setMetadataColumns(
      Benchmark::MetadataColumns(
         {
            { "precision", getType< Real >() },
            { "problem", problem.getName() },
            { "size", problem.getSizeString() },
            { "rows", std::to_string( problem.getRows() ) },
            { "nonzeros", std::to_string( problem.getNonzeroElementsCount() ) },
            { "max row length", std::to_string( problem.getMaxRowLength() ) },
         } ) );
   // amount of data of the assembled matrix in the CSR format
   benchmark.setDatasetSize(
      problem.getNonzeroElementsCount() * ( sizeof( Real ) + sizeof( Index ) ) + ( problem.getRows() + 1 ) * sizeof( Index ) );

   if( device == "host" || device == "all" )
      runOnDevice< Real, Devices::Host, Index >( benchmark, problem, libraries, formats, verify );

#if defined( __CUDACC__ ) || defined( __HIP__ )
   if( device == "cuda" || device == "hip" || device == "all" )
      runOnDevice< Real, Devices::GPU, Index >( benchmark, problem, libraries, formats, verify );
#endif
}

template< typename Real, typename Index >
void
runBenchmark( Benchmark& benchmark, const Config::ParameterContainer& parameters )
{
   const auto& problemParameter = parameters.getParameter< std::string >( "problem" );
   const auto& dimensionParameter = parameters.getParameter< std::string >( "dimension" );

   if( problemParameter == "laplace" || problemParameter == "all" ) {
      for( int dimension = 1; dimension <= 3; dimension++ ) {
         if( dimensionParameter != "all" && dimensionParameter != std::to_string( dimension ) )
            continue;
         const LaplaceProblem< Real, Index > problem(
            dimension,
            getAxisSize( parameters, "x", dimension ),
            getAxisSize( parameters, "y", dimension ),
            getAxisSize( parameters, "z", dimension ) );
         runProblem< Real, Index >( benchmark, parameters, problem );
      }
   }

   if( problemParameter == "graph" || problemParameter == "all" ) {
      const GraphProblem< Real, Index > problem(
         parameters.getParameter< int >( "graph-scale" ),
         parameters.getParameter< int >( "graph-edge-factor" ),
         parameters.getParameter< int >( "graph-seed" ) );
      runProblem< Real, Index >( benchmark, parameters, problem );
   }
}

}  // namespace TNL::Benchmarks::MatrixAssembly
