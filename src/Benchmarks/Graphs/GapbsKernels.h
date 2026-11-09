// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

/**
 * Wrappers of the kernels of the GAP Benchmark Suite reference code (gapbs, https://github.com/sbeamer/gapbs).
 *
 * The kernels are compiled in a separate translation unit (GapbsKernels.cpp) with the host compiler and OpenMP,
 * so this header does not expose any gapbs types and can be included in CUDA and HIP sources. The graphs are
 * loaded by gapbs itself from its serialized `.sg` and `.wsg` files.
 *
 * The `run` methods call the kernels exactly as the gapbs benchmark drivers do. The kernels allocate their outputs,
 * so the allocation is included in the measured time as required by GAP. The outputs are kept inside the wrappers
 * and can be copied out after the measurement.
 */
namespace TNL::Benchmarks::Graphs::Gapbs {

//! \brief Direction-optimizing BFS (`DOBFS` from `bfs.cc`), computes the BFS parents.
class BFS
{
public:
   explicit BFS( const std::string& sgFile );
   ~BFS();

   [[nodiscard]] std::int64_t
   getVertexCount() const;

   //! \brief Releases the output of the last run, so that the next run allocates it again.
   void
   release();

   void
   run( std::int32_t source );

   //! \brief Returns the parents of the last run, -1 for unreached vertices.
   void
   getParents( std::vector< std::int32_t >& parents ) const;

private:
   struct Impl;
   std::unique_ptr< Impl > impl;
};

//! \brief Delta-stepping SSSP (`DeltaStep` from `sssp.cc`), computes the distances with 32-bit integer weights.
class SSSP
{
public:
   explicit SSSP( const std::string& wsgFile );
   ~SSSP();

   [[nodiscard]] std::int64_t
   getVertexCount() const;

   //! \brief Releases the output of the last run, so that the next run allocates it again.
   void
   release();

   void
   run( std::int32_t source, std::int32_t delta );

   //! \brief Returns the distances of the last run, -1 for unreached vertices.
   void
   getDistances( std::vector< std::int64_t >& distances ) const;

private:
   struct Impl;
   std::unique_ptr< Impl > impl;
};

//! \brief Afforest connected components (`Afforest` from `cc.cc`).
class CC
{
public:
   explicit CC( const std::string& sgFile );
   ~CC();

   [[nodiscard]] std::int64_t
   getVertexCount() const;

   //! \brief Releases the output of the last run, so that the next run allocates it again.
   void
   release();

   void
   run();

   //! \brief Returns the component labels of the last run (vertex indexes).
   void
   getComponents( std::vector< std::int32_t >& components ) const;

private:
   struct Impl;
   std::unique_ptr< Impl > impl;
};

}  // namespace TNL::Benchmarks::Graphs::Gapbs
