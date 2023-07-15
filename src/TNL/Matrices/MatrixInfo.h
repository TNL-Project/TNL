// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Matrices/DenseMatrixView.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Matrices/SparseMatrixView.h>
#include <TNL/Matrices/Sandbox/SparseSandboxMatrix.h>
#include <TNL/Algorithms/Segments/CSRView.h>
#include <TNL/Algorithms/Segments/EllpackView.h>
#include <TNL/Algorithms/Segments/SlicedEllpackView.h>

namespace TNL::Matrices {

template< typename Matrix >
struct MatrixInfo
{};

/// This is to prevent from appearing in Doxygen documentation.
/// \cond HIDDEN_CLASS
template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
struct MatrixInfo< DenseMatrixView< Real, Device, Index, Organization > >
{
   [[nodiscard]] static std::string
   getDensity()
   {
      return "dense";
   }

   [[nodiscard]] static std::string
   getFormat()
   {
      return "Dense";
   }
};

template< typename Real, typename Device, typename Index, ElementsOrganization Organization, typename RealAllocator >
struct MatrixInfo< DenseMatrix< Real, Device, Index, Organization, RealAllocator > >
: public MatrixInfo< typename DenseMatrix< Real, Device, Index, Organization, RealAllocator >::ViewType >
{};

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename Device_, typename Index_ >
          class SegmentsView >
struct MatrixInfo< SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView > >
{
   [[nodiscard]] static std::string
   getDensity()
   {
      return "sparse";
   }

   [[nodiscard]] static std::string
   getFormat()
   {
      std::string prefix;
      if( MatrixType::isSymmetric() ) {
         if( std::is_same< Real, bool >::value )
            prefix = "Symmetric Binary ";
         else
            prefix = "Symmetric ";
      }
      else if( std::is_same< Real, bool >::value )
         prefix = "Binary ";
      return prefix + SegmentsView< Device, Index >::getSegmentsType();
   }
};

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename Device_, typename Index_, typename IndexAllocator_ >
          class Segments,
          typename RealAllocator,
          typename IndexAllocator >
struct MatrixInfo< SparseMatrix< Real, Device, Index, MatrixType, Segments, RealAllocator, IndexAllocator > >
: public MatrixInfo<
     typename SparseMatrix< Real, Device, Index, MatrixType, Segments, RealAllocator, IndexAllocator >::ViewType >
{};

template< typename Real, typename Device, typename Index, typename MatrixType >
struct MatrixInfo< Sandbox::SparseSandboxMatrixView< Real, Device, Index, MatrixType > >
{
   [[nodiscard]] static std::string
   getDensity()
   {
      return "sparse";
   }

   [[nodiscard]] static std::string
   getFormat()
   {
      if( MatrixType::isSymmetric() )
         return "Symmetric Sandbox";
      else
         return "Sandbox";
   }
};

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
struct MatrixInfo< Sandbox::SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator > >
: public MatrixInfo<
     typename Sandbox::SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::ViewType >
{};

/// \endcond
}  // namespace TNL::Matrices
