// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <ostream>

#include <TNL/Cuda/CudaCallable.h>
#include <TNL/Algorithms/Segments/SegmentElement.h>

namespace TNL::Algorithms::Segments {

/**
 * \brief Iterator for iterating over elements of a segment.
 *
 * The iterator can be used even in GPU kernels.
 *
 * \tparam SegmentView is a type of related segment view.
 */
template< typename SegmentView >
class SegmentViewIterator
{
public:
   /**
    * \brief Type of SegmentView
    */
   using SegmentViewType = SegmentView;

   /**
    * \brief The type used for matrix elements indexing.
    */
   using IndexType = typename SegmentViewType::IndexType;

   /**
    * \brief The type of related matrix element.
    */
   using SegmentElementType = SegmentElement< IndexType >;

   __cuda_callable__
   SegmentViewIterator( const SegmentViewType& segmentView, const IndexType& localIdx );

   /**
    * \brief Comparison of two matrix Segment iterators.
    *
    * \param other is another matrix Segment iterator.
    * \return \e true if both iterators points at the same point of the same matrix, \e false otherwise.
    */
   [[nodiscard]] __cuda_callable__
   bool
   operator==( const SegmentViewIterator& other ) const;

   /**
    * \brief Comparison of two matrix Segment iterators.
    *
    * \param other is another matrix Segment iterator.
    * \return \e false if both iterators points at the same point of the same matrix, \e true otherwise.
    */
   [[nodiscard]] __cuda_callable__
   bool
   operator!=( const SegmentViewIterator& other ) const;

   /**
    * \brief Operator for incrementing the iterator, i.e. moving to the next element.
    *
    * \return reference to this iterator.
    */
   __cuda_callable__
   SegmentViewIterator&
   operator++();

   /**
    * \brief Operator for decrementing the iterator, i.e. moving to the previous element.
    *
    * \return reference to this iterator.
    */
   __cuda_callable__
   SegmentViewIterator&
   operator--();

   /**
    * \brief Operator for derefrencing the iterator.
    *
    * It returns structure \ref SegmentElementType which represent one element of a segment.
    * \return segment element the iterator points to.
    */
   [[nodiscard]] __cuda_callable__
   SegmentElementType
   operator*() const;

protected:
   const SegmentViewType& segmentView;

   IndexType localIdx = 0;
};

}  // namespace TNL::Algorithms::Segments

#include <TNL/Algorithms/Segments/SegmentViewIterator.hpp>
