// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Tomas Oberhuber

#pragma once

#include <TNL/Containers/ArrayView.h>
#include <TNL/Containers/Expressions/ExpressionTemplates.h>
#include <TNL/Algorithms/AtomicOperations.h>

namespace TNL::Containers {

template< typename Real = double, typename Device = Devices::Host, typename Index = int >
class AtomicVectorView : public VectorView< Real, Device, Index >
{
   using BaseType = VectorView< Real, Device, Index >;
   using NonConstReal = typename std::remove_const< Real >::type;

public:
   /**
    * \brief Type of elements stored in this vector.
    */
   using RealType = Real;

   /**
    * \brief Device used to run operations on the vector.
    *
    * See \ref TNL::Devices for the available options.
    */
   using DeviceType = Device;

   /**
    * \brief Type being used for the vector elements indexing.
    */
   using IndexType = Index;

   /**
    * \brief Compatible VectorView type.
    */
   using ViewType = AtomicVectorView< Real, Device, Index >;

   /**
    * \brief Compatible constant VectorView type.
    */
   using ConstViewType = AtomicVectorView< std::add_const_t< Real >, Device, Index >;

   /**
    * \brief A template which allows to quickly obtain a
    * \ref TNL::Containers::VectorView "VectorView" type with changed template
    * parameters.
    */
   template< typename _Real, typename _Device = Device, typename _Index = Index >
   using Self = AtomicVectorView< _Real, _Device, _Index >;

   template< typename Real_ >  // template catches both const and non-const qualified Element
   __cuda_callable__
   AtomicVectorView( const ArrayView< Real_, Device, Index >& view ) : BaseType( view ) {}

   __cuda_callable__
   AtomicVectorView( RealType* data, IndexType size ) : BaseType( data, size ) {}

   ViewType getView() { return ViewType( this->getData(), this->getSize() ); }

   ConstViewType getConstView() const { return ConstViewType( this->getData(), this->getSize() ); }

   __cuda_callable__
   Real atomicAdd( IndexType i, RealType value)
   {
      return Algorithms::AtomicOperations< Device >::add( BaseType::data[ i ], value );
   }
};

} // namespace TNL::Containers
