// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Tomáš Oberhuber, Yury Hayeu

#pragma once


template< int Size, typename Real = double >
struct UnrolledStaticVector : public UnrolledStaticVector< Size - 1, Real >
{
   UnrolledStaticVector( const Real& v ) : UnrolledStaticVector< Size - 1, Real >( v ), data( v ){};

   Real getL1Norm(){ return TNL::abs( data ) + UnrolledStaticVector< Size - 1, Real >::getL1Norm(); }

   protected:

   Real data;
};

template< typename Real >
struct UnrolledStaticVector< 0, Real >
{
   UnrolledStaticVector( const Real& v ) : data( v ){};

   Real getL1Norm(){ return TNL::abs( data ); }

   protected:

   Real data;
};
