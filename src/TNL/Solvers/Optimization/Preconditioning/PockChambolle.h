// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

namespace TNL::Solvers::Optimization::Preconditioning {

template< typename Matrix, typename Vector >
void
pockChambole( Matrix& K,
              Matrix& KT,
              typename Matrix::IndexType m1,
              Vector& c,
              Vector& q,
              Vector& l,
              Vector& u,
              Vector& D1,
              Vector& D2,
              const typename Matrix::RealType& alfa = 1.0 )
{
   using RealType = typename Matrix::RealType;
   using IndexType = typename Matrix::IndexType;

   TNL_ASSERT_EQ( K.getRows(), KT.getColumns(), "" );
   TNL_ASSERT_EQ( K.getColumns(), KT.getRows(), "" );
   TNL_ASSERT_EQ( K.getColumns(), c.getSize(), "" );
   TNL_ASSERT_EQ( K.getRows(), q.getSize(), "" );
   TNL_ASSERT_EQ( K.getColumns(), l.getSize(), "" );
   TNL_ASSERT_EQ( K.getColumns(), u.getSize(), "" );

   D1.setSize( K.getColumns() );
   D2.setSize( K.getRows() );

   auto D1_view = D1.getView();
   auto D2_view = D2.getView();

   K.reduceAllRows(
      [ = ] __cuda_callable__( IndexType rowIdx, IndexType columnIdx, const RealType& value ) -> RealType
      {
         return pow( abs( value ), 2.0 - alfa );
      },
      TNL::Plus{},
      [ = ] __cuda_callable__( IndexType rowIdx, const RealType& value ) mutable
      {
         D2_view[ rowIdx ] = 1.0 / value;
      },
      0.0 );
   KT.reduceAllRows(
      [ = ] __cuda_callable__( IndexType rowIdx, IndexType columnIdx, const RealType& value ) -> RealType
      {
         return pow( abs( value ), alfa );
      },
      TNL::Plus{},
      [ = ] __cuda_callable__( IndexType rowIdx, const RealType& value ) mutable
      {
         D1_view[ rowIdx ] = 1.0 / value;
      },
      0.0 );

   // Apply the preconditioner
   K.forAllElements(
      [ = ] __cuda_callable__( IndexType rowIdx, IndexType columnIdx, IndexType globalIdx, RealType & value ) mutable
      {
         value *= D1_view[ columnIdx ] * D2_view[ rowIdx ];
      } );
   KT.forAllElements(
      [ = ] __cuda_callable__( IndexType rowIdx, IndexType columnIdx, IndexType globalIdx, RealType & value ) mutable
      {
         value *= D1_view[ rowIdx ] * D2_view[ columnIdx ];
      } );

   // Apply the preconditioner to the bounds
   c *= D2;
   q *= D1;
   u /= D2;
   l /= D2;
}

}  // namespace TNL::Solvers::Optimization::Preconditioning
