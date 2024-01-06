// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Devices/Host.h>

namespace TNL {
namespace Algorithms::Segments {

enum ElementsOrganization
{
   //! \brief Column-major order
   ColumnMajorOrder = 0,
   //! \brief Row-major order
   RowMajorOrder
};

template< typename Device >
struct DefaultElementsOrganization
{
   static constexpr ElementsOrganization
   getOrganization()
   {
      if( std::is_same< Device, Devices::Host >::value )
         return RowMajorOrder;
      else
         return ColumnMajorOrder;
   }
};

}  // namespace Algorithms::Segments

inline std::string
getSerializationType( Algorithms::Segments::ElementsOrganization Organization )
{
   if( Organization == Algorithms::Segments::RowMajorOrder )
      return "RowMajorOrder";
   return "ColumnMajorOrder";
}

}  // namespace TNL
