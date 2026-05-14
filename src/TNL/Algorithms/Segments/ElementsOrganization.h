// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <string>

#include <magic_enum/magic_enum.hpp>

#include <TNL/Devices/Host.h>

namespace TNL {
namespace Algorithms::Segments {

enum ElementsOrganization : std::uint8_t
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
      if( std::is_same_v< Device, Devices::Host > )
         return RowMajorOrder;
      else
         return ColumnMajorOrder;
   }
};

}  // namespace Algorithms::Segments

inline std::string
getSerializationType( Algorithms::Segments::ElementsOrganization Organization )
{
   return std::string( magic_enum::enum_name( Organization ) );
}

}  // namespace TNL
