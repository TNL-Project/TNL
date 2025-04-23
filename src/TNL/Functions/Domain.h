// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

namespace TNL::Functions {

enum DomainType : std::uint8_t
{
   NonspaceDomain,
   SpaceDomain,
   MeshDomain,
   MeshInteriorDomain,
   MeshBoundaryDomain
};

template< int Dimension, DomainType DomainType_ = SpaceDomain >
class Domain
{
public:
   using DeviceType = void;

   [[nodiscard]] static constexpr int
   getDomainDimension()
   {
      return Dimension;
   }

   [[nodiscard]] static constexpr DomainType
   getDomainType()
   {
      return DomainType_;
   }
};

}  // namespace TNL::Functions
