// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

/**
 * \brief Namespace for image processing.
 */
namespace TNL::Images {

template< typename Index = int >
class Image
{
public:
   using IndexType = Index;

   Image() : width( 0 ), height( 0 ) {}

   [[nodiscard]] IndexType
   getWidth() const
   {
      return this->width;
   }

   [[nodiscard]] IndexType
   getHeight() const
   {
      return this->height;
   }

protected:
   IndexType width, height;
};

}  // namespace TNL::Images
