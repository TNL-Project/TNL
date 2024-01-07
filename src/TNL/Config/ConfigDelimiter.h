// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Config/ConfigEntryBase.h>

namespace TNL::Config {

class ConfigDelimiter : public ConfigEntryBase
{
public:
   ConfigDelimiter( const std::string& delimiter ) : ConfigEntryBase( "", delimiter, false ) {}

   [[nodiscard]] bool
   isDelimiter() const override
   {
      return true;
   }

   [[nodiscard]] std::string
   getUIEntryType() const override
   {
      return "";
   }
};

}  // namespace TNL::Config
