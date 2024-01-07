// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Config/ConfigEntry.h>

namespace TNL::Config {

template< typename EntryType >
class ConfigEntryList : public ConfigEntry< EntryType, std::vector< EntryType > >
{
public:
   // inherit constructors
   using ConfigEntry< EntryType, std::vector< EntryType > >::ConfigEntry;
};

}  // namespace TNL::Config
