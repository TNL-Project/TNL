// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <utility>

namespace TNL::Config {

class ConfigEntryBase
{
protected:
   std::string name;

   std::string description;

   bool required;

public:
   ConfigEntryBase( std::string name, std::string description, bool required )
   : name( std::move( name ) ), description( std::move( description ) ), required( required )
   {}

   [[nodiscard]] const std::string&
   getName() const
   {
      return name;
   }

   [[nodiscard]] const std::string&
   getDescription() const
   {
      return description;
   }

   [[nodiscard]] bool
   isRequired() const
   {
      return required;
   }

   [[nodiscard]] virtual bool
   hasDefaultValue() const
   {
      return false;
   }

   [[nodiscard]] virtual std::string
   getUIEntryType() const = 0;

   [[nodiscard]] virtual bool
   isDelimiter() const
   {
      return false;
   }

   [[nodiscard]] virtual std::string
   printDefaultValue() const
   {
      return "";
   }

   [[nodiscard]] virtual bool
   hasEnumValues() const
   {
      return false;
   }

   virtual void
   printEnumValues( std::ostream& str ) const
   {}

   virtual ~ConfigEntryBase() = default;
};

}  // namespace TNL::Config
