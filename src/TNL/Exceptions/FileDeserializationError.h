// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <stdexcept>

namespace TNL::Exceptions {

class FileDeserializationError : public std::runtime_error
{
public:
   FileDeserializationError( const std::string& fileName, const std::string& details )
   : std::runtime_error( "Failed to deserialize an object from the file '" + fileName + "': " + details )
   {}
};

}  // namespace TNL::Exceptions
