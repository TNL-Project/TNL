// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>

#include <TNL/String.h>
#include <TNL/File.h>

/**
 * \brief The main TNL namespace.
 */
namespace TNL {

/**
 * \brief Extracts object type from a binary file.
 *
 * Throws \ref Exceptions::FileDeserializationError if the object type cannot be detected.
 *
 * \param file File where the object is stored.
 * \return String with the object type.
 */
[[nodiscard]] String
getObjectType( File& file );

/**
 * \brief Does the same as \ref getObjectType but with a `fileName` parameter instead of file.
 *
 * Throws \ref Exceptions::FileDeserializationError if the object type cannot be detected.
 *
 * \param fileName Name of a file where the object is stored.
 * \return String with the object type.
 */
[[nodiscard]] String
getObjectType( const String& fileName );

/**
 * \brief Parses the object type.
 *
 * \param objectType String with the object type to be parsed.
 * \return Vector of strings where the first one is the object type and the next
 *         strings are the template parameters.
 *
 * \par Example
 * \include ParseObjectTypeExample.cpp
 * \par Output
 * \include ParseObjectTypeExample.out
 */
[[nodiscard]] std::vector< String >
parseObjectType( const String& objectType );

/**
 * \brief Saves object type into a binary file.
 *
 * Throws \ref Exceptions::FileDeserializationError if the object type cannot be detected.
 *
 * \param file File where the object will be saved.
 * \param type Object type to be saved.
 */
void
saveObjectType( File& file, const String& type );

}  // namespace TNL

#include <TNL/Object.hpp>
