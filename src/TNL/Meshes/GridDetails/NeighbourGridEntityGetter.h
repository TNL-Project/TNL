/***************************************************************************
                          NeighbourGridEntityGetter.h  -  description
                             -------------------
    begin                : Nov 23, 2015
    copyright            : (C) 2015 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {
namespace Meshes {

template<class, int>
class GridEntity;

template<int GridDimension, int ParentEntityDimension, int NeighbourEntityDimension>
class NeighbourGridEntityGetter;

}  // namespace Meshes
}  // namespace TNL

#include <TNL/Meshes/GridDetails/Implementations/NeighbourGridEntityGetter.hpp>
