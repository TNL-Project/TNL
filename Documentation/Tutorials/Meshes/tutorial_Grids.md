# Orthogonal grids tutorial

[TOC]

## Introduction

Grids are regular orthognal meshes. Similar to unstructured numerical meshes they provide indexing of mesh entites and express their adjacency. The difference, compared to the unstructured meshes, is that the adjacency of the mesh entities are not stored explicitly in the memory but the are computed on-the-fly. The interface of grids is as simillar as possible to the unstructured meshes but there are some differences. The main difference is that the mesh entities are given by their coordinates and orientation. The type and orientation of the entity is given by its *basis*. It is a vector having one for axes, along which the entity has non-zero length, and zeros otherwise. The following tables show all possible grid entities in 1D, 2D and 3D.

Grid entities in 1D are as follows:

| Entities in 1D             | Basis        |
|---------------------------:|-------------:|
| Cells                      | ( 1 )        |
| Vertexes                   | ( 0 )        |

Grid entities in 2D are as follows:

| Entities in 2D             | Basis        |
|---------------------------:|-------------:|
| Cells                      | ( 1, 1 )     |
| Faces along x- axis        | ( 1, 0 )     |
| Faces along y- axis        | ( 0, 1 )     |
| Vertexes                   | ( 0, 0 )     |

Grid entities in 3D are as follows:

| Entities in 3D             | Basis        |
|---------------------------:|-------------:|
| Cells                      | ( 1, 1, 1 )  |
| Faces along x- and y- axes | ( 1, 1, 0 )  |
| Faces along x- and z- axes | ( 1, 0, 1 )  |
| Faces along y- and z- axes | ( 0, 1, 1 )  |
| Edges along x-axis         | ( 1, 0, 0 )  |
| Edges along y-axis         | ( 0, 1, 0 )  |
| Edges along z-axis         | ( 0, 0, 1 )  |
| Vertexes                   | ( 0, 0, 0 )  |

Grid may have arbitrary dimension i.e. even higher than 3D. It is represented by the templated class \ref TNL::Meshes::Grid which has the wollowing template parameters:

-  `Dimension` is dimension of the grid. This can be any interger value greater than zero.
-  `Real` is a precision of the arithmetics used by the grid. It is `double` by default.
-  `Device` is the device where the grid shall be allocated. Currently it can be either \ref TNL::Devices::Host for CPU or \ref TNL::Devices::Cuda for CUDA supporting GPUs. It is \ref TNL::Devices::Host by default.
-  `Index` is a type being used for indexing. It is `int` by default.

## Grid creation

The grid is defined by its dimension, domain covered by the grid and its resolution. The following example shows how to create a grid:

\includelineno GridExample_Constructor-1.h

Here we create set of grids with different dimension. For each grid we set different resolution along each axis (using the constructor of the grid) and different length along each axis (by calling method \ref TNL::Meshes::Grid::setDomain). 

The result looks as follows:

\include GridExample_Constructor-1.out

The following example shows creation of a grid independently on the grid dimension. The domain covered by the grid is \f$ [0,1]^d\f$ where \f$ d \f$ is the grid dimension. The resolution os the same along each axis. Tho code looks as follows:

\includelineno GridExample_Constructor-2.h

The result looks as follows:

\include GridExample_Constructor-2.out

## Traversing the grid

The grid does not store any data it only provides only indexing of the grid entities. The indexes then serve for accessing data stored in an array or vector. The grid entities may be traversed in parallel as we show in the following example:

\includelineno GridExample_traverse.h

The result looks as follows:

\include GridExample_traverse.out



