# Orthogonal grids tutorial

[TOC]

## Introduction

Grids are regular orthognal meshes. Similar to unstructured numerical meshes they provide indexing of mesh entites and express their adjacency. The difference, compared to the unstructured meshes, is that the adjacency of the mesh entities are not stored explicitly in the memory but the are computed on-the-fly. The interface of grids is as simillar as possible to the unstructured meshes but there are some differences. The main difference is that the mesh entities are given by their coordinates and orientation. The type and orientation of the entity is given by its *basis* and *normals*. Basis is a vector having one for axes, along which the entity has non-zero length, and zeros otherwise. Normals is a vector orthogonal to the basis vector, i.e. it has ones where basis vector has zeros and vice versa. The meaning of the normals vector is such that it is like a pack of all vectors of standart basis which are orthogonal to the grid entity. The following tables show all possible grid entities in 1D, 2D and 3D.

Grid entities in 1D are as follows:

| Entities in 1D             | Basis        | Normals     | Unpacked normal vectors |
|---------------------------:|-------------:|:-----------:|:-----------------------:|
| Cells                      | ( 1 )        | ( 0 )      :|: N/a                   :| 
| Vertexes                   | ( 0 )        | ( 1 )      :|: ( 1 )                 :|

Grid entities in 2D are as follows:

| Entities in 2D             | Basis        | Normals    | Unpacked normal vectors |
|---------------------------:|-------------:|:----------:|:-----------------------:|
| Cells                      | ( 1, 1 )     | ( 0, 0 )   | N/A                     |
| Faces along x- axis        | ( 1, 0 )     | ( 0, 1 )   | ( 0, 1 )                |
| Faces along y- axis        | ( 0, 1 )     | ( 1, 0 )   | ( 1, 0 )                |
| Vertexes                   | ( 0, 0 )     | ( 1, 1 )   | ( 1, 0 ), ( 0, 1 )      |

Grid entities in 3D are as follows:

| Entities in 3D             | Basis        | Normals      | Unpacked normal vectors               |
|---------------------------:|-------------:|:------------:|:-------------------------------------:|
| Cells                      | ( 1, 1, 1 )  | ( 0, 0, 0 )  | N/A                                   |
| Faces along x- and y- axes | ( 1, 1, 0 )  | ( 0, 0, 1 )  | ( 0, 0, 1 )                           |
| Faces along x- and z- axes | ( 1, 0, 1 )  | ( 0, 1, 0 )  | ( 0, 1, 0 )                           |
| Faces along y- and z- axes | ( 0, 1, 1 )  | ( 1, 0, 0 )  | ( 1, 0, 0 )                           |
| Edges along x-axis         | ( 1, 0, 0 )  | ( 0, 1, 1 )  | ( 0, 1, 0 ), ( 0, 0, 1 )              |
| Edges along y-axis         | ( 0, 1, 0 )  | ( 1, 0, 1 )  | ( 1, 0, 0 ), ( 0, 0, 1 )              |
| Edges along z-axis         | ( 0, 0, 1 )  | ( 1, 1, 0 )  | ( 1, 0, 0 ), ( 0, 1, 0 )              |
| Vertexes                   | ( 0, 0, 0 )  | ( 1, 1, 1 )  | ( 1, 0, 0 ), ( 0, 1, 0 ), ( 0, 0, 1 ) |

The grid entity stores the vector with packed normals, tha basis vector is always computed on the fly. So whenever it possible, using the the normals vector is preferred for better performance.

** Remark: The entity orientation given by the normals or basis vector should be encoded staticaly in the type of the entity. This would make the implementation of the grid entities more efficient. Such implementation, however, requires suppport of the generic lambda function by the compiler. Since the CUDA compiler `nvcc` is not able to compile a code with the generic lambda functions we stay with current implementation which is not optimal. Therefore, in the future, the implementation of the grid entities may change. ** 

The following figures show coordinates and indexing of the grid entities in 2D for demonstration. Indexing of cells looks as follows:

```
+-------+-------+-------+-------+-------+    +-------+-------+-------+-------+-------+
| (0,4) | (1,4) | (2,4) | (3,4) | (4,4) |    | (20 ) | (21 ) | (22 ) | (23 ) | (24 ) |
+-------+-------+-------+-------+-------+    +-------+-------+-------+-------+-------+
| (0,3) | (1,3) | (2,3) | (3,3) | (4,3) |    | (15 ) | (16 ) | (17 ) | (18 ) | (19 ) |
+-------+-------+-------+-------+-------+    +-------+-------+-------+-------+-------+
| (0,2) | (1,2) | (2,2) | (3,2) | (4,2) |    | (10 ) | (11 ) | (12 ) | (13 ) | (14 ) |
+-------+-------+-------+-------+-------+    +-------+-------+-------+-------+-------+
| (0,1) | (1,1) | (2,1) | (3,1) | (4,1) |    | ( 5 ) | ( 6 ) | ( 7 ) | ( 8 ) | ( 9 ) |
+-------+-------+-------+-------+-------+    +-------+-------+-------+-------+-------+
| (0,0) | (1,0) | (2,0) | (3,0) | (4,0) |    | ( 0 ) | ( 1 ) | ( 2 ) | ( 3 ) | ( 4 ) |
+-------+-------+-------+-------+-------+    +-------+-------+-------+-------+-------+  
```

Indexing of faces looks as:

```
  +-(0,6)-+-(1,6)-+-(2,6)-+-(3,6)-+-(4,6)-+       +-( 30)-+-( 31)-+-( 32)-+-( 33)-+-( 34)-+  
  |       |       |       |       |       |       |       |       |       |       |       |
(0,5)   (1,5)   (2,5)   (3,5)   (4,5)   (5,5)   ( 65)   ( 66)   ( 67)   ( 68)   ( 69)   ( 70)
  |       |       |       |       |       |       |       |       |       |       |       |
  +-(0,5)-+-(1,5)-+-(2,5)-+-(3,5)-+-(4,5)-+       +-( 25)-+-( 26)-+-( 27)-+-( 28)-+-( 29)-+
  |       |       |       |       |       |       |       |       |       |       |       |
(0,4)   (1,4)   (2,4)   (3,4)   (4,4)   (5,4)   ( 59)   ( 60)   ( 61)   ( 62)   ( 63)   ( 64)
  |       |       |       |       |       |       |       |       |       |       |       |
  +-(0,4)-+-(1,4)-+-(2,4)-+-(3,4)-+-(4,4)-+       +-( 20)-+-( 21)-+-( 22)-+-( 23)-+-( 24)-+
  |       |       |       |       |       |       |       |       |       |       |       |
(0,3)   (1,3)   (2,3)   (3,3)   (4,3)   (5,3)   ( 53)   ( 54)   ( 55)   ( 56)   ( 57)   ( 58)
  |       |       |       |       |       |       |       |       |       |       |       |
  +-(0,3)-+-(1,3)-+-(2,3)-+-(3,3)-+-(4,3)-+       +-( 15)-+-( 16)-+-( 17)-+-( 18)-+-( 19)-+
  |       |       |       |       |       |       |       |       |       |       |       |
(0,2)   (1,2)   (2,2)   (3,2)   (4,2)   (5,2)   ( 47)   ( 48)   ( 49)   ( 50)   ( 51)   ( 52)
  |       |       |       |       |       |       |       |       |       |       |       |
  +-(0,2)-+-(1,2)-+-(2,2)-+-(3,2)-+-(4,2)-+       +-( 10)-+-( 11)-+-( 12)-+-( 13)-+-( 14)-+
  |       |       |       |       |       |       |       |       |       |       |       |
(0,1)   (1,1)   (2,1)   (3,1)   (4,1)   (5,1)   ( 41)   ( 42)   ( 43)   ( 44)   ( 45)   ( 46)
  |       |       |       |       |       |       |       |       |       |       |       |
  +-(0,1)-+-(1,1)-+-(2,1)-+-(3,1)-+-(4,1)-+       +-( 5 )-+-( 6 )-+-( 7 )-+-( 8 )-+-( 9 )-+
  |       |       |       |       |       |       |       |       |       |       |       |
(0,0)   (1,0)   (2,0)   (3,0)   (4,0)   (5,0)   ( 35)   ( 36)   ( 37)   ( 38)   ( 39)   ( 40)
  |       |       |       |       |       |       |       |       |       |       |       |
  +-(0,0)-+-(1,0)-+-(2,0)-+-(3,0)-+-(4,0)-+       +-( 0 )-+-( 1 )-+-( 2 )-+-( 3 )-+-( 4 )-+  
```

And indexing of vertexes looks as follows:

```
(0,5)--(1,5)--(2,5)--(3,5)--(4,5)--(5,5)      ( 30)--( 31)--( 32)--( 33)--( 34)--( 35)
  |      |      |      |      |      |          |      |      |      |      |      |
(0,4)--(1,4)--(2,4)--(3,4)--(4,4)--(5,4)      ( 24)--( 25)--( 26)--( 27)--( 28)--( 29)
  |      |      |      |      |      |          |      |      |      |      |      |
(0,3)--(1,3)--(2,3)--(3,3)--(4,3)--(5,3)      ( 18)--( 19)--( 20)--( 21)--( 22)--( 23)
  |      |      |      |      |      |          |      |      |      |      |      |
(0,2)--(1,2)--(2,2)--(3,2)--(4,2)--(5,2)      ( 12)--( 13)--( 14)--( 15)--( 16)--( 17)
  |      |      |      |      |      |          |      |      |      |      |      |
(0,1)--(1,1)--(2,1)--(3,1)--(4,1)--(5,1)      (  6)--(  7)--(  8)--(  9)--( 10)--( 11)
  |      |      |      |      |      |          |      |      |      |      |      |
(0,0)--(1,0)--(2,0)--(3,0)--(4,0)--(5,0)      (  0)--(  1)--(  2)--(  3)--(  4)--(  5) 
```

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



