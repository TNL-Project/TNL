# Orthogonal grids  {#ug_Grids}

[TOC]

## Introduction

Grids are regular orthognal meshes. Similar to unstructured numerical meshes they provide indexing of mesh entites and express their adjacency. The difference, compared to the unstructured meshes, is that the adjacency of the mesh entities are not stored explicitly in the memory but the are computed on-the-fly. The interface of grids is as simillar as possible to the unstructured meshes but there are some differences. The main difference is that the mesh entities are given by their coordinates and orientation. The type and orientation of the entity is given by its *basis* and *normals*. Basis is a vector having one for axes, along which the entity has non-zero length, and zeros otherwise. Normals is a vector orthogonal to the basis vector, i.e. it has ones where basis vector has zeros and vice versa. The meaning of the normals vector is such that it is like a pack of all vectors of standart basis which are orthogonal to the grid entity. The following tables show all possible grid entities in 1D, 2D and 3D.

### TODO: Bases and normals

Grid entities in 1D are as follows:

| Entities in 1D             | Basis        | Normals     | Unpacked normal vectors |
|---------------------------:|-------------:|:-----------:|:-----------------------:|
| Cells                      | ( 1 )        | ( 0 )       |  N/a                    |
| Vertexes                   | ( 0 )        | ( 1 )       |  ( 1 )                  |

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

**Remark: The entity orientation given by the normals or basis vector should be encoded staticaly in the type of the entity. This would make the implementation of the grid entities more efficient. Such implementation, however, requires suppport of the generic lambda function by the compiler. Since the CUDA compiler `nvcc` is not able to compile a code with the generic lambda functions we stay with current implementation which is not optimal. Therefore, in the future, the implementation of the grid entities may change.**

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

\includelineno GridExample_constructor-1.h

Here we create set of grids with different dimension. For each grid we set different resolution along each axis (using the constructor of the grid) and different length along each axis (by calling method \ref TNL::Meshes::Grid::setDomain).

The result looks as follows:

\include GridExample_constructor-1.out

The following example shows creation of a grid independently on the grid dimension. The domain covered by the grid is \f$ [0,1]^d\f$ where \f$ d \f$ is the grid dimension. The resolution os the same along each axis. Tho code looks as follows:

\includelineno GridExample_constructor-2.h

The result looks as follows:

\include GridExample_constructor-2.out

## Traversing the grid

The grid does not store any data it only provides only indexing of the grid entities. The indexes then serve for accessing data stored in an array or vector. The grid entities may be traversed in parallel as we show in the following example:

\includelineno GridExample_traverse.h

In this example we start with writting an index of each cell into the cell. Next, we fill each face with average number computed from the neighbour cells and at the end, we do the same with vertices. We also write values stored in particular grid entities to the console.

The whoel example consits of the following steps:

1. Setting dimension and resoltion of the grid (lines 12 and 13).
2. We define necessary types which we will need later. It is the grid type ( `GridType` on line 18), type for coordinates of the grid entities ( `CoordinatesType` line 19), type for the real-world coordinates (`PointType` on lines 20) and type of container for storing the values of particular grid entities (`VectorType` on line 21).
3. We defiene types of the following grid entities - cell (`GridCell`, line 26), face (`GridFace`, line 27) and vertex (`GridVertex`, line 28).
4. We create an instance of the grid (lines 33-35). The resolution of the grid, which equals 5x5, is set by the grid constructor. The domain covered by the grid is given by points `origin` and `proportions` (defined on the line 34) which are passed to the method \ref TNL::Meshes::Grid::setDomain (line 35).
5. Next we allocate vectors for storing of values in particular grid entities (lines 40-42). The number of grid entities are given by templated method \ref TNL::Meshes::Grid::getEntitiesCount. The template parameter defines dimension of the grid entities the number of which we are asking for.
6. In the next step, we prepare vector views (`cells_view`, `faces_view` and `vertexes_view`, lines 48-50) which we will need later in lambda functions.
7. On the lines 55-57, we iterate over all grid cells using templated method \ref TNL::Meshes::Grid::forAllEntities. The template parameter again says dimension of the grid entities we want to iterate over. The method takes one parameter which is a lambda function that is supposed to be evaluated for each cell. The lamda function receives one parameter, type of which is \ref TNL::Meshes::GridEntity. This grid entity represents particular cells over which we iterate. The index of the cell is obtained by method \ref TNL::Meshes::GridEntity::getIndex.
8. Next we print the values of all cells. This must be done sequentially on the CPU and tehrefore we do not use parallal for (\ref TNL::Meshes::Grid::forAllEntities) anymore. Now we iterate over all cells sequentialy row by row (for loops on lines 63 and 64). We create grid cell with given coordinates (line 65) and we ask for its global index (line 66) which we use for acces to the vector with cell values (`cell`). The result of this is a matrix of cell values printed on a console.
9. On the lines (76-91), we iterate over all faces of the grid. Again we use the method \ref TNL::Meshes::Grid::forAllEntities but now the template parameter telling the dimension of the entites is dimensions of the grid minus one (`Dimension-1`). The lambda function being performed for each face now gets one parameter which is a grid entity (\ref TNL::Meshes::GridEntity) with dimensions equal to one. First of all we fetch the normal vector of the face (line 77) by calling a method \ref TNL::Meshes::GridEntity::getNormals which in case of faces is realy a normal to the face (there is no more than one orthogonal normal to the face in the space having the same dimension as the grid). For horizontal faces, the normal is \f$n_h=(0,1)\f$, and for vertical faces, the normal is \f$n_v=(1,0)\f$. Note that if the resolution of the grid is \f$(N_x,N_y)\f$ then the number of horizontal faces along \f$x\f$ and \f$y\f$ axes is \f$(N_x,N_y+1)=(N_x,N_y)+n_h\f$ and the number of vertical faces along \f$x\f$ and \f$y\f$ axes is \f$(N_x+1,N_y)=(N_x,N_y)+n_v\f$. To compute the average number of values of the neighbour cells we need to get value from the cell in the direction opposite to the face normal. We first check if there is such a cell (if statement on the line 80) and if it is the case we fetch its global index (line 81), add its value to the `sum` (line 82) and then we increase variable `count` which is the number off values that contrtibuted to the average (line 83). Next we have to get value from the cell having the same coordinates as the face itself. We first check if there is such a cell (line 85) - note that the number of faces can be higher than the number of cells along the \f$x\f$ axis for vertical faces and along \f$y\f$ axis for horizontal faces. If there is such neighbour cell we fatch its index (line 86), add its value to variable `sum` (line 87) and increase variable `count` (line 88). Finally we compute the average value of the neighbout cells and store in the vector with values of faces at the position given by the global index of the current face (line 90).
10. Next we print the values of the faces on the console. Line by line, we print first horizontal faces with row index equal five, followed by vertical faces with row index equal five, next horizontal faces with row index four and so on. As before, this must be done sequentially on the CPU so we do not use parallel for (TNL::Meshes::Grid::forAllEntities). So we iterate over all rows of the grid (line 97) and for each row we first print the horizontal faces (lines 99-103) and then the vertical ones (lines 106-110). Note that in both cases we create instance of grid face (lines 100 and 107) where the last parameter of the constructor defines the orientatio of the face - `{0,1}` is normal of horizontal faces (line 100) and `{1,0}` is normal of vertical faces.
11. Finally, we iterate over all vertexes and compute average value of all neghbouring cells. The vertexes have no orientation so we do not need to care the normals. We only check the number of neighbour cells based on the vertex coordinates. To iterate over all vertexes in parallel we use the method \ref TNL::Meshes::Grid::forAllEntities with entity dimension set to zero (line 117). The lambda function, we perform on each vertex, provides us parameter `vertex` which represents the vertex that we currently operate on. We just fetch the coordinates of the vertex (line 120) and then we check what are the neighbour cells (if statements on lines 121, 126, 131 and 136). For each such a cell we add its value to variable `sum`. Finaly we compute the average value and store it in the vertex (line 141).
12. At the end we print values of all vertexes the same way as we did it with cells (lines 148-155).

The result looks as follows:

\include GridExample_traverse.out

## Writers

Writers help to export data linked with a grid to some of standart formats like [VTK, VTI](https://kitware.github.io/vtk-examples/site/VTKFileFormats/) or [Gnuplot](http://www.gnuplot.info/). The following example shows the use of the grid writers:

\includelineno GridExample_writer.h

It is a modification of the previous example. As before, we first set values linked with the grid cells and vertexes and then we write the into output files. The values linked with cells are exported to VTI format on the lines 63-69, to VTK format on the lines 74-80 and the Gnuplot format on lines. The writers have the same interface. They have constructors which require output stream as a parameter (\ref TNL::Meshes::Writers::VTIWriter::VTIWriter, \ref TNL::Meshes::Writers::VTKWriter::VTKWriter ). Next we call a method `writeEntities` (\ref TNL::Meshes::Writers::VTIWriter::writeEntities, \ref TNL::Meshes::Writers::VTKWriter::writeEntities) which exports the grid entities to the file. In the case of the Gnuplot format, this method just writes simple header to the file and does not need to be called (the method is mainly for the compatibility with other writers). Finaly we may export data linked with the grid cells using a method `writeCellData` (\ref TNL::Meshes::Writers::VTIWriter::writeCellData, \ref TNL::Meshes::Writers::VTKWriter::writeCellData).

In the same way, we can export data linked with vertexes (line 125-152).
