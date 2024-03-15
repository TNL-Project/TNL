# Orthogonal grids  {#ug_Grids}

[TOC]

## Introduction

Grids are regular orthogonal meshes. Similar to unstructured numerical meshes
they provide indexing of mesh entities and express their adjacency. The
difference, compared to the unstructured meshes, is that the adjacency of the
mesh entities are not stored explicitly in the memory but the are computed
on-the-fly. The interface of grids is as similar as possible to the
unstructured meshes but there are some differences. The main difference is that
the mesh entities are given by their coordinates and orientation. The type and
orientation of the entity is given by its *basis* and *normals*. Basis is a
vector having one for axes, along which the entity has non-zero length, and
zeros otherwise. Normals is a vector orthogonal to the basis vector, i.e. it
has ones where basis vector has zeros and vice versa. The meaning of the
normals vector is such that it is like a pack of all vectors of standard basis
which are orthogonal to the grid entity. The following tables show all possible
grid entities in 1D, 2D and 3D.

### Bases and normals

Grid entities in 1D are as follows:

| Entities in 1D             | Basis       | Normals     | Unpacked normal vectors |
|---------------------------:|------------:|:-----------:|:-----------------------:|
| Cells                      | ( 1 )       | ( 0 )       |  N/a                    |
| Vertexes                   | ( 0 )       | ( 1 )       |  ( 1 )                  |

Grid entities in 2D are as follows:

| Entities in 2D             | Basis       | Normals     | Unpacked normal vectors |
|---------------------------:|------------:|:-----------:|:-----------------------:|
| Cells                      | ( 1, 1 )    | ( 0, 0 )    | N/A                     |
| Faces along x- axis        | ( 1, 0 )    | ( 0, 1 )    | ( 0, 1 )                |
| Faces along y- axis        | ( 0, 1 )    | ( 1, 0 )    | ( 1, 0 )                |
| Vertexes                   | ( 0, 0 )    | ( 1, 1 )    | ( 1, 0 ), ( 0, 1 )      |

Grid entities in 3D are as follows:

| Entities in 3D             | Basis       | Normals     | Unpacked normal vectors               |
|---------------------------:|------------:|:-----------:|:-------------------------------------:|
| Cells                      | ( 1, 1, 1 ) | ( 0, 0, 0 ) | N/A                                   |
| Faces along x- and y- axes | ( 1, 1, 0 ) | ( 0, 0, 1 ) | ( 0, 0, 1 )                           |
| Faces along x- and z- axes | ( 1, 0, 1 ) | ( 0, 1, 0 ) | ( 0, 1, 0 )                           |
| Faces along y- and z- axes | ( 0, 1, 1 ) | ( 1, 0, 0 ) | ( 1, 0, 0 )                           |
| Edges along x-axis         | ( 1, 0, 0 ) | ( 0, 1, 1 ) | ( 0, 1, 0 ), ( 0, 0, 1 )              |
| Edges along y-axis         | ( 0, 1, 0 ) | ( 1, 0, 1 ) | ( 1, 0, 0 ), ( 0, 0, 1 )              |
| Edges along z-axis         | ( 0, 0, 1 ) | ( 1, 1, 0 ) | ( 1, 0, 0 ), ( 0, 1, 0 )              |
| Vertexes                   | ( 0, 0, 0 ) | ( 1, 1, 1 ) | ( 1, 0, 0 ), ( 0, 1, 0 ), ( 0, 0, 1 ) |

The grid entity stores the vector with packed normals, the basis vector is
always computed on the fly. So whenever it possible, using the normals
vector is preferred for better performance.

**Remark:** The entity orientation given by the normals or basis vector should
be encoded statically in the type of the entity. This would make the
implementation of the grid entities more efficient. Such implementation,
however, requires support of the generic lambda function by the compiler.
Since the CUDA compiler `nvcc` is not able to compile a code with the generic
lambda functions we stay with current implementation which is not optimal.
Therefore, in the future, the implementation of the grid entities may change.

The following figures show coordinates and indexing of the grid entities in 2D
for demonstration. Indexing of cells looks as follows:

```text
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

```text
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

```text
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

Grid may have arbitrary dimension i.e. even higher than 3D. It is represented
by the templated class \ref TNL::Meshes::Grid which has the following template
parameters:

- `Dimension` is dimension of the grid. This can be any integer value greater
  than zero.
- `Real` is a precision of the arithmetics used by the grid. It is `double` by
  default.
- `Device` is the device where the grid shall be allocated. Currently it can be
  either \ref TNL::Devices::Host for CPU or \ref TNL::Devices::Cuda for CUDA
  supporting GPUs. It is \ref TNL::Devices::Host by default.
- `Index` is a type being used for indexing. It is `int` by default.

## Grid creation

The grid is defined by its dimension, domain covered by the grid and its
resolution. The following example shows how to create a grid:

\includelineno GridExample_constructor-1.h

Here we create set of grids with different dimension. For each grid we set
different resolution along each axis (using the constructor of the grid) and
different length along each axis (by calling method \ref TNL::Meshes::Grid::setDomain).

The result looks as follows:

\include GridExample_constructor-1.out

The following example shows creation of a grid independently on the grid
dimension. The domain covered by the grid is \f$ [0,1]^d\f$ where \f$ d \f$ is
the grid dimension. The resolution is the same along each axis. The code looks
as follows:

\includelineno GridExample_constructor-2.h

The result looks as follows:

\include GridExample_constructor-2.out

## Traversing the grid

The grid does not store any data, it only provides indexing of the grid
entities. The indexes then serve for accessing data stored in an array or
vector. The grid entities may be traversed in parallel as we show in the
following example:

\includelineno GridExample_traverse.h

In this example we start with writing an index of each cell into the cell.
Next, we fill each face with average number computed from the neighbor cells
and at the end, we do the same with vertices. We also write values stored in
particular grid entities to the console.

The whole example consists of the following steps:

1. Setting dimension and resolution of the grid, defining necessary types that
   will be used later (grid type `GridType`, type for coordinates of the grid
   entities `CoordinatesType`, type for the real-world coordinates `PointType`,
   and type of container for storing the values of particular grid entities
   `VectorType`), and definition of types for grid entities (`GridCell`,
   `GridFace`, and `GridVertex`):

   \snippet GridExample_traverse.h setup

2. We create an instance of the grid:

   \snippet GridExample_traverse.h create grid

   The resolution of the grid, which equals 5x5, is set by the grid constructor.
   The domain covered by the grid is given by points `origin` and `proportions`
   which are passed to the method \ref TNL::Meshes::Grid::setDomain.

3. Next we allocate vectors for storing of values in particular grid entities:

   \snippet GridExample_traverse.h allocate vectors

   The number of grid entities are given by templated method
   \ref TNL::Meshes::Grid::getEntitiesCount. The template parameter defines
   dimension of the grid entities the number of which we are asking for.

4. In the next step, we prepare vector views (`cells_view`, `faces_view` and
   `vertexes_view`) which we will need later in lambda functions:

   \snippet GridExample_traverse.h prepare vector views

5. We iterate over all grid cells using templated method
   \ref TNL::Meshes::Grid::forAllEntities. The template parameter again says
   the dimension of the grid entities we want to iterate over. The method takes
   one parameter which is a lambda function that is supposed to be evaluated
   for each cell. The lambda function receives one parameter, type of which is
   \ref TNL::Meshes::GridEntity. This grid entity represents particular cells
   over which we iterate. The index of the cell is obtained by method
   \ref TNL::Meshes::GridEntity::getIndex.

   \snippet GridExample_traverse.h initialize cells

6. Next we print the values of all cells. This must be done sequentially on the
   CPU and therefore we do not use parallel for (\ref TNL::Meshes::Grid::forAllEntities)
   anymore. Now we iterate over all cells sequentially row by row:

   \snippet GridExample_traverse.h print cells

   In the inner loop we create a grid cell with given coordinates and we ask for
   its global index which we use for access to the vector with cell values (`cell`).
   The result of this is a matrix of cell values printed to the standard output.

7. We iterate over all faces of the grid. Again we use the method
   \ref TNL::Meshes::Grid::forAllEntities but now the template parameter
   specifying the dimension of the entities is `Dimension-1`. The lambda
   function being performed for each face now gets one parameter which is a
   grid entity (\ref TNL::Meshes::GridEntity) with dimension equal to one:

   \snippet GridExample_traverse.h initialize faces

   First of all we fetch the normal vector of the face by calling the method
   \ref TNL::Meshes::GridEntity::getNormals which in case of faces is really a
   normal to the face (there is no more than one orthogonal normal to the face
   in the space having the same dimension as the grid). For horizontal faces,
   the normal is \f$n_h=(0,1)\f$, and for vertical faces, the normal is
   \f$n_v=(1,0)\f$. Note that if the resolution of the grid is \f$(N_x,N_y)\f$
   then the number of horizontal faces along \f$x\f$ and \f$y\f$ axes is
   \f$(N_x,N_y+1)=(N_x,N_y)+n_h\f$ and the number of vertical faces along
   \f$x\f$ and \f$y\f$ axes is \f$(N_x+1,N_y)=(N_x,N_y)+n_v\f$.

   To compute the average number of values of the neighbor cells we need to get
   value from the cell in the direction opposite to the face normal. We first
   check if there is such a cell (first `if` statement in the previous snippet)
   and if it is the case, we fetch its global index, add its value to the `sum`
   and then we increment the variable `count` which is the number of values
   that contributed to the average. Next we have to get the value from the cell
   that has the same coordinates as the face itself. We first check if there is
   such a cell (second `if` statement in the previous snippet) - note that the
   number of faces can be higher than the number of cells along the \f$x\f$
   axis for vertical faces and along \f$y\f$ axis for horizontal faces. If
   there is such neighbor cell, we fetch its index, add its value to the `sum`
   and increment the variable `count`. Finally, we compute the average value of
   the neighbor cells and store in the vector with values of faces at the
   position given by the global index of the current face.

8. Next we print the values of the faces on the console. Line by line, we print
   first horizontal faces with row index equal to five, followed by vertical
   faces with row index equal to five, next horizontal faces with row index
   equal to four, and so on. As before, this must be done sequentially on the
   CPU so we do not use parallel for (\ref TNL::Meshes::Grid::forAllEntities):

   \snippet GridExample_traverse.h print faces

   So we iterate over all rows of the grid and for each row we first print the
   horizontal faces and then the vertical faces. Note that in both cases we
   create an instance of a grid face where the last parameter of the constructor
   determines the orientation of the face - `{0, 1}` is the normal of horizontal
   faces and `{1, 0}` is the normal of vertical faces.

9. Finally, we iterate over all vertexes and compute average value of all
   neighboring cells. The vertexes have no orientation so we do not need to
   care about the normals. We only check the number of neighbor cells based on
   the vertex coordinates. To iterate over all vertexes in parallel, we use the
   method \ref TNL::Meshes::Grid::forAllEntities with entity dimension set to
   zero:

   \snippet GridExample_traverse.h initialize vertexes

   The lambda function we perform on each vertex, has the parameter `vertex`
   which represents the vertex that we currently operate on. We just fetch the
   coordinates of the vertex and then we check what are the neighbor cells
   (four `if` statements in the previous snippet). For each such a cell we add
   its value to the `sum`. Finally, we compute the average value and store it
   in the vector.

10. At the end we print values of all vertexes the same way as we did it with
    cells:

   \snippet GridExample_traverse.h print vertexes

The result looks as follows:

\include GridExample_traverse.out

## Writers

Writers help to export data linked with a grid to some of standard formats like
[VTK, VTI](https://kitware.github.io/vtk-examples/site/VTKFileFormats/) or
[Gnuplot](http://www.gnuplot.info/). The following example shows the use of the
grid writers:

\includelineno GridExample_writer.h

It is a modification of the previous example. As before, we first set values
linked with the grid cells and vertexes and then we write the into output files.
The values linked with cells are exported to the VTI format, to the VTK format
and to the Gnuplot format. The writers have the same interface. They have
constructors which require output stream as a parameter
(\ref TNL::Meshes::Writers::VTIWriter::VTIWriter,
\ref TNL::Meshes::Writers::VTKWriter::VTKWriter). Next we call a method
`writeEntities` (\ref TNL::Meshes::Writers::VTIWriter::writeEntities,
\ref TNL::Meshes::Writers::VTKWriter::writeEntities) which exports the grid
entities to the file. In the case of the Gnuplot format, this method just
writes simple header to the file and does not need to be called (the method is
mainly for the compatibility with other writers). Finally, we may export data
linked with the grid cells using a method `writeCellData`
(\ref TNL::Meshes::Writers::VTIWriter::writeCellData,
\ref TNL::Meshes::Writers::VTKWriter::writeCellData). In the same way, we
export data linked with vertexes.
