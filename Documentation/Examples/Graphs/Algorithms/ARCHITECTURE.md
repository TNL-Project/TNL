# Graph algorithm examples architecture

This document describes the conventions for the example programs in
`Documentation/Examples/Graphs/Algorithms/`.  Each algorithm (BFS, SSSP, CC,
SCC, Trees, MIS, Coloring) ships a single example source file (`GraphExample_<X>.cpp`).
Doxygen `\snippet` directives in the public headers include named blocks from
these files into the API documentation.

### `.cu` and `.hip` are symbolic links

The `.cpp` file is the **only** real source file.  The `.cu` and `.hip`
variants are symbolic links to it:

```bash
ln -s GraphExample_BFS.cpp GraphExample_BFS.cu
ln -s GraphExample_BFS.cpp GraphExample_BFS.hip
```

This avoids code duplication: the same source is compiled by the host
compiler (via `.cpp`), by `nvcc` (via `.cu`), and by `hipcc` (via `.hip`)
without maintaining three copies.  When editing an example, **always edit
the `.cpp` file** — never the symlinks.

## Conventions

### One graph per algorithm

All overload variants of a single algorithm can share the same input graph.  The
graph must be rich enough to produce distinct outputs for each variant
(isolated vertices, branches, optional edges that the edge predicate can filter
out).  This lets the user compare outputs across variants without juggling
multiple graph definitions.

### Self-contained snippets

Every `//! [snippet name]` block must be **self-contained**: it must include
all lambdas, vertex-index arrays, and output vectors required by the call it
documents.  A reader looking at a single snippet in the API documentation must
understand the call without scrolling back to earlier snippets.  In particular:

- **Do not** reference a lambda defined in an earlier snippet — pass it inline
  directly as an argument to the algorithm call.
- **Do not** reuse a `VectorType` or `activeVertices` array from an earlier
  snippet — declare a fresh one inside the current snippet (e.g.
  `VectorType{ 0, 1, 2, 3 }` can be passed directly in the argument list).
- Use **unique names** for output vectors only (e.g. `distancesEdge`,
  `distancesInduced`), since they must be declared before the call.  Lambdas
  and vertex-index arrays should be passed inline without a name.
- Add a brief comment before each lambda identifying its role:
  `// edge predicate`, `// vertex predicate`, `// edge weight callable`,
  `// visitor`.

The only exception is the `[graph type definition]` block, which defines the
`GraphType`, `IndexType`, and `VectorType` aliases.  It is shared across all
snippets in a file and is referenced once from the algorithm overview page.

### Single graph type definition

The `[graph type definition]` block appears once near the top of each example
file.  Snippets that need a `VectorType` simply use the alias — they do not
redefine it.

### Output after every call

Every snippet ends with a `std::cout` line that prints the result, so that
the generated `.out` file demonstrates the expected output.

## Adding a new overload

When adding a new overload variant to an existing algorithm:

1. Add a new `//! [snippet name]` block in the **`.cpp`** file, following the
   self-contained rule above.
2. Reference the new snippet from the corresponding function's `\par Example`
   block in the public header (`.h` file).
3. If the new variant requires a new graph shape to be meaningful, consider
   whether the existing graph still works for all other variants — prefer
   keeping one graph if possible.
4. Rebuild the example and regenerate the `.out` file.

When adding a **new algorithm** example:

1. Create `GraphExample_<Name>.cpp`.
2. Create symlinks: `ln -s GraphExample_<Name>.cpp GraphExample_<Name>.cu`
   and `ln -s GraphExample_<Name>.cpp GraphExample_<Name>.hip`.
3. Register the target in `CMakeLists.txt` (add to `COMMON_EXAMPLES`).
