---
name: doxygen-writer
description: Use when writing or editing Doxygen documentation comments in C++ source files (.h, .hpp, .cpp, .cu, .hip) under src/TNL/. Covers JavaDoc block style (/** */ and //!), backslash commands (\brief \param \tparam \return \ref \par \include \note \warning), markdown emphasis over \e \b \c tags, heading case for \par and \section, semantic line breaks, and the C++ line-length limit. Triggers: doxygen comment, doc block, \brief, \param, \tparam, \ref, \par, \include, \code, /** */, //!, documentation comment, API docs, C++ documentation.
---

# Doxygen Documentation Writer

Guide for writing and editing Doxygen documentation comments
in C++ source files under `src/TNL/`.
The project uses `MARKDOWN_SUPPORT = YES` and `MARKDOWN_STRICT = YES`
(see `Documentation/Doxyfile`),
so Doxygen commands and Markdown syntax coexist inside comment blocks.

## Comment block style

Two Doxygen comment styles are used, each for a distinct purpose:

- **`//!`** is the preferred style
  for simple one-line docstrings
  and short multi-line docstrings.
  Continuation lines also use `//!`:
  ```cpp
  //! \brief Type of the values stored in the array.
  using ValueType = Value;

  //! \brief Returns the *static* size of a specific dimension identified by
  //! a *runtime* parameter `level`.
  [[nodiscard]] static constexpr Index
  getStaticSize( Index level );
  ```

  (adapted from `src/TNL/Containers/ndarray/SizesHolder.h`)

- **`/** ... */`** JavaDoc-style block comments
  are used for complex documentation
  — classes, methods with multiple `\tparam` / `\param` entries,
  detailed descriptions, `\par Example` sections, etc.
  ```cpp
  /**
   * \brief Manages memory, element access, and operations for arrays.
   *
   * \tparam Value  The type of array elements.
   * \tparam Device The device for execution of array operations.
   *
   * \par Example
   * \include Containers/ArrayExample.cpp
   */
  template< typename Value, ... >
  class Array
  ```

Rules:

- Use `//!` only for short docstrings that are **at most three lines total**
  and contain **a single paragraph** (no blank line inside the comment).
  If the docstring has more than three lines or more than one paragraph,
  use the `/** ... */` block style instead.
- Every doc block — `//!` or `/** */` — starts with `\brief`
  on the first content line.
- For `/** */` blocks,
  align the `*` continuation marker
  one space after the `/**` opening
  (or indented to match the surrounding code for nested members),
  and close with ` */` on its own line.
- `/** */` blocks are **only** for Doxygen-processed comments.
  Never use them for regular comments inside function bodies
  or for implementation notes that are not documentation.
  Use `//` single-line comments for those:
  ```cpp
  void resize( IndexType size )
  {
     // Allocating zero bytes is useless.
     // The allocators don't behave the same way.
     ...
  }
  ```
- Do not use `///` triple-slash comments —
  `//!` is the slash-bang equivalent preferred in this project.
- Every public class, method, free function, namespace,
  and type alias must have a Doxygen doc block.
- Private members need documentation only when their behaviour
  is non-obvious.

## Doxygen command style

Always use the backslash form, never the at-sign form.
The at-sign form is not used in this project.

| Use | Write | Not |
|-----|-------|-----|
| Brief description | `\brief` | `@brief` |
| Template parameter | `\tparam Value The type...` | `@tparam ...` |
| Function parameter | `\param index The element index.` | `@param ...` |
| Return value | `\return Reference to the element.` | `@return` / `@returns` |
| Cross-reference | `\ref TNL::Containers::Array` | `@ref ...` |
| Named paragraph | `\par Example` | `@par ...` |
| Include example file | `\include Containers/ArrayExample.cpp` | `@include ...` |
| Include with line numbers | `\includelineno file.cpp` | `@includelineno` |
| Code block | `\code{.cpp} ... \endcode` | `@code ... @endcode` |
| Note | `\note` | `@note` |
| Warning | `\warning` | `@warning` |
| See also | `\see` | `@sa` |
| Inline math | `\f$ x^2 \f$` | `@f$ ... @f$` |
| Display math | `\f[ ... \f]` | `@f[ ... @f]` |

Rules:

- Use `\return`, not `\returns`.
- Use `\ref <symbol>` for unadorned references,
  e.g. `\ref TNL::Devices`, `\ref std::lexicographical_compare`,
  `\ref getSize`.
- Use `\ref <symbol> "display text"`
  when the display text should differ from the symbol name,
  e.g. `\ref TNL::Containers::Array "Array"`.
- `\param` entries go after the detailed description,
  one per parameter, in declaration order.
- `\tparam` entries go after the detailed description,
  one per template parameter, in declaration order.
- `\ref`, `\param`, `\tparam`, `\brief`, `\par`, `\include`,
  `\code`, `\note`, `\warning`, `\see`, `\f[`
  must appear at the start of a line within the comment block
  (after the ` * ` prefix),
  not mid-sentence —
  except `\ref` and `\f$`, which may appear inline in prose.

## Markdown emphasis — never Doxygen tags

Use Markdown for all inline emphasis inside `/** */` and `//!` blocks.
Doxygen processes Markdown when `MARKDOWN_SUPPORT = YES`.
Never use `\b`, `\e`, `\c`, `<b>...</b>`, `<tt>...</tt>`,
or `<em>...</em>`.

Both asterisk and underscore forms are allowed for bold and italic,
but **asterisks are preferred** for both.

| Meaning | Preferred | Also allowed |
|--------|-----------|--------------|
| Bold | `**host**` | `__host__` |
| Italic | `*binding*` | `_binding_` |
| Inline code | `` `ArrayView` `` in backticks | — |

Forbidden: `\b host`, `\e binding`, `\c ArrayView`,
`<b>host</b>`, `<em>binding</em>`, `<tt>ArrayView</tt>`.

Existing code uses `\e` in many places.
When editing a doc block, replace `\e word` with `*word*`
and `\c word` with backticks.
If `word` is an identifier or type name (e.g. `\e Device`, `\e Array`),
use backticks (`` `Device` ``, `` `Array` ``) rather than italic —
identifiers and type names are always inline code.
When adding new content, use Markdown emphasis from the start.

Do not confuse literal symbols with emphasis.
Identifiers that contain underscores —
`__cuda_callable__`, `__CUDACC__`, `_binding` —
are written in backticks as inline code,
never as Markdown bold or italic.
The `__...__` and `_..._` forms only count as emphasis
when they wrap prose words, not when they appear
inside an identifier written in backticks.

## Headings: `\par` and `\section`

### `\par` paragraph titles

`\par` creates a named paragraph heading inside a doc block.
Use **title case** for `\par` titles
(the convention in existing code: `\par Example`, `\par Output`):

```cpp
/**
 * \brief Sorts elements in ascending order.
 *
 * \par Example
 * \include Sorting/SortingExample.cpp
 * \par Output
 * \include Sorting/SortingExample.out
 */
```

### `\section` / `\subsection` / `\subsubsection`

Use these commands only for standalone documentation pages
(see the Markdown documentation skill for those).
In C++ source files, prefer `\par` for named sections
within a class or function doc block.

When `\section` or `\subsection` is used,
apply the same case rule as Markdown documentation:
title case for top-level (`\section`),
sentence case for lower levels (`\subsection`, `\subsubsection`).

## Code examples

### `\include` — compiled examples

Prefer `\include` over inlining code.
The included file is compiled by the `documentation` CMake target,
so the docs cannot drift from the code.

```cpp
/**
 * \brief Demonstrates array allocation and element access.
 *
 * \par Example
 * \include Containers/ArrayExample.cpp
 * \par Output
 * \include ArrayExample.out
 */
```

Rules:

- `\include <path>` shows the file without line numbers.
- `\includelineno <path>` shows it with line numbers.
- Paths are relative to the `Documentation/` directory
  (or the Doxygen `INPUT` root).
- Output files (`.out`) are produced by running the example
  and committed alongside the source.
- `\include` and `\includelineno` directives
  must be on their own line within the comment block.

### `\code{.cpp} ... \endcode` — inline snippets

Use `\code{.cpp}` for short snippets
that are not part of a compilable example:

```cpp
/**
 * \brief Evaluates the function `f` in parallel for all array elements.
 *
 * The function `f` is called as `f(indices...)`, where `indices...` are
 * substituted by the actual indices of all array elements. For example:
 *
 * \code{.cpp}
 * auto setter = [&a] ( int i, int j, int k )
 * {
 *    a( i, j, k ) = 1;
 * };
 * a.forAll( setter );
 * \endcode
 */
```

Rules:

- Always use `{.cpp}` as the language specifier.
- Keep snippets short —
  anything longer than ~10 lines belongs in a separate file
  pulled in with `\include`.
- Indent the snippet body to match the surrounding ` * ` prefix.

### Inline code in prose

Use backticks for inline identifiers, file names, type names,
and short code fragments:
`` `Array` ``, `` `operator[]` ``, `` `ArrayAllocation.cpp` ``.
Never use `\c` or `<tt>...</tt>`.

## Links

### External URLs

Use Markdown link syntax inside `/** */` blocks
(adapted from `src/TNL/Algorithms/scan.h`):

```cpp
/**
 * \brief Computes an inclusive scan (prefix sum) of an input array.
 *
 * [Inclusive scan (or prefix sum)](https://en.wikipedia.org/wiki/Prefix_sum)
 * turns a sequence \f$a_1, \ldots, a_n\f$ into a sequence
 * \f$s_1, \ldots, s_n\f$.
 */
```

### Internal cross-references

- Use `\ref <symbol>` for any Doxygen-documented symbol
  (class, function, namespace, file, group, type alias).
  Examples:
  `\ref TNL::Containers::Array "Array"`,
  `\ref getSize`,
  `\ref std::vector::resize`.
- Never use `\ref` for external URLs — use Markdown links.

## Math

Use Doxygen's LaTeX formula commands for mathematical expressions:

- Inline: `\f$ x^2 + y^2 = r^2 \f$`
- Display: `\f[ s_i = \sum_{j=1}^i a_i. \f]`

Rules:

- `\f$` wraps inline math;
  place it on the same line as the surrounding prose.
- `\f[ ... \f]` wraps display math;
  place the opening and closing markers on their own lines.
- Use real LaTeX commands: `\sum`, `\ldots`, `\infty`, `\frac{}{}`,
  `\sqrt{}`, subscripts `_`, superscripts `^`.

## Documenting entities

### Classes and structs

(adapted from `src/TNL/Containers/Array.h`)

```cpp
/**
 * \brief Manages memory, element access, and operations for arrays.
 *
 * \tparam Value  The type of array elements.
 * \tparam Device The device for execution of array operations.
 *                It can be any class defined in the \ref TNL::Devices
 *                namespace.
 * \tparam Index  The indexing type.
 * \tparam Allocator The allocator type for memory management.
 *                    By default, an appropriate allocator for the
 *                    specified *Device* is selected with
 *                    \ref TNL::Allocators::Default.
 *
 * Memory management is handled by constructors and destructors
 * following the [RAII](https://en.wikipedia.org/wiki/RAII) principle
 * and by methods \ref resize, \ref setSize, \ref swap, and \ref reset.
 *
 * Methods annotated as `__cuda_callable__` can be called from host
 * or from kernels executing on a device.
 *
 * \par Example
 * \include Containers/ArrayExample.cpp
 * \par Output
 * \include ArrayExample.out
 */
template< typename Value, typename Device = Devices::Host, ... >
class Array
```

### Methods and functions

```cpp
/**
 * \brief Resizes the array to the given size.
 *
 * If the array size changes, the current data will be deallocated,
 * thus all pointers and views to the array elements become invalid.
 *
 * \param size The new size of the array.
 */
void
resize( IndexType size );
```

Rules:

- `\brief` is a single sentence ending with a period.
- The detailed description follows after a blank line.
- `\param` entries come last, one per line.
- `\return` comes after `\param` entries when the function has a return value.
- If a method is callable from device kernels,
  mention `__cuda_callable__` in the detailed description.

### Namespaces

```cpp
//! \brief Namespace for TNL containers.
namespace TNL::Containers {
```

### Type aliases

```cpp
//! \brief Type of elements stored in this array.
using ValueType = Value;
```

## Line length and semantic line breaks

### Line length

The C++ line-length limit is **128 characters**
(per `AGENTS.md`).
Comment blocks follow the same limit.
By convention, comment prose wraps at approximately 80 columns
(including the ` * ` prefix and indentation),
matching the existing style in `src/TNL/`.
Lines longer than 80 columns are acceptable
when they contain a single token that cannot be wrapped —
a long URL, a long `\ref` command, or a long code span.

When editing an existing file,
match its wrapping width rather than rewrapping the whole file.

### Semantic line breaks

Wrap prose at semantic boundaries — one thought per line —
while staying within the ~80-column prose limit.
Apply the Semantic Line Breaks rules (<https://sembr.org>):

1. Break after every sentence ending in `.`, `!`, or `?`.
2. Prefer a break after independent clauses
   ending in `,`, `;`, `:`, or `—`.
3. Optionally break after a dependent clause
   when it clarifies structure.
4. Never break inside a hyphenated word, a code span, a URL,
   or a `\ref` command.
5. When a single sentence still exceeds the ~80-column prose limit,
   wrap at phrase boundaries (after conjunctions, prepositions, or commas)
   and align the continuation with the start of the sentence's text
   (after the ` * ` prefix).

Example (adapted from `src/TNL/Containers/Array.h`):

```cpp
/**
 * \brief `Array` is responsible for memory management,
 * access to array elements, and general array operations.
 *
 * \tparam Value  The type of array elements.
 * \tparam Device The device to be used for the execution of array operations.
 *                It can be any class defined
 *                in the \ref TNL::Devices namespace.
 */
```

### Regions where line breaks are syntactically significant

Do not apply semantic wrapping inside:

- `\code{.cpp} ... \endcode` blocks,
- `\f[ ... \f]` display math,
- `\include` / `\includelineno` / `\snippet` directives.

## Verification checklist

Before considering a documentation edit done, verify:

- [ ] All Doxygen commands use the backslash form, not `@`.
- [ ] Every doc block starts with `\brief`.
- [ ] `//!` is used for simple/short docstrings;
      `/** */` for complex blocks with multiple `\tparam`/`\param`/`\par`
      or a detailed description.
- [ ] No `/** */` blocks inside function bodies —
      use `//` for non-Doxygen implementation comments.
- [ ] No `///` triple-slash comments — use `//!` instead.
- [ ] No `\b`, `\e`, `\c`, `<b>`, `<tt>`, or `<em>` tags —
      Markdown emphasis only.
- [ ] Literal symbols with underscores (`__cuda_callable__`)
      are in backticks, not Markdown emphasis.
- [ ] `\par` titles use title case (`Example`, `Output`).
- [ ] `\param` and `\tparam` entries are in declaration order,
      after the detailed description.
- [ ] `\return` (not `\returns`) when the function returns a value.
- [ ] Cross-references to symbols use `\ref`;
      external URLs use Markdown links.
- [ ] Code examples use `\include` for compiled examples,
      `\code{.cpp}` for short snippets.
- [ ] Semantic Line Breaks rules are properly applied.
- [ ] Prose wraps at ~80 columns (or the file's existing width)
      at semantic boundaries.
- [ ] `just check-typos` passes on the edited file.
- [ ] If the edit changed indentation or other code formatting,
      run `just format` to fix it before finishing.
- [ ] Run `just build documentation` to catch Doxygen warnings
      and ensure the documentation still generates.
      Note that this target also runs documentation examples;
      environment-specific failures such as MPI accelerator conflicts are unrelated to documentation quality.
      If Open MPI aborts with "only supports one accelerator framework per node",
      set `OMPI_MCA_accelerator=cuda` (or `rocm`) before running the build.

## Reference files

Mirror the structural patterns in these existing source files,
but replace any legacy `\e` / `\c` / `_..._` / `__...__` emphasis per the rules above —
the referenced files predate this skill
and may contain forbidden forms.

| File | Pattern |
|------|---------|
| `src/TNL/Containers/Array.h` | `/** */` class doc: `\brief` + `\tparam` + detailed description + `\par Example` / `\par Output`; `\ref` cross-references; Markdown links |
| `src/TNL/Containers/NDArray.h` | `//!` one-line docstrings for type aliases and simple methods; `\code{.cpp} ... \endcode` snippets inside method docs |
| `src/TNL/Containers/ndarray/SizesHolder.h` | `//!` multi-line docstrings with `//!` continuation lines |
| `src/TNL/Algorithms/scan.h` | Free function docs with `\f$` / `\f[` math, Markdown links to Wikipedia |
| `src/TNL/Containers/ndarray/Reduce.h` | `\note` blocks for multi-paragraph explanations; `\see` for related functions |
| `src/TNL/Backend/Stream.h` | `\warning` block with `\code{.cpp}` example showing incorrect usage |
| `src/TNL/Meshes/Grid.h` | Struct doc with `\tparam` entries on single lines |
| `src/TNL/Containers/Array.hpp` | `//` single-line comments inside function bodies (never `/** */`) |
