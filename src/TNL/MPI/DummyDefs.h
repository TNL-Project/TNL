// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#ifndef HAVE_MPI
using MPI_Request = int;
using MPI_Comm = int;
using MPI_Info = int;

// NOLINTNEXTLINE(performance-enum-size)
enum MPI_Op
{
   MPI_MAX,
   MPI_MIN,
   MPI_SUM,
   MPI_PROD,
   MPI_LAND,
   MPI_BAND,
   MPI_LOR,
   MPI_BOR,
   MPI_LXOR,
   MPI_BXOR,
   MPI_MINLOC,
   MPI_MAXLOC,
};

// Comparison results
// NOLINTNEXTLINE(performance-enum-size)
enum
{
   MPI_IDENT,
   MPI_CONGRUENT,
   MPI_SIMILAR,
   MPI_UNEQUAL
};

// MPI_Init_thread constants
// NOLINTNEXTLINE(performance-enum-size)
enum
{
   MPI_THREAD_SINGLE,
   MPI_THREAD_FUNNELED,
   MPI_THREAD_SERIALIZED,
   MPI_THREAD_MULTIPLE
};

// Miscellaneous constants
// NOLINTNEXTLINE(performance-enum-size)
enum
{
   MPI_ANY_SOURCE = -1,    /* match any source rank */
   MPI_PROC_NULL = -2,     /* rank of null process */
   MPI_ROOT = -4,          /* special value for intercomms */
   MPI_ANY_TAG = -1,       /* match any message tag */
   MPI_UNDEFINED = -32766, /* undefined stuff */
   MPI_DIST_GRAPH = 3,     /* dist graph topology */
   MPI_CART = 1,           /* cartesian topology */
   MPI_GRAPH = 2,          /* graph topology */
   MPI_KEYVAL_INVALID = -1 /* invalid key value */
};

// MPI handles
// (According to the MPI standard, they are only link-time constants (not
// compile-time constants). OpenMPI implements them as global variables.)
// NOLINTNEXTLINE(performance-enum-size)
enum
{
   MPI_COMM_WORLD = 1,
   MPI_COMM_SELF = MPI_COMM_WORLD,
};

// NULL handles
// NOLINTNEXTLINE(performance-enum-size)
enum
{
   MPI_GROUP_NULL = 0,
   MPI_COMM_NULL = 0,
   MPI_REQUEST_NULL = 0,
   MPI_MESSAGE_NULL = 0,
   MPI_OP_NULL = 0,
   MPI_ERRHANDLER_NULL = 0,
   MPI_INFO_NULL = 0,
   MPI_WIN_NULL = 0,
   MPI_FILE_NULL = 0,
   MPI_T_ENUM_NULL = 0
};

#endif
