// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Code for performance counters on Apple is inspired by https://lemire.me/blog/2021/03/24/counting-cycles-and-instructions-on-the-apple-m1-processor/

#pragma once

#include <TNL/Containers/Array.h>

namespace TNL {

/**
 * \brief Performance counter for measuring CPU cycles.
 */
struct PerformanceCounter
{

   /**
    * \brief Constructor with no parameters.
    */
   PerformanceCounter();

   /**
    * \brief Function for counting the number of CPU cycles (machine cycles).
    */
   unsigned long long int getCPUCycles() const;


#ifdef __APPLE__
   static constexpr int appleKpcCountersMaxCount = 10;
   static constexpr int appleKpcConfigMaxCount = 8;
#endif

protected:

   /**
    * \brief Time Stamp Counter returning number of CPU cycles since reset.
    */
   static inline unsigned long long
   rdtsc();

#ifdef __APPLE__
   // inspired by https://lemire.me/blog/2021/03/24/counting-cycles-and-instructions-on-the-apple-m1-processor/
   static void* kperf;

   const int CFGWORD_EL0A32EN_MASK = 0x10000;
   const int CFGWORD_EL0A64EN_MASK = 0x20000;
   const int CFGWORD_EL1EN_MASK = 0x40000;
   const int CFGWORD_EL3EN_MASK = 0x80000;
   const int CFGWORD_ALLMODES_MASK = 0xf0000;

   const int CPMU_NONE = 0;
   const int CPMU_CORE_CYCLE = 0x02;
   const int CPMU_INST_A64 = 0x8c;
   const int CPMU_INST_BRANCH = 0x8d;
   const int CPMU_SYNC_DC_LOAD_MISS = 0xbf;
   const int CPMU_SYNC_DC_STORE_MISS = 0xc0;
   const int CPMU_SYNC_DTLB_MISS = 0xc1;
   const int CPMU_SYNC_ST_HIT_YNGR_LD = 0xc4;
   const int CPMU_SYNC_BR_ANY_MISP = 0xcb;
   const int CPMU_FED_IC_MISS_DEM = 0xd3;
   const int CPMU_FED_ITLB_MISS = 0xd4;

   const int KPC_CLASS_FIXED = 0;
   const int KPC_CLASS_CONFIGURABLE = 1;
   const int KPC_CLASS_POWER  = 2;
   const int KPC_CLASS_RAWPMU = 3;
   const int KPC_CLASS_FIXED_MASK = 1u << KPC_CLASS_FIXED;
   const int KPC_CLASS_CONFIGURABLE_MASK = 1u << KPC_CLASS_CONFIGURABLE;
   const int KPC_CLASS_POWER_MASK = 1u << KPC_CLASS_POWER;
   const int KPC_CLASS_RAWPMU_MASK = 1u << KPC_CLASS_RAWPMU;
   const int KPC_MASK = KPC_CLASS_CONFIGURABLE_MASK | KPC_CLASS_FIXED_MASK;

   static uint64_t g_counters[ appleKpcCountersMaxCount ];
   static uint64_t g_config[ appleKpcConfigMaxCount ];

   static int ( *kpc_set_config )( uint32_t, void* );
   static int ( *kpc_force_all_ctrs_set )( int );
   static int ( *kpc_set_counting )( int );
   static int ( *kpc_set_thread_counting )( int );
   static int ( *kpc_get_counter_count )( int );
   static int ( *kpc_get_config_count )( int );
   static int ( *kpc_get_thread_counters )( int, unsigned int, void * );

   int appleKpcCountersCount = 0;
#endif
};

#ifdef __APPLE__
inline void* PerformanceCounter::kperf = nullptr;
inline uint64_t PerformanceCounter::g_counters[PerformanceCounter::appleKpcCountersMaxCount];
inline uint64_t PerformanceCounter::g_config[PerformanceCounter::appleKpcConfigMaxCount];
#endif


} // namespace TNL

#include <TNL/PerformanceCounter.hpp>