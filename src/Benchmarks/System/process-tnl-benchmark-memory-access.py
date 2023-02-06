#!/usr/bin/python3

import os
import json
import pandas as pd
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt
import numpy as np
import math
from os.path import exists

threads = [ ]
accesses = [ "sequential", "random" ]
element_sizes = [ 1, 2, 4, 8, 16, 32, 64, 128, 256 ]

####
# Create multiindex for columns
def get_multiindex():
    level1 = [ 'size' ]
    level2 = [ '',    ]
    level3 = [ '',    ]
    level4 = [ '',    ]
    df_data = [[ ' ' ]]
    for threads_count in threads:
        for access in accesses:
            for element_size in element_sizes:
                values = ['time', "bandwidth", "CPU cycles"]
                for value in values:
                    level1.append( f'Threads {threads_count}' )
                    level2.append( access )
                    level3.append( element_size )
                    level4.append( value )
                    df_data[0].append( '' )

    multiColumns = pd.MultiIndex.from_arrays([ level1, level2, level3, level4 ] )
    return multiColumns, df_data


####
# Process dataframe for given precision - float or double
def processDf( df ):
    multicolumns, df_data = get_multiindex()

    frames = []
    in_idx = 0
    out_idx = 0

    sizes = list(set(df['array size']))

    print( sizes )

    for size in sizes:
        aux_df=df.loc[ ( df['array size'] == size ) ]
        #print( aux_df )
        new_df = pd.DataFrame( df_data, columns = multicolumns, index = [out_idx] )
        out_idx += 1
        new_df.iloc[0][ ('size','','','') ]  = size
        for index, row in aux_df.iterrows():
            threads = row[ 'threads' ]
            access_type = row[ 'access type' ]
            element_size = row[ 'element size' ]
            time = row[ 'time' ]
            bandwidth = row[ 'bandwidth' ]
            cpu_cycles = row[ 'cycles/op.' ]
            print( f'Threads {threads} {access_type} {element_size} {time} {bandwidth} {cpu_cycles} ')
            new_df.iloc[0][( f'Threads {threads}', access_type, element_size,'time') ] = time
            new_df.iloc[0][( f'Threads {threads}', access_type, element_size,'bandwidth') ] = bandwidth
            new_df.iloc[0][( f'Threads {threads}', access_type, element_size,'CPU cycles') ] = cpu_cycles
        #print( new_df.iloc[0][( f'Threads {threads}', access_type, element_size,'time') ] )
        frames.append( new_df)
    result = pd.concat( frames )
    result.to_html( f'tnl-benchmark-memory-access.html' )

#####
# Parse input files
parsed_lines = []
filename = f"tnl-benchmark-memory-access.log"
if not exists( filename ):
    print( f"Skipping non-existing input file {filename} ...." )
print( f"Parsing input file {filename} ...." )
with open( filename ) as f:
    lines = f.readlines()
    for line in lines:
        parsed_line = json.loads(line)
        parsed_lines.append( parsed_line )

df = pd.DataFrame(parsed_lines)
for x in df['threads']:
    if x not in threads:
        threads.append( x )

keys = ['array size','time', 'bandwidth', 'cycles/op' ]

for key in keys:
    if key in df.keys():
        df[key] = pd.to_numeric(df[key])

df.to_html( 'tnl-benchmark-memory-access.html' )
processDf( df )
