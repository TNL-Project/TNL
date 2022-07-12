#!/usr/bin/python3

import os
import json
import pandas as pd
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt
import numpy as np
import math

devices = [ "host", "cuda" ]
precisions = [ "float", "double" ]
tests = [ "parallel-for", "parallel-for-with-memory-load", "grid", "nd-grid"  ]

####
# Create multiindex for columns
def get_multiindex():
    level1 = [ 'xSize', 'ySize' ]
    level2 = [ '',      ''      ]
    level3 = [ '',      ''      ]
    df_data = [[ ' ',' ']]
    for test in tests:
        for device in [ 'Host', 'Cuda' ]:
            values = ['time']
            if test != 'parallel-for':
                values.append( 'parallel-for speed-up' )
            if device == 'Cuda':
                values.append( 'CPU speed-up' )
            for value in values:
                level1.append( test )
                level2.append( device )
                level3.append( value )
                df_data[0].append( '' )

    multiColumns = pd.MultiIndex.from_arrays([ level1, level2, level3 ] )
    return multiColumns, df_data


####
# Process dataframe for given precision - float or double
def processDf( df, precision ):
    multicolumns, df_data = get_multiindex()

    frames = []
    in_idx = 0
    out_idx = 0

    x_sizes = list(set(df['xSize']))
    x_sizes.sort()
    y_sizes = list(set(df['ySize']))
    y_sizes.sort()

    for x_size in x_sizes:
        for y_size in y_sizes:
            aux_df=df.loc[ ( df['xSize'] == x_size ) & ( df['ySize'] == y_size ) ]
            new_df = pd.DataFrame( df_data, columns = multicolumns, index = [out_idx] )
            out_idx += 1
            new_df.iloc[0][ ('xSize','','') ]  = x_size
            new_df.iloc[0][ ('ySize','','') ]  = y_size
            for index, row in aux_df.iterrows():
                device = 'Host'
                if row[ 'performer'] == 'TNL::Devices::Cuda':
                    device = "Cuda"
                test = row[ 'id' ]
                time = row[ 'time' ]
                new_df.iloc[0][(test,device,'time') ] = time
            #print( new_df )
            frames.append( new_df)
    result = pd.concat( frames )
    idx = 0
    for index, row in result.iterrows():
        for test in tests:
            result.iloc[idx][ (test, 'Cuda', 'CPU speed-up') ] =  row[ (test, 'Host', 'time')] / row[ (test, 'Cuda', 'time')]
            if test != 'parallel-for':
                for device in [ 'Host', 'Cuda' ]:
                    result.iloc[idx][ (test, device, 'parallel-for speed-up') ] =  row[ ('parallel-for', device, 'time')] / row[ (test, device, 'time')]
        idx += 1

    result.to_html( f'output-{precision}.html' )






#####
# Parse input files

parsed_lines = []
for device in devices:
    for precision in precisions:
        for test in tests:
            filename = f"tnl-benchmark-heat-equation-{test}-{device}-{precision}.json"
            print( f"Parsing input file {filename} ...." )
            with open( filename ) as f:
                lines = f.readlines()
                for line in lines:
                    parsed_line = json.loads(line)
                    parsed_lines.append( parsed_line )

df = pd.DataFrame(parsed_lines)

keys = ['xSize', 'ySize', 'zSize', 'time', 'bandwidth' ]

for key in keys:
    if key in df.keys():
        df[key] = pd.to_numeric(df[key])

for precision in precisions:
    aux_df = df.loc[ ( df['precision'] == precision ) ]
    processDf( aux_df, precision )

