#!/usr/bin/env python3
import argparse
import cudf
import cugraph
import time

def read_input_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('#'):
                continue  # Skip comments
            src, dst = map(int, line.strip().split(' '))
            data.append((src, dst))
    df = cudf.DataFrame(data, columns=['src', 'dst'])
    return df


# Create a cuGraph graph from the DataFrame
def create_cugraph_graph(df):
    G = cugraph.Graph(directed=True)
    G.from_cudf_edgelist(df, source='src', destination='dst')
    return G

# Perform BFS and return a vector of distances from the source node
def bfs_distances(G, source_vertex):
    bfs_df = cugraph.bfs(G, source_vertex)
    bfs_df_pd = bfs_df.to_pandas()
    distances = bfs_df_pd['distance'].to_numpy()
    return distances

def main():
    parser = argparse.ArgumentParser(description='Process a graph file and run cuGraph algorithms.')
    parser.add_argument('--input-file', type=str, required=True, help='Path to the input file')
    args = parser.parse_args()

    # Read the input file and create a DataFrame
    print(f"Reading input file: {args.input_file}")
    df = read_input_file(args.input_file)

    # Create a cuGraph graph from the DataFrame
    G = create_cugraph_graph(df)

    # Perform BFS and get distances from the source vertex
    print("Performing BFS...")
    source_vertex = 0

    start_time = time.time()
    distances = bfs_distances(G, 0 ) #source_vertex)
    end_time = time.time()
    print(f"cuGraph BFS took {end_time - start_time} seconds")

    # Print the distances
    #for i, distance in enumerate(distances):
    #    print(f"Distance from vertex {source_vertex} to vertex {i}: {distance}")

if __name__ == '__main__':
    main()
