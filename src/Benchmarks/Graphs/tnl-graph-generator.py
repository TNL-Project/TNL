#!/usr/bin/env python3

import argparse
import random
import networkx as nx
from itertools import combinations

def generate_random_graph(nodes, edges, edge_weights_distribution):
    G = nx.Graph()
    G.add_nodes_from(range(nodes))

    # Generate all possible edges
    possible_edges = list(combinations(range(nodes), 2))

    # Randomly select edges to be added
    selected_edges = random.sample(possible_edges, min(edges, len(possible_edges)))

    for u, v in selected_edges:
        if edge_weights_distribution == "uniform":
            weight = random.randint(1, 10)
        elif edge_weights_distribution == "normal":
            weight = max(1, round(random.gauss(5, 2)))
        elif edge_weights_distribution == "exponential":
            weight = random.expovariate(1/5)
        elif edge_weights_distribution == "poisson":
            weight = random.poisson(3)
        elif edge_weights_distribution == "lognormal":
            weight = random.lognormvariate(0, 1)
        elif edge_weights_distribution == "triangular":
            weight = random.triangular(1, 10, 5)
        elif edge_weights_distribution == "beta":
            weight = random.betavariate(2, 5)
        else:
            raise ValueError("Invalid edge_weights_distribution. Must be 'uniform' or 'normal'.")

        G.add_edge(u, v, weight=weight)

    return G

def write_edge_list_to_file(graph, file_name):
    with open(file_name, 'w') as file:
        for u, v, data in graph.edges(data=True):
            file.write(f"{u} {v} {data['weight']}\n")

def main():
    parser = argparse.ArgumentParser(description='Generate a random graph.')
    parser.add_argument('--nodes', type=int, required=True, help='Number of nodes in the graph')
    parser.add_argument('--edges', type=int, required=True, help='Number of edges in the graph')
    parser.add_argument('--edge-weights-distribution', type=str, choices=['uniform', 'normal', 'exponential', 'poisson', 'lognormal', 'triangular', 'beta'], default='normal',
                    help='Distribution of edge weights: "uniform", "normal", "exponential", "poisson", "lognormal", "triangular", or "beta"')
    parser.add_argument('--file-name', type=str, required=True, help='Name of the file to write the edge list')

    args = parser.parse_args()

    graph = generate_random_graph(args.nodes, args.edges, args.edge_weights_distribution)
    write_edge_list_to_file(graph, args.file_name)

if __name__ == "__main__":
    main()
