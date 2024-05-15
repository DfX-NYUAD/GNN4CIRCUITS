'''
Author: Hana Selmani
Created on 30/05/2023
Last modified on 25/04/2024
'''

import os
import re
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import sys
import dgl
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
from dgl.nn import GraphConv, GINConv, PNAConv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.nn import BatchNorm1d
import time
import argparse
from pyverilog.vparser.parser import parse
from pyverilog.dataflow.dataflow_analyzer import VerilogDataflowAnalyzer
from pyverilog.dataflow.optimizer import VerilogDataflowOptimizer
from pyverilog.dataflow.graphgen import VerilogGraphGenerator

global parsed_library
graph_list = []
current_module_types = []
flag_vector = []
feature_vector = []
gate_types = []
gate_type_dict = {}
current_module = ""
current_file_num = 0
names_dict = {}

"""
Gate-level Netlist parsing
"""

# function which parses the technology library
# it returns a dictionary with the module name as the key and the input and output pins as the value
def parse_library(path):
    module_info = {}
    with open(path, 'r') as file:
        found_module = found_input = found_output = inNode = False
        module_name = ""
        for line in file:
            line = line.strip()  # Remove leading and trailing whitespace
            if line:
                words = line.split()
                if words:
                    # if the line starts with module, we can start parse module info
                    if words[0] == "module":
                        inNode = True
                    # if the line starts with endmodule, we are done parsing the module
                    elif words[0] == "endmodule":
                        inNode = False
                if inNode:
                    if words[0] == "module": # if we are in the first line where module is declared
                        # get current module name as the first word after 'module'
                        module_name = gate_type = words[1].strip("();")
                        gate_type = remove_trailing_numbers(gate_type.split('_')[0])
                        if "gate_type" in flag_vector:
                            if gate_type not in gate_types:
                                gate_types.append(gate_type)
                        found_module = False
                    # if we are in the input section, add the word to the input list
                    elif found_input:
                        if line.endswith(';'):  
                            found_input = False
                        if module_name not in module_info:
                            module_info[module_name] = [[], []]
                        words = [word.strip(",") for word in words]
                        module_info[module_name][0].extend(words)
                    # if we are in the output section, add the word to the output list
                    elif found_output:
                        if line.endswith(';'):                           
                            found_output = False
                        if module_name not in module_info:
                            module_info[module_name] = [[], []]
                        words = [word.strip(",") for word in words]
                        module_info[module_name][1].extend(words)
                    # if we find the word 'input', we are in the input section
                    elif words[0] == "input":
                        if line.endswith(';'):
                            # the last line of input is found, so there 
                            # is no need to repeat searching for ports
                            found_input = False
                        else:
                            found_input = True
                        if module_name not in module_info:
                            module_info[module_name] = [[], []]
                        words = [word.strip(",;") for word in words]
                        module_info[module_name][0].extend(words[1:])
                    # if we find the word 'output', we are in the output section
                    elif words[0] == "output":
                        if line.endswith(';'):
                            found_output = False
                        else:
                            found_output = True
                        if module_name not in module_info:
                                module_info[module_name] = [[], []]
                        words = [word.strip(",;") for word in words]
                        module_info[module_name][1].extend(words[1:])
    return module_info

# function which parses the passed verilog file
# and adds edges to the graph based on the inputs and outputs
def verilog_to_gate(path):
    G = graph_list[current_file_num]
    with open(path, 'r') as file:
        reached_gate_info = False
        module_lines = []
        for line in file:
            # if the parsing has reached the section of the file with the gate information
            # proceed with parsing
            if line.strip().startswith(tuple(parsed_library.keys())):
                reached_gate_info = True
            if reached_gate_info:
                line = line.strip()
                module_lines.append(line)
                if line.endswith(';'):
                    current_line = ' '.join(module_lines)
                    line_parts = current_line.split('(', 1)  # Split the line by '('
                    if len(line_parts) < 2:
                        continue  # Skip lines that don't conform to the expected format
                    module_info = line_parts[0].strip().split()  
                    line_parts[1] = line_parts[1].strip(');\n')  
                    pin_info = line_parts[1].split(',')  
                    pin_info = re.findall(r'\.(\w+)', line_parts[1])  # Extract the pin names using regex
                    pin_specific = re.findall(r'\((.*?)\)', line_parts[1])  # Extract the specific pin using regex

                    module_name = module_info[0]
                    instance_name = module_info[1]

                    module_name_copy = module_name
                    module_name_copy = remove_trailing_numbers(module_name_copy.split('_')[0])
                    # for each input and output type as is specified in the library for this module type
                    for input in parsed_library[module_name][0]:
                        for output in parsed_library[module_name][1]:
                            # match the library pin type to see if it is an input or an output. using the
                            # respective indices, get the specific pin string which corresponds to the type
                            # and whether it is an input or an output
                            postfix = "_" + current_module
                            
                            G.add_edge(pin_specific[pin_info.index(input)]+postfix, pin_specific[pin_info.index(output)]+postfix, input_type=input, output_type=output)
                            G.nodes[pin_specific[pin_info.index(input)]+postfix]['label'] = current_module
                            G.nodes[pin_specific[pin_info.index(output)]+postfix]['label'] = current_module
                            if "gate_type" in flag_vector:
                                if module_name_copy not in current_module_types:
                                    current_module_types.append(module_name_copy)
                                gate_type_dict[pin_specific[pin_info.index(output)]+postfix] = module_name_copy
                    module_lines = []  # Reset the accumulated lines for the next module


# function which parses the passed verilog file
# specifically the module information at the beginning
# it returns a dictionary with the module name as the key and the input, output pins and wires as the value
def parse_file_information(file_path):
    modules = {}
    inModule = inInput = inOutput = inWire = found_array = False
    array_pattern = r'\[(\d+):(\d+)\]'

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line.endswith(');') and not line.endswith(';'):
                line = line.rstrip('\n')

            # if the line starts with module, we can start parse module info
            if line.startswith('module'):
                inModule = True
                module_name = line.split(' ')[1]
                modules[module_name] = [[], [], []]
                global current_module
                current_module = module_name
            # if the line starts with input, we are in the input section
            elif line.startswith('input'):
                inputs = re.findall(r'\b(?!input\b)(\w+)\b', line)
                for i in range(len(inputs)):
                    inputs[i] = inputs[i] + "_" + current_module
                modules[current_module][0].extend(inputs)
                # if the line ends with a semicolon, we are done with the input section
                if line.endswith(';'):
                    inInput = False
                else:
                    inInput = True
            # if more lines in the input section
            elif inInput:
                inputs = re.findall(r'\b(\w+)\b', line)
                #add postfix to all elements in the list
                for i in range(len(inputs)):
                    inputs[i] = inputs[i] + "_" + current_module
                modules[current_module][0].extend(inputs)
                if line.endswith(';'):
                    inInput = False
            # if the line starts with output, we are in the output section
            elif line.startswith('output'):
                outputs = re.findall(r'\b(?!output\b)(\w+)\b', line)
                for i in range(len(outputs)):
                    outputs[i] = outputs[i] + "_" + current_module
                modules[current_module][1].extend(outputs)
                # if the line ends with a semicolon, we are done with the output section
                if line.endswith(';'):
                    inOutput = False
                else:
                    inOutput = True
            # if more lines in the output section
            elif inOutput:
                outputs = re.findall(r'\b(\w+)\b', line)
                for i in range(len(outputs)):
                    outputs[i] = outputs[i] + "_" + current_module
                modules[current_module][1].extend(outputs)
                if line.endswith(';'):
                    inOutput = False
            # if the line starts with wire, we are in the wire section
            elif line.startswith('wire'):
                wires = re.findall(r'\b(?!wire\b)(\w+)\b', line)
                for i in range(len(wires)):
                    wires[i] = wires[i] + "_" + current_module
                modules[current_module][2].extend(wires)
                # if the line ends with a semicolon, we are done with the wire section
                if line.endswith(';'):
                    inWire = False
                else:
                    inWire = True

            # if more lines in the wire section
            elif inWire:
                wires = re.findall(r'\b(\w+)\b', line)
                for i in range(len(wires)):
                    wires[i] = wires[i] + "_" + current_module
                modules[current_module][2].extend(wires)
                if line.endswith(';'):
                    inWire = False
            elif line.startswith(tuple(parsed_library.keys())): # If the line starts with the gate types, stop parsing
                return modules
    return modules

def RTL_parser(verilog_file):
    # Setup the dataflow analysis with the provided Verilog file
    dataflow_analyzer = VerilogDataflowAnalyzer(verilog_file, "top")
    dataflow_analyzer.generate()
    terms_data = dataflow_analyzer.getTerms()
    bindings_dict = dataflow_analyzer.getBinddict()
    
    # Optimize the dataflow from the Verilog code
    dataflow_optimizer = VerilogDataflowOptimizer(terms_data, bindings_dict)
    dataflow_optimizer.resolveConstant()
    optimized_terms = dataflow_optimizer.getResolvedTerms()
    optimized_bindings = dataflow_optimizer.getResolvedBinddict()
    constants_list = dataflow_optimizer.getConstlist()
    graph_generator = VerilogGraphGenerator("top", terms_data, bindings_dict, optimized_terms, 
                                            optimized_bindings, constants_list, './separate_modules.pdf')

    # Convert the bindings dictionary to a list of strings
    string_bind = [str(binding) for binding in graph_generator.binddict]

    # Generate the graph for each signal, set walking through hierarchy to False
    for bind in sorted(string_bind, key=str.casefold):
        graph_generator.generate(bind, walk=False)

    # Get source node labels
    source_nodes = {}
    for node in graph_generator.graph.nodes():
        if graph_generator.graph.in_degree(node) == 0:
            label = node.attr['label'] if node.attr['label'] != '\\N' else str(node)
            source_nodes[label] = node

    # Node deletion and edge redirection
    for node in graph_generator.graph.nodes():
        node_label = node.attr['label'] if node.attr['label'] != '\\N' else str(node)
        if '_' in node_label and node_label.replace('_', '.') in source_nodes:
            node_predecessors = graph_generator.graph.predecessors(node)
            graph_generator.graph.remove_node(node)
            for predecessor in node_predecessors:
                graph_generator.graph.add_edge(predecessor, source_nodes[node_label.replace('_', '.')])

    # Construct the graph 
    nx_graph = nx.DiGraph()
    
    for node in graph_generator.graph.nodes():
        node_name = node.name
        if '_graphrename' in node_name:
            node_name = node_name[:node_name.index('_graphrename')]
        node_type = node_name.split('.')[-1] if '.' in node_name else (node_name.split('_')[-1] if '_' in node_name else node_name.lower())
        nx_graph.add_node(node.name, label=node_type)
        for child in graph_generator.graph.successors(node):
            nx_graph.add_edge(node.name, child.name)

    return nx_graph

# Populate the feature vector for each node
def update_features(flag_vector, key_string):
    functions = {
        "input_degree": input_degree,
        "output_degree": output_degree,
        "io_degree": io_degree,
        "min_dist_input": min_dist_input,
        "min_dist_output": min_dist_output,
        "primary_input": direct_link_to_pi,
        "primary_output": direct_link_to_po,
        "gate_type": gate_type,
    }
    for G in graph_list:
        for node in G.nodes():
            node_features = [node] + [func(G, node) for flag, func in functions.items() if flag in flag_vector and flag != "ki" and flag!= "gate_type"]
            if "ki" in flag_vector:
                node_features.append(key_input(G, node, key_string))
            if "gate_type" in flag_vector:
                node_features.extend(gate_type(G, node))
            G.add_node(node, feat=node_features[1:])
            feature_vector.append(node_features)

def input_degree(G, node):
    return G.in_degree(node)

def output_degree(G, node):
    return G.out_degree(node)

def io_degree(G, node):
    return G.in_degree(node) + G.out_degree(node)

# function to get the minimum distance from a node to a primary input
def min_dist_input(G, source_node):
    reversed_graph = G.reverse()

    # Calculate shortest path lengths from target inputs to the source node
    shortest_path_lengths = nx.single_source_shortest_path_length(reversed_graph, source_node)

    # list which will store all possible primary inputs
    target_nodes = []
    # Get the primary inputs from the module_in_file dictionary
    for value in module_in_file.values():
        target_nodes.extend(value[0])

    # Filter out target nodes without a valid shortest path length
    valid_target_nodes = [node for node in target_nodes if node != source_node and node in shortest_path_lengths]

    if valid_target_nodes:
        # Find the primary input with the minimum shortest path length to the source
        closest_target = min(valid_target_nodes, key=lambda x: shortest_path_lengths.get(x))
        distance = shortest_path_lengths[closest_target]
        # return closest_target, distance
        return distance

    # in the case that there are no paths to a primary input
    else:
        return 0

# function to get the minimum distance from a node to a primary output
def min_dist_output(G, source_node):

   # Calculate shortest path lengths from the given node
    shortest_path_lengths = nx.single_source_shortest_path_length(G, source_node)

    # list which will store all possible primary outputs
    target_nodes = []
    # Get the primary outputs from the module_in_file dictionary
    for value in module_in_file.values():
        target_nodes.extend(value[1])

    # Filter out primary outputs without a valid shortest path length
    valid_target_nodes = [node for node in target_nodes if node != source_node and node in shortest_path_lengths]

    if valid_target_nodes:
        # Find the primary outputs with the minimum shortest path length
        closest_target = min(valid_target_nodes, key=lambda x: shortest_path_lengths.get(x))
        distance = shortest_path_lengths[closest_target]
        # return closest_target, distance
        return distance

    # in the case that there are no paths to a primary output
    else:
        return 0

# function to check if a node is directly linked to a primary input
def direct_link_to_pi(G, node):
    # list which will store all possible primary inputs
    target_nodes = []
    # Get the primary inputs from the module_in_file dictionary
    for value in module_in_file.values():
        target_nodes.extend(value[0])

    #check if an edge is present from target node to node
    for target in target_nodes:
        if G.has_edge(target, node):
            return 1
    return 0

# function to check if a node is directly linked to a primary output
def direct_link_to_po(G, node):
        # list which will store all possible primary outputs
        target_nodes = []
        # Get the primary outputs from the module_in_file dictionary
        for value in module_in_file.values():
            target_nodes.extend(value[1])
    
        #check if an edge is present from target node to node
        for target in target_nodes:
            if G.has_edge(node, target):
                return 1
        return 0

# function which gets the gate type
def gate_type(G, node):
    one_hot = [0] * len(gate_types)
    if node in gate_type_dict.keys():
        gate_type = gate_type_dict[node]
        # one hot encoding
        one_hot[gate_types.index(gate_type)] = 1
    return one_hot

def get_inputs(G, node):
    inputs = []
    for edge in G.in_edges(node):
        inputs.append(edge[0])
    return inputs

def get_outputs(G, node):
    outputs = []
    for edge in G.out_edges(node):
        outputs.append(edge[1])
    return outputs

# function to check if a node has an input which matches the user specified string
def key_input(G, node, key_string):
    regular_node = False
    inputs = get_inputs(G, node)
    count = 0
    for char in key_string:
        if char.isalpha():
            count += 1
    # distinguishing between key input and a regular node
    # regular nodes have a structure of beginning with n or N, and being followed by numbers
    # hence only having one alpha character
    if count==1:
        regular_node = True
    if not regular_node:
        # matching the string and node without any numbers and insensitive to case
        # key_string = remove_trailing_numbers(key_string)
        for i in range(len(inputs)):
            # inputs[i] = remove_trailing_numbers(inputs[i])
            if key_string.lower() in inputs[i].lower():
                return 1
        return 0
    # if the node is a regular node, then we check if the key string is present in the inputs
    else:
        # the structure is more rigid for regular nodes
        # so no string manipulation is required
        if key_string in inputs:
            return 1
        return 0

def create_csv_files(graphs, graph_ids, labels, node_features_list):
    # Ensure the input lists have the same length
    if len(graphs) != len(graph_ids) or len(graphs) != len(labels) or len(graphs) != len(node_features_list):
        raise ValueError("Input lists must have the same length.")

    os.makedirs("files4training", exist_ok=True)
    # Write header to graph_edges CSV file
    edges_file = 'files4training/graph_edges.csv'
    fieldnames_edges = ['graph_id', 'src', 'dst']

    with open(edges_file, 'w', newline='') as csvfile:
        writer_edges = csv.DictWriter(csvfile, fieldnames=fieldnames_edges)
        writer_edges.writeheader()

        # Iterate through each graph and write edges
        for graph, graph_id, label in zip(graphs, graph_ids, labels):
            src, dst = graph.edges()
            for s, d in zip(src.numpy(), dst.numpy()):
                writer_edges.writerow({'graph_id': graph_id, 'src': s, 'dst': d})

    # Write header to graph_properties CSV file
    properties_file = 'files4training/graph_properties.csv'

    labels_set = list(set(labels.values()))
    label_mapping = {l: index for index, l in enumerate(labels_set)}

    fieldnames_properties = ['graph_id', 'label', 'num_nodes', 'label_string']

    with open(properties_file, 'w', newline='') as csvfile:
        writer_properties = csv.DictWriter(csvfile, fieldnames=fieldnames_properties)
        writer_properties.writeheader()

        # Iterate through each graph and write properties
        for graph, graph_id in zip(graphs, graph_ids):
            num_nodes = graph.number_of_nodes()
            label = labels[graph_id]  # Corrected line
            writer_properties.writerow({'graph_id': graph_id, 'label': label_mapping[label], 'num_nodes': num_nodes, 'label_string': label})

     # Write header to node_features CSV file
    node_features_file = 'files4training/node_features.csv'
    fieldnames_node_features = ['graph_id', 'node_id'] + [f'feat_{i}' for i in range(len(list(node_features_list[0][0])))]

    with open(node_features_file, 'w', newline='') as csvfile:
        writer_node_features = csv.DictWriter(csvfile, fieldnames=fieldnames_node_features)
        writer_node_features.writeheader()

        # Iterate through each graph and write node features for each node
        for graph, graph_id, features_dict in zip(graphs, graph_ids, node_features_list):
            for node_id in graph.nodes():
                feat = features_dict[node_id.item()]
                writer_node_features.writerow({'graph_id': graph_id, 'node_id': node_id.item(), **{f'feat_{i}': feat[i] for i in range(len(feat))}})

def save_dataset():
    # Create subgraphs for nodes with the same 'label'
    subgraphs = []
    labels = {}
    graph_ids = []
    node_features_list = []
    i = 0

    for G in graph_list:
        dgl_graph = dgl.from_networkx(G, node_attrs=['feat'])
        subgraphs.append(dgl_graph)
        # for graph classification, labels will be taken from the graph attribute
        if G.graph['label'] not in labels:
            labels[i] = G.graph['label']
        graph_ids.append(i)
        node_features_list.append(dgl_graph.ndata['feat'].numpy())
        i += 1

    create_csv_files(subgraphs, graph_ids, labels, node_features_list)

def remove_trailing_numbers(string):
    pattern = r'\d+$'
    result = re.sub(pattern, '', string)
    return result

def traverse_directory(path, module_in_file, hw):
    global current_file_num
    if os.path.isfile(path):
        new_graph = create_graph(path, hw)
        graph_list.append(new_graph)
        if hw == "GL":
            module_in_file = parse_file_information(path)
            verilog_to_gate(path)
        current_file_num += 1
    elif os.path.isdir(path):
        for file_name in os.listdir(path):
            file_path = os.path.join(path, file_name)
            if os.path.exists(file_path):
                traverse_directory(file_path, module_in_file, hw)

def create_graph(path, hw):
    new_graph = nx.DiGraph() if hw == "GL" else RTL_parser(path)
    path_parts = path.split("/")
    label = path_parts[-2] if len(path_parts) > 1 else path_parts[-1]
    new_graph.graph['label'] = label
    return new_graph

"""
Model Development and Training
"""

class SyntheticDataset(DGLDataset):
    def __init__(self, node_features_file, edges_file, properties_file):
        self.node_features_file = node_features_file
        self.edges_file = edges_file
        self.properties_file = properties_file
        super().__init__(name="synthetic")

    def process(self):
        # Load node features from the external file
        node_features_df = pd.read_csv(self.node_features_file)
        node_features_dict = {}
        for _, row in node_features_df.iterrows():
            graph_id = row["graph_id"]
            node_id = row["node_id"]
            features = row.drop(["graph_id", "node_id"]).to_numpy()
            if graph_id not in node_features_dict:
                node_features_dict[graph_id] = {}
            node_features_dict[graph_id][node_id] = features

        # Load graph properties from the properties file
        properties = pd.read_csv(self.properties_file)
        self.graphs = []
        self.labels = []
        self.node_features_list = []

        # Create a graph for each graph ID from the edges table.
        # First process the properties table into two dictionaries with graph IDs as keys.
        # The label and number of nodes are values.
        label_dict = {}
        num_nodes_dict = {}
        for _, row in properties.iterrows():
            label_dict[row["graph_id"]] = row["label"]
            num_nodes_dict[row["graph_id"]] = row["num_nodes"]

        # For the edges, first group the table by graph IDs.
        edges_group = pd.read_csv(self.edges_file).groupby("graph_id")

        # For each graph ID...
        for graph_id in edges_group.groups:
            # Find the edges as well as the number of nodes, its label, and node features.
            edges_of_id = edges_group.get_group(graph_id)
            src = edges_of_id["src"].to_numpy()
            dst = edges_of_id["dst"].to_numpy()
            num_nodes = num_nodes_dict[graph_id]
            label = label_dict[graph_id]  
            node_features = node_features_dict.get(graph_id, {})

            # Create a graph and add it to the list of graphs, labels, and node features.
            g = dgl.graph((src, dst), num_nodes=num_nodes)
            g = dgl.add_self_loop(g)
            self.graphs.append(g)
            self.labels.append(label)
            self.node_features_list.append(node_features)

        # Convert the label list to tensor for saving.
        # self.labels = torch.LongTensor(self.labels)
    def __getitem__(self, i):
        graph = self.graphs[i]
        label = self.labels[i]
        node_features_dict = self.node_features_list[i]

        #Initialize an array to store node features
        num_nodes = graph.number_of_nodes()
        all_node_features = torch.zeros((num_nodes, self.dim_nfeats), dtype=torch.float32)

        # Fill in the array with node features
        for node_id, features in node_features_dict.items():
            node_idx = int(node_id)  # Convert node_id to integer
            all_node_features[node_idx] = torch.tensor(features).float()

        # Add node features as node attributes
        graph.ndata['attr'] = all_node_features  # Use 'attr' as the key for node features

        return graph, label

    def __len__(self):
        return len(self.graphs)

    def calculate_degs(self):
        """
        Calculates the degree sequence for all graphs in the dataset.
        This method populates the `degs` attribute of the dataset with the maximum
        degree of nodes across all graphs, used for initializing PNA layers.
        """
        # Flatten degrees from all graphs to find the global max degree
        all_degs = []
        for g in self.graphs:
            degs = g.in_degrees().numpy()  # Assuming undirected or in-degrees for directed graphs
            all_degs.extend(degs)

        # Find unique degrees and their frequencies as required by PNA
        unique_degs, counts = np.unique(all_degs, return_counts=True)
        self.degs = torch.tensor(unique_degs, dtype=torch.float32)
        self.deg_counts = torch.tensor(counts, dtype=torch.float32)

        # The degs array can be directly used in the PNAConv layer initialization
        return self.degs, self.deg_counts

    @property
    def dim_nfeats(self):
        return len(next(iter(self.node_features_list[0].values())))

    @property
    def gclasses(self):
        return len(set(self.labels))

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        g.ndata["h"] = h
        return dgl.mean_nodes(g, "h")

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.batch_norm2 = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        x = F.relu(self.batch_norm1(self.linear1(x)))
        x = self.batch_norm2(self.linear2(x))
        return x

class GIN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GIN, self).__init__()
        self.mlp1 = MLP(in_feats, h_feats, h_feats)
        self.conv1 = GINConv(self.mlp1, aggregator_type='sum')
        self.mlp2 = MLP(h_feats, h_feats, num_classes)
        self.conv2 = GINConv(self.mlp2, aggregator_type='sum')

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        g.ndata["h"] = h
        return dgl.mean_nodes(g, "h")

class PNA(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, aggregators, scalers, delta):
        super(PNA, self).__init__()
        self.conv1 = PNAConv(in_dim, hidden_dim, aggregators=aggregators, scalers=scalers, delta=delta)
        self.conv2 = PNAConv(hidden_dim, hidden_dim, aggregators=aggregators, scalers=scalers, delta=delta)
        self.fc = nn.Linear(hidden_dim, n_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')
        out = self.fc(hg)
        return out

def main():
    parser = argparse.ArgumentParser(description="Tool to parse and analyze Verilog data.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parse_parser = subparsers.add_parser('parse', help='Parse command options')
    parse_parser.add_argument('-ver', '--verilog', required=True, help='Path to Verilog file')
    parse_parser.add_argument('-hw', '--hardware', required=True, choices=['GL', 'RTL'], help='Hardware type (GL for gate-level, RTL for register-transfer level)')
    parse_parser.add_argument('-lib', '--library', help='Path to library (required for GL hardware type)')

    # Optional parameters as flags
    parse_parser.add_argument('-id', '--input_degree', action='store_true', help='Include input degree feature')
    parse_parser.add_argument('-od', '--output_degree', action='store_true', help='Include output degree feature')
    parse_parser.add_argument('-iod', '--io_degree', action='store_true', help='Include I/O degree feature')
    parse_parser.add_argument('-mdi', '--min_dist_input', action='store_true', help='Include minimum distance to input feature')
    parse_parser.add_argument('-mdo', '--min_dist_output', action='store_true', help='Include minimum distance to output feature')
    parse_parser.add_argument('-pi', '--primary_input', action='store_true', help='Include primary input feature')
    parse_parser.add_argument('-po', '--primary_output', action='store_true', help='Include primary output feature')
    parse_parser.add_argument('-ki', '--key_string', type=str, help='Key string for security features')
    parse_parser.add_argument('-gt', '--gate_type', action='store_true', help='Include gate type feature (only applicable if hardware is GL)')

    # Setup for the 'graph' command
    graph_parser = subparsers.add_parser('graph', help='Graph operations')
    graph_parser.add_argument('-model', type=str, default='GCN', help='Model type (default: GCN)')
    graph_parser.add_argument('-hdim', type=int, required=True, help='Hidden dimensions')
    graph_parser.add_argument('-train', nargs=3, required=True, metavar=('NODE_FEATURES_FILE', 'EDGES_FILE', 'PROPERTIES_FILE'), help='Files for training: node features, edges, and properties')
    graph_parser.add_argument('-test', nargs=3, metavar=('NODE_FEATURES_FILE', 'EDGES_FILE', 'PROPERTIES_FILE'), help='Files for testing: node features, edges, and properties')
    graph_parser.add_argument('-val', nargs=3, metavar=('NODE_FEATURES_FILE', 'EDGES_FILE', 'PROPERTIES_FILE'), help='Files for validation: node features, edges, and properties')

    args = parser.parse_args()

    # Process the arguments further based on the command
    if args.command == 'parse':
        # Check if library path is provided when hardware type is GL
        if args.hardware == 'GL' and not args.library:
            print("Error: -lib/--library must be specified when -hw/--hardware is set to 'GL'")
            sys.exit(1)

        # Check if gate_type is used correctly
        if args.gate_type and args.hardware != 'GL':
            print("Error: -gate_type is only applicable when -hw/--hardware is set to 'GL'")
            sys.exit(1)
        handle_parse(args)

    elif args.command == 'graph':
        handle_graph(args)

def handle_parse(args):
    print(f"Parsing Verilog files with the following settings:")
    print(f"Hardware type: {args.hardware}")
    if args.library:
        print(f"Library path: {args.library}")
    print(f"Verilog file path: {args.verilog}")

    # Build flag vector
    global flag_vector
    flag_vector = []
    for flag in ['input_degree', 'output_degree', 'io_degree', 'min_dist_input', 'min_dist_output', 'primary_input', 'primary_output', 'gate_type']:
        if getattr(args, flag):
            flag_vector.append(flag)

    if args.key_string:
        flag_vector.append("ki")
        print(f"Key String: {args.key_string}")

    print(f"Node features: {flag_vector}")

    if (args.hardware == "GL"):
        if not os.path.exists(args.library):
            print("File not found")
            sys.exit(1)
        global parsed_library, module_in_file
        parsed_library = parse_library(args.library)

    module_in_file = {}

    traverse_directory(args.verilog, module_in_file, args.hardware)

    update_features(flag_vector, args.key_string)
    save_dataset()

    # for i in graph_list:
    #     print(graph_list.index(i), len(i.nodes()), i.graph['label'])
    
    print("Successfully saved dataset in files4training/")
    

def handle_graph(args):
    print(f"Processing graph data with the following settings:")
    print(f"Model Type: {args.model}, Hidden Dimensions: {args.hdim}")
    print("Training Files:", args.train)
    if args.test:
        print("Testing Files:", args.test)
    if args.val:
        print("Validation Files:", args.val)

    # Record the start time
    start_time = time.time()

    # Initialize DataLoader variables for clarity
    train_dataloader = val_dataloader = test_dataloader = None

    # Always load the training dataset
    train_dataset = SyntheticDataset(
        node_features_file=args.train[0],
        edges_file=args.train[1],
        properties_file=args.train[2]
    )
    num_examples = len(train_dataset)

    # Check different combinations of provided datasets
    if args.val and args.test:
        # Load both validation and test datasets
        val_dataset = SyntheticDataset(
            node_features_file=args.val[0],
            edges_file=args.val[1],
            properties_file=args.val[2]
        )
        test_dataset = SyntheticDataset(
            node_features_file=args.test[0],
            edges_file=args.test[1],
            properties_file=args.test[2]
        )
        
        # Use the entire training dataset
        train_dataloader = GraphDataLoader(train_dataset, batch_size=5, shuffle=True)
        val_dataloader = GraphDataLoader(val_dataset, batch_size=5, shuffle=True)
        test_dataloader = GraphDataLoader(test_dataset, batch_size=5, shuffle=True)
    elif args.val and not args.test:
        # Only the validation dataset is provided
        val_dataset = SyntheticDataset(
            node_features_file=args.val[0],
            edges_file=args.val[1],
            properties_file=args.val[2]
        )
        
        # Split the training dataset for test
        num_examples = len(train_dataset)
        indices = torch.randperm(num_examples)
        train_indices, test_indices = indices[:int(num_examples * 0.9)], indices[int(num_examples * 0.9):]
        
        train_dataloader = GraphDataLoader(train_dataset, batch_size=5, shuffle=True)
        val_dataloader = GraphDataLoader(val_dataset, batch_size=5, shuffle=True)
        test_dataloader = GraphDataLoader(train_dataset, sampler=SubsetRandomSampler(test_indices), batch_size=5, drop_last=False)
    elif not args.val and args.test:
        # Only the test dataset is provided
        test_dataset = SyntheticDataset(
            node_features_file=args.test[0],
            edges_file=args.test[1],
            properties_file=args.test[2]
        )
        
        # Split the training dataset for validation
        num_examples = len(train_dataset)
        indices = torch.randperm(num_examples)
        train_indices, val_indices = indices[:int(num_examples * 0.9)], indices[int(num_examples * 0.9):]
        
        train_dataloader = GraphDataLoader(train_dataset, batch_size=5, shuffle=True)
        val_dataloader = GraphDataLoader(train_dataset, sampler=SubsetRandomSampler(val_indices), batch_size=5, drop_last=False)
        test_dataloader = GraphDataLoader(test_dataset, batch_size=5, shuffle=True)
    else:
        # Neither validation nor test datasets are provided, perform a standard split
        num_examples = len(train_dataset)
        indices = torch.randperm(num_examples)
        num_train = int(num_examples * 0.7)
        num_val = int(num_examples * 0.1)
        train_indices, val_indices, test_indices = indices[:num_train], indices[num_train:num_train + num_val], indices[num_train + num_val:]

        train_dataloader = GraphDataLoader(train_dataset, sampler=SubsetRandomSampler(train_indices), batch_size=5, drop_last=False)
        val_dataloader = GraphDataLoader(train_dataset, sampler=SubsetRandomSampler(val_indices), batch_size=5, drop_last=False)
        test_dataloader = GraphDataLoader(train_dataset, sampler=SubsetRandomSampler(test_indices), batch_size=5, drop_last=False)
    
  
    # Model initialization
    if args.model == "GCN":
        model = GCN(train_dataset.dim_nfeats, args.hdim, train_dataset.gclasses)
    elif args.model == "GIN":
        model = GIN(train_dataset.dim_nfeats, args.hdim, train_dataset.gclasses)
    elif args.model == "PNA":
        # Assuming degs have been calculated for your dataset
        degs, _ = train_dataset.calculate_degs()
        delta = degs.float().log().mean().item()  # Computing delta as mean(log(degree+1)) over the dataset

        # Define default aggregators and scalers
        aggregators = ['mean', 'max', 'min', 'std']  # Example aggregators
        scalers = ['identity', 'amplification', 'attenuation']  # Example scalers

        # Initialize PNA model with the computed delta and your chosen configurations
        model = PNA(
            in_dim=train_dataset.dim_nfeats,
            hidden_dim=h_dim,
            n_classes=train_dataset.gclasses,
            aggregators=aggregators,
            scalers=scalers,
            delta=delta  # Make sure to include the delta parameter here
        )
    else:
        raise ValueError(f"Unsupported model type: {args.model}")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training and validation
    best_val_loss = float('inf')
    patience = 3  # Early stopping patience
    patience_counter = 0

    for epoch in range(30):  
        model.train()
        total_loss = 0
        for batched_graph, labels in train_dataloader:
            pred = model(batched_graph, batched_graph.ndata["attr"].float())
            loss = F.cross_entropy(pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch + 1}/{30}, Training Loss: {total_loss / len(train_dataloader):.4f}")

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batched_graph, labels in val_dataloader:
                pred = model(batched_graph, batched_graph.ndata["attr"].float())
                val_loss += F.cross_entropy(pred, labels).item()
                val_correct += (pred.argmax(1) == labels).sum().item()
                val_total += labels.size(0)
        
        val_loss /= len(val_dataloader)
        val_accuracy = val_correct / val_total
        print(f"Validation Epoch {epoch + 1}, Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2%}")

        # Early stopping based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save the model state if necessary
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered due to no improvement.")
                break

    num_correct = 0
    num_tests = 0
    all_preds = []
    all_labels = []

    with open("test_results.txt", "a") as output_file:
        for batched_graph, labels in test_dataloader:
            pred = model(batched_graph, batched_graph.ndata["attr"].float())
            # output_file.write(f"Predicted Labels: {pred.argmax(1)}\n")
            # output_file.write(f"Actual Labels: {labels}\n")
            all_preds.extend(pred.argmax(1).tolist())
            all_labels.extend(labels.tolist())
            num_correct += (pred.argmax(1) == labels).sum().item()
            num_tests += len(labels)

        output_file.write(f"Number of Correct Predictions: {num_correct}\n")
        output_file.write(f"Total Tests: {num_tests}\n")

        # Print confusion matrix
        conf_matrix = confusion_matrix(all_labels, all_preds)
        output_file.write("Confusion Matrix:\n")
        output_file.write(f"{conf_matrix}\n")

        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
        output_file.write(f"Precision: {precision:.2f}\n")
        output_file.write(f"Recall: {recall:.2f}\n")
        output_file.write(f"F1 Score: {f1:.2f}\n")
        output_file.write(f"Test Accuracy: {num_correct / num_tests:.2%}\n")

        end_time = time.time()

        # Calculate and print the elapsed time
        elapsed_time = end_time - start_time
        output_file.write(f"Elapsed Time: {elapsed_time:.2f} seconds\n")

        output_file.write(f"Number of Graphs: {num_examples}, Model: {args.model}\n")
        output_file.write(f"Feature Dimensions: {train_dataset.dim_nfeats}, Hidden Dimensions: {args.hdim}, Graph Classes: {train_dataset.gclasses}\n")
        output_file.write("================================\n\n")

if __name__ == "__main__":
    main()