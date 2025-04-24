'''
Author: Hana Selmani
Created on 30/05/2023
Last modified on 02/04/2025
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.nn import BatchNorm1d
import time
import argparse
from pyverilog.vparser.parser import parse
from pyverilog.dataflow.dataflow_analyzer import VerilogDataflowAnalyzer
from pyverilog.dataflow.optimizer import VerilogDataflowOptimizer
from pyverilog.dataflow.graphgen import VerilogGraphGenerator
import ast

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
class_type = ""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# bench file parser
def parse_bench(path):
    top = os.path.basename(path).replace('.bench', '')
    with open(path, 'r') as f:
        data = f.read()
    # Create  graph from bench file content
    G = bench2gates(data)
    path_parts = path.split(os.sep)  
    label = path_parts[-2] if len(path_parts) > 1 else path_parts[-1]
    G.graph['label'] = label
    return G

def bench2gates(bench):
    Dict_gates = {
        'xor': [0, 1, 0, 0, 0, 0, 0, 0],
        'XOR': [0, 1, 0, 0, 0, 0, 0, 0],
        'OR':  [0, 0, 1, 0, 0, 0, 0, 0],
        'or':  [0, 0, 1, 0, 0, 0, 0, 0],
        'XNOR':[0, 0, 0, 1, 0, 0, 0, 0],
        'xnor':[0, 0, 0, 1, 0, 0, 0, 0],
        'and': [0, 0, 0, 0, 1, 0, 0, 0],
        'AND': [0, 0, 0, 0, 1, 0, 0, 0],
        'nand':[0, 0, 0, 0, 0, 1, 0, 0],
        'NAND':[0, 0, 0, 0, 0, 1, 0, 0],
        'buf': [0, 0, 0, 0, 0, 0, 0, 1],
        'BUF': [0, 0, 0, 0, 0, 0, 0, 1],
        'not': [0, 0, 0, 0, 0, 0, 1, 0],
        'NOT': [0, 0, 0, 0, 0, 0, 1, 0],
        'nor': [1, 0, 0, 0, 0, 0, 0, 0],
        'NOR': [1, 0, 0, 0, 0, 0, 0, 0],
    }
    G = nx.DiGraph()
    ML_count = 0

    # regex for BUF and NOT gates
    regex_not = r"\s*([^\s=]+)\s*=\s*(BUF|NOT)\(([^)]+)\)"
    for output, function, net_str in re.findall(regex_not, bench, flags=re.I):
        input_net = net_str.strip()
        G.add_edge(input_net, output.strip())
        G.nodes[output.strip()]['gate'] = function.upper()
        G.nodes[output.strip()]['count'] = ML_count
        ML_count += 1

    # regex for multi-input gates (OR, XOR, AND, NAND, XNOR, NOR)
    regex_multi = r"([^\s=]+)\s*=\s*(OR|XOR|AND|NAND|XNOR|NOR)\(([^)]+)\)"
    for output, function, net_str in re.findall(regex_multi, bench, flags=re.I):
        nets = [net.strip() for net in net_str.split(',')]
        G.add_edges_from((net, output.strip()) for net in nets)
        G.nodes[output.strip()]['gate'] = function.upper()
        G.nodes[output.strip()]['count'] = ML_count
        ML_count += 1

    # set nodes without a gate to 'input'
    for n in G.nodes():
        if 'gate' not in G.nodes[n]:
            G.nodes[n]['gate'] = 'input'
    # initialize output flag for all nodes
    for n in G.nodes():
        G.nodes[n]['output'] = False

    # process OUTPUT declarations
    out_regex = r"OUTPUT\(([^)]+)\)\n"
    for net_str in re.findall(out_regex, bench, flags=re.I):
        nets = [net.strip() for net in net_str.split(',')]
        for net in nets:
            if net not in G:
                print("Output " + net + " is Float")
            else:
                G.nodes[net]['output'] = True
    return G

def read_txtfile(cell, label, feat, row, col):
    graphs_by_file = {}
    node_to_file = {}  # maps node_id -> file name

    # Process nodes from cell.txt, label.txt, and feat.txt
    with open(cell, "r") as cell_file, \
        open(label, "r") as label_file, \
        open(feat, "r") as feat_file:
        
        for cell_line, label_line, feat_line in zip(cell_file, label_file, feat_file):
            cell_line = cell_line.strip()
            label_line = label_line.strip()
            feat_line = feat_line.strip()
            
            # Example cell line: "1841 \multiplier_1/U57 from file Test_add_mul_16_bit_syn.v"
            parts = cell_line.split(" from file ")
            left_part = parts[0]  # "1841 \multiplier_1/U57"
            file_name = parts[1] if len(parts) > 1 else None
            
            tokens = left_part.split()
            node_id = int(tokens[0])            # Node ID
            node_info = " ".join(tokens[1:]) if len(tokens) > 1 else ""
            
            # Label as an integer
            label = int(label_line)
            
            features = [node_id] + [int(x) for x in feat_line.split()]
            
            # create a new graph for this file if needed
            if file_name not in graphs_by_file:
                graphs_by_file[file_name] = nx.Graph()
            
            graphs_by_file[file_name].add_node(node_id, 
                                                info=node_info, 
                                                source_file=file_name, 
                                                label=label, 
                                                feat=features)
            # Record file
            node_to_file[node_id] = file_name

    # process edges from row.txt and col.txt
    row_indices = []
    col_indices = []

    with open(row, "r") as row_file:
        for line in row_file:
            row_indices.append(int(line.strip()))
            
    with open(col, "r") as col_file:
        for line in col_file:
            col_indices.append(int(line.strip()))

    # only add an edge if both endpoints come from the same file
    for src, dst in zip(row_indices, col_indices):
        if src in node_to_file and dst in node_to_file:
            if node_to_file[src] == node_to_file[dst]:
                file_name = node_to_file[src]
                graphs_by_file[file_name].add_edge(src, dst)

    # convert the dictionary of graphs to a list.
    graph_list.extend(list(graphs_by_file.values()))


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

def create_csv_files(graphs, graph_ids, graph_labels, node_features_list):
    # Ensure the input lists have the same length
    if len(graphs) != len(graph_ids) or len(graphs) != len(node_features_list):
        raise ValueError("Input lists must have the same length.")

    os.makedirs("files4training", exist_ok=True)
    
    # Write header to graph_edges CSV file 
    edges_file = 'files4training/graph_edges.csv'
    fieldnames_edges = ['graph_id', 'src', 'dst']
    with open(edges_file, 'w', newline='') as csvfile:
        writer_edges = csv.DictWriter(csvfile, fieldnames=fieldnames_edges)
        writer_edges.writeheader()
        for graph, graph_id in zip(graphs, graph_ids):
            src, dst = graph.edges()
            for s, d in zip(src.numpy(), dst.numpy()):
                writer_edges.writerow({'graph_id': graph_id, 'src': s, 'dst': d})
    
    # Write header to graph_properties CSV file 
    properties_file = 'files4training/graph_properties.csv'
    if class_type == "graph":
        labels_set = list(set(labels.values()))
        label_mapping = {l: index for index, l in enumerate(labels_set)}
        fieldnames_properties = ['graph_id', 'label', 'num_nodes', 'label_string']
    else:
        fieldnames_properties = ['graph_id', 'num_nodes']
    with open(properties_file, 'w', newline='') as csvfile:
        writer_properties = csv.DictWriter(csvfile, fieldnames=fieldnames_properties)
        writer_properties.writeheader()
        for graph, graph_id in zip(graphs, graph_ids):
            num_nodes = graph.number_of_nodes()
            if class_type == "graph":
                label_val = labels[graph_id]  # Using the label from the dictionary
                writer_properties.writerow({'graph_id': graph_id, 'label': label_mapping[label_val], 'num_nodes': num_nodes, 'label_string': label_val})
            else:
                writer_properties.writerow({'graph_id': graph_id, 'num_nodes': num_nodes})

    node_features_file = 'files4training/node_features.csv'

    # Determine the number of features from the first node's feature vector.
    num_feats = len(list(node_features_list[0][0]))-1
    if class_type == "node":
        fieldnames_node_features = ['graph_id', 'node_id', 'label'] + [f'feat_{i}' for i in range(num_feats)]
    else:
        fieldnames_node_features = ['graph_id', 'node_id'] + [f'feat_{i}' for i in range(num_feats)]
    
    with open(node_features_file, 'w', newline='') as csvfile:
        writer_node_features = csv.DictWriter(csvfile, fieldnames=fieldnames_node_features)
        writer_node_features.writeheader()
    
        for graph, graph_id, features_dict in zip(graphs, graph_ids, node_features_list):
            for node_id in graph.nodes():
                feat = features_dict[node_id][1:]
                # Get the original node id (as written in the features file)
                orig_node_id = str(features_dict[node_id][0])
                if class_type == "node":
                    label_node = int(graph.ndata['label'][node_id].item())

                    writer_node_features.writerow(
                        {'graph_id': graph_id, 
                        'node_id': orig_node_id, 
                        'label': label_node,
                        **{f'feat_{i}': feat[i] for i in range(len(feat))}}
                    )
                else:
                    writer_node_features.writerow(
                        {'graph_id': graph_id, 
                        'node_id': orig_node_id, 
                        **{f'feat_{i}': feat[i] for i in range(len(feat))}}
                    )

def save_dataset():
    subgraphs = []
    labels = {}
    graph_ids = []
    node_features_list = []

    for i, G in enumerate(graph_list):
        dgl_graph = dgl.from_networkx(G, node_attrs=['feat','label'])
        subgraphs.append(dgl_graph)
        
        # For graph classification, assign the label from the graph attribute.
        if class_type == "graph":
            if G.graph['label'] not in labels:
                labels[i] = G.graph['label']
            
        graph_ids.append(i)
        node_features_list.append(dgl_graph.ndata['feat'].numpy())

    create_csv_files(subgraphs, graph_ids, labels, node_features_list)

def remove_trailing_numbers(string):
    pattern = r'\d+$'
    result = re.sub(pattern, '', string)
    return result

def traverse_directory(path, module_in_file, hw):
    global current_file_num
    if os.path.isfile(path):
        # Check file extension: use bench parser if file ends with '.bench'
        if path.lower().endswith('.bench'):
            new_graph = parse_bench(path)  
            graph_list.append(new_graph)
            current_file_num += 1
            return  
        else:
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
    parent_dir = os.path.basename(os.path.dirname(os.path.normpath(path)))
    label = parent_dir if parent_dir else os.path.basename(path)
    new_graph.graph['label'] = label
    return new_graph

"""
Model Development and Training
"""

class GraphDataset(DGLDataset):
    def __init__(self, node_features_file, edges_file, properties_file):
        self.node_features_file = node_features_file
        self.edges_file = edges_file
        self.properties_file = properties_file
        super().__init__(name="graph_classification")

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

class NodeClassificationDataset(DGLDataset):
    def __init__(self, node_features_file, edges_file, properties_file):
        self.node_features_file = node_features_file
        self.edges_file = edges_file
        self.properties_file = properties_file
        super().__init__(name="node_classification")

    def process(self):
        df_nodes = pd.read_csv(self.node_features_file)
        nan_cols = [col for col in df_nodes.columns if df_nodes[col].isna().all()]
        if nan_cols:
            print("Columns with all NaN values:", nan_cols)
            df_nodes = df_nodes.drop(columns=nan_cols)
        # Build a dictionary: graph_id -> { global_node_id: {"features": ..., "label": ..., "split": ...} }
        node_data = {}
        for _, row in df_nodes.iterrows():
            graph_id = row["graph_id"]
            node_id = row["node_id"]
            label = row["label"]
            # Extract all columns starting with 'feat_'
            feature_cols = [col for col in df_nodes.columns if col.startswith("feat_")]
            features = row[feature_cols].to_numpy()
            if graph_id not in node_data:
                node_data[graph_id] = {}
            # Convert node_id to string for consistent key mapping
            node_data[graph_id][str(node_id)] = {"features": features, "label": label}

        # Load graph properties to know the expected number of nodes
        df_props = pd.read_csv(self.properties_file)
        # Create a mapping: graph_id -> num_nodes 
        num_nodes_dict = {}
        for _, row in df_props.iterrows():
            num_nodes_dict[row["graph_id"]] = row["num_nodes"]

        # Load edges CSV and group by graph_id.
        df_edges = pd.read_csv(self.edges_file)
        edges_group = df_edges.groupby("graph_id")

        self.graphs = []
        self.graph_ids = []  # To keep track of which graph is which

        # For each graph id in properties, create the DGL graph
        for graph_id in num_nodes_dict.keys():
            if graph_id in node_data:
                # Sort global node IDs (as strings) by their integer value 
                global_node_ids = sorted(list(node_data[graph_id].keys()), key=lambda x: int(float(x)))
            else:
                global_node_ids = []

            # Create a mapping from global node ID to a local sequential index
            local_mapping = {global_id: i for i, global_id in enumerate(global_node_ids)}
            num_nodes_local = len(global_node_ids)  

            # Remap edges for this graph using the local mapping
            if graph_id in edges_group.groups:
                group = edges_group.get_group(graph_id)
                src_global = group["src"].to_numpy()
                dst_global = group["dst"].to_numpy()
                src_local = []
                dst_local = []
                for s, d in zip(src_global, dst_global):
                    s_str = str(s)
                    d_str = str(d)
                    if s_str in local_mapping and d_str in local_mapping:
                        src_local.append(local_mapping[s_str])
                        dst_local.append(local_mapping[d_str])
                src_local = np.array(src_local)
                dst_local = np.array(dst_local)
            else:
                src_local = np.array([])
                dst_local = np.array([])

            # Create the DGL graph using the local number of nodes
            g = dgl.graph((src_local, dst_local), num_nodes=int(num_nodes_local))
            g = dgl.add_self_loop(g)

            # Initialize containers for node features, labels
            node_feats = torch.zeros((int(num_nodes_local), self.dim_nfeats), dtype=torch.float32)
            node_labels = torch.zeros((int(num_nodes_local),), dtype=torch.long)

            if graph_id in node_data:
                for global_node_id, data in node_data[graph_id].items():
                    if global_node_id in local_mapping:
                        local_index = local_mapping[global_node_id]
                        # Convert features to float32.
                        node_feats[local_index] = torch.tensor(np.array(data["features"], dtype=np.float32))
                        node_labels[local_index] = int(data["label"])

            g.ndata["feat"] = node_feats
            g.ndata["label"] = node_labels
            # Create masks based on the split flag.
            # train_mask = torch.tensor([s == "tr" for s in node_splits], dtype=torch.bool)
            # val_mask = torch.tensor([s == "va" for s in node_splits], dtype=torch.bool)
            # test_mask = torch.tensor([s == "te" for s in node_splits], dtype=torch.bool)
            # g.ndata["train_mask"] = train_mask
            # g.ndata["val_mask"] = val_mask
            # g.ndata["test_mask"] = test_mask

            self.graphs.append(g)
            self.graph_ids.append(graph_id)

    def __getitem__(self, i):
        return self.graphs[i]

    def __len__(self):
        return len(self.graphs)

    @property
    def dim_nfeats(self):
        df = pd.read_csv(self.node_features_file, nrows=1)
        feature_cols = [col for col in df.columns if col.startswith("feat_")]
        return len(feature_cols)

    @property
    def n_classes(self):
        return len(set(label.item() for g in self.graphs for label in g.ndata["label"]))

    
    def calculate_degs(self):

        # Calculates the degree sequence for all graphs in the dataset.
        # This method populates self.degs and self.deg_counts used for PNA layer initialization.
        all_degs = []
        for g in self.graphs:
            # Using in-degrees for undirected graphs.
            degs = g.in_degrees().numpy()
            all_degs.extend(degs)
        unique_degs, counts = np.unique(all_degs, return_counts=True)
        self.degs = torch.tensor(unique_degs, dtype=torch.float32)
        self.deg_counts = torch.tensor(counts, dtype=torch.float32)
        return self.degs, self.deg_counts


class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, n_layers, mode="node"):
        """
        n_layers: total number of GraphConv layers.
                  If n_layers==1, maps directly from in_feats to num_classes.
                  If n_layers>=2, first layer: in_feats -> h_feats,
                  intermediate layers (if any): h_feats -> h_feats,
                  final layer: h_feats -> num_classes.
        """
        super(GCN, self).__init__()
        self.mode = mode  # 'graph' or 'node'
        self.layers = nn.ModuleList()
        if n_layers == 1:
            self.layers.append(GraphConv(in_feats, num_classes))
        else:
            # First layer: in_feats -> h_feats
            self.layers.append(GraphConv(in_feats, h_feats))
            # Intermediate layers: h_feats -> h_feats
            for _ in range(n_layers - 2):
                self.layers.append(GraphConv(h_feats, h_feats))
            # Final layer: h_feats -> num_classes
            self.layers.append(GraphConv(h_feats, num_classes))

    def forward(self, g, in_feat):
        h = in_feat
        for i, conv in enumerate(self.layers):
            h = conv(g, h)
            # Apply ReLU after every layer except the final one.
            if i != len(self.layers) - 1:
                h = F.relu(h)
        g.ndata["h"] = h
        if self.mode == "graph":
            return dgl.max_nodes(g, "h")
        else:
            return h  # return node-level logits

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
    def __init__(self, in_feats, h_feats, num_classes, n_layers, mode="node"):
        super(GIN, self).__init__()
        self.mode = mode  # 'graph' or 'node'
        self.layers = nn.ModuleList()
        if n_layers == 1:
            mlp = MLP(in_feats, h_feats, num_classes)
            self.layers.append(GINConv(mlp, aggregator_type='sum'))
        else:
            # First layer: in_feats -> h_feats
            mlp1 = MLP(in_feats, h_feats, h_feats)
            self.layers.append(GINConv(mlp1, aggregator_type='sum'))
            # Intermediate layers: h_feats -> h_feats
            for _ in range(n_layers - 2):
                mlp_mid = MLP(h_feats, h_feats, h_feats)
                self.layers.append(GINConv(mlp_mid, aggregator_type='sum'))
            # Final layer: h_feats -> num_classes
            mlp_final = MLP(h_feats, h_feats, num_classes)
            self.layers.append(GINConv(mlp_final, aggregator_type='sum'))

    def forward(self, g, in_feat):
        h = in_feat
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
            if i != len(self.layers) - 1:
                h = F.relu(h)
        g.ndata["h"] = h
        if self.mode == "graph":
            return dgl.max_nodes(g, "h")
        else:
            return h  # node-level logits


class PNA(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, aggregators, scalers, delta, n_layers, mode="node"):
        super(PNA, self).__init__()
        self.mode = mode  # "graph" or "node"
        self.layers = nn.ModuleList()
        if n_layers == 1:
            self.layers.append(dgl.nn.PNAConv(in_dim, n_classes, aggregators=aggregators, scalers=scalers, delta=delta))
        else:
            # First layer: in_dim -> hidden_dim
            self.layers.append(dgl.nn.PNAConv(in_dim, hidden_dim, aggregators=aggregators, scalers=scalers, delta=delta))
            # Intermediate layers: hidden_dim -> hidden_dim
            for _ in range(n_layers - 2):
                self.layers.append(dgl.nn.PNAConv(hidden_dim, hidden_dim, aggregators=aggregators, scalers=scalers, delta=delta))
            # Final layer: hidden_dim -> n_classes
            self.layers.append(dgl.nn.PNAConv(hidden_dim, hidden_dim, aggregators=aggregators, scalers=scalers, delta=delta))
            self.fc = nn.Linear(hidden_dim, n_classes)

    def forward(self, g, in_feat):
        h = in_feat
        for i, conv in enumerate(self.layers):
            h = conv(g, h)
            if i != len(self.layers) - 1:
                h = F.relu(h)
        g.ndata["h"] = h
        if self.mode == "graph":
            hg = dgl.max_nodes(g, "h")
            out = self.fc(hg)
            return out
        else:
            out = self.fc(h)
            return out  # returns node-level logits

def main():
    parser = argparse.ArgumentParser(description="Tool to parse and analyze Verilog data.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parse_parser = subparsers.add_parser('parse', help='Parse command options')
    parse_parser.add_argument('-ver', '--verilog', required=True, help='Path to Verilog file')
    parse_parser.add_argument('-hw', '--hardware', required=True, choices=['GL', 'RTL', 'BENCH', 'TXT'], help='Hardware type (GL for gate-level, RTL for register-transfer level), Bench')
    parse_parser.add_argument('-class', type=str, required=True, choices=['graph', 'node'], help='Classification type: "graph" classification or "node" classification')
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

    # args for training portion
    graph_parser = subparsers.add_parser('train', help='Graph operations')
    graph_parser.add_argument('-class', '--classification', type=str, required=True, choices=['graph', 'node'], help='Classification type: "graph" classification or "node" classification')
    graph_parser.add_argument('-model', type=str, default='GCN', help='Model type. Default: GCN')
    graph_parser.add_argument('-hdim', type=int, default=64, help='Hidden dimensions. Default: 64')
    graph_parser.add_argument('-n_layers', type=int, default=2, help='Number of layers for the GNN model. Default: 2')
    graph_parser.add_argument('-batch_size', type=int, default=5, help='Batch size. Default: 5')
    graph_parser.add_argument('-lr', type=float, default=0.01, help='Learning rate. Default: 0.01')
    graph_parser.add_argument('-epochs', type=int, default=300, help='Number of epochs. Default: 300')
    graph_parser.add_argument('-input', type=str, default="files4training", help='Path to Files for training: node features, edges, and properties')
    graph_parser.add_argument('-test', type=str, help='Path to Files for testing: node features, edges, and properties')
    graph_parser.add_argument('-val', type=str, help='Path to Files for validation: node features, edges, and properties')
    graph_parser.add_argument('-output', type=str, default='gnn4circuits_results.txt', help='Results output txt file')

    # args for parsing from text files
    txt_parser = subparsers.add_parser('parse_txt', help='Parse extracted graphs from .txt files')
    txt_parser.add_argument('-path', required=True, help='Path to txt file')
    

    args = parser.parse_args()

    # Process the arguments further based on the command
    if args.command == 'parse':
        class_type = args.classification
        # Check if library path is provided when hardware type is GL
        if args.hardware == 'GL' and not args.library:
            print("Error: -lib/--library must be specified when -hw/--hardware is set to 'GL'")
            sys.exit(1)

        # Check if gate_type is used correctly
        if args.gate_type and args.hardware != 'GL':
            print("Error: -gate_type is only applicable when -hw/--hardware is set to 'GL'")
            sys.exit(1)
        handle_parse(args)

    elif args.command == 'train':
        class_type = args.classification
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        handle_training(args)

    elif args.command == 'parse_txt':
        cell = args.path + "/cell.txt"
        label = args.path + "/label.txt"
        feat = args.path + "/feat.txt"
        row = args.path + "/row.txt"
        col = args.path + "/col.txt"
        print(f"Parsing text files: \n{cell}\n{label}\n{feat}\n{row}\n{col}\n")

        read_txtfile(cell, label, feat, row, col)
        save_dataset()

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

    for i in graph_list:
        print(graph_list.index(i), len(i.nodes()), i.graph['label'])
    
    print("Successfully saved dataset in files4training/")


def train_epoch(model, optimizer, dataloader, classification_type):
    model.train()
    total_loss = 0.0
    num_batches = 0
    all_preds = []
    all_labels = []

    for batch in dataloader:
        optimizer.zero_grad()
        if classification_type == "graph":
            graphs, labels = batch
            graphs = graphs.to(device)
            labels = labels.to(device)
            feats = graphs.ndata["attr"].float().to(device)
            logits = model(graphs, feats)
            loss = F.cross_entropy(logits, labels)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

        elif classification_type == "node":
            g = batch.to(device)
            mask = g.ndata["train_mask"]
            if mask.sum().item() == 0:
                continue
            logits = model(g, g.ndata["feat"].float().to(device))
            labels = g.ndata["label"][mask].to(device)
            preds = logits.argmax(dim=1)[mask]
            loss = F.cross_entropy(logits[mask], labels)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
        else:
            raise ValueError("Unknown classification type")

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1

    if not all_preds:
        return float('nan'), (0, 0, 0, 0, None)

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    avg_loss = total_loss / num_batches if num_batches > 0 else float('nan')

    return avg_loss, (acc, f1)


def validate_epoch(model, dataloader, classification_type):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for batch in dataloader:
            if classification_type == "graph":
                graphs, labels = batch
                graphs = graphs.to(device)
                labels = labels.to(device)
                feats = graphs.ndata["attr"].float().to(device)
                logits = model(graphs, feats)
                loss = F.cross_entropy(logits, labels)
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.tolist())
                total_loss += loss.item()
                num_batches += 1
            elif classification_type == "node":
                g = batch.to(device)
                mask = g.ndata["val_mask"]
                if mask.sum().item() == 0:
                    continue          
                logits = model(g, g.ndata["feat"].float().to(device))
                labels = g.ndata["label"][mask].to(device)
                loss = F.cross_entropy(logits[mask], g.ndata["label"][mask])
                preds = logits.argmax(dim=1)[mask]
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.tolist())
                total_loss += loss.item()
                num_batches += 1
    if not all_preds:
        return float('nan'), (0, 0, 0, None)

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    avg_loss = total_loss / num_batches if num_batches > 0 else float('nan')
    return avg_loss, (accuracy, f1)

def test_epoch(model, dataloader, classification_type):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            if classification_type == "graph":
                graphs, labels = batch
                graphs = graphs.to(device)
                labels = labels.to(device)
                feats = graphs.ndata["attr"].float().to(device)
                logits = model(graphs, feats)
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.tolist())
            elif classification_type == "node":
                g = batch.to(device)
                mask = g.ndata["test_mask"]
                if mask.sum().item() == 0:
                    continue
                logits = model(g, g.ndata["feat"].float().to(device))
                labels = g.ndata["label"][mask].to(device)
                preds = logits.argmax(dim=1)[mask]
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.tolist())

    if not all_preds:
        return 0, 0, 0, 0, None

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    return acc, prec, recall, f1, cm

def handle_training(args):
    classification_type = args.classification  # "graph" or "node"

    print("Loading dataset...")

    if classification_type == "graph":
        train_dataset = GraphDataset(
            node_features_file=args.input + "/node_features.csv",
            edges_file=args.input + "/graph_edges.csv",
            properties_file=args.input +"/graph_properties.csv"
        )
        train_dataset.process()
        val_dataset = train_dataset  
        train_dataloader = GraphDataLoader(train_dataset, batch_size=args.batch, shuffle=True)
        val_dataloader = GraphDataLoader(val_dataset, batch_size=args.batch, shuffle=False)

        if args.model == "GCN":
            model = GCN(train_dataset.dim_nfeats, args.hdim, train_dataset.gclasses,
                                args.n_layers, mode="graph")
        elif args.model == "GIN":
            model = GIN(train_dataset.dim_nfeats, args.hdim, train_dataset.gclasses,
                                args.n_layers, mode="graph")
        elif args.model == "PNA":
            degs, _ = train_dataset.calculate_degs()
            delta = degs.float().log().mean().item()
            aggregators = ['mean', 'max', 'min', 'std']
            scalers = ['identity', 'amplification', 'attenuation']
            model = PNA(train_dataset.dim_nfeats, args.hdim, train_dataset.gclasses,
                                aggregators, scalers, delta, args.n_layers, mode="graph")
        else:
            raise ValueError(f"Unsupported model type: {args.model}")


    elif classification_type == "node":
        dataset = NodeClassificationDataset(
            node_features_file=args.input + "/node_features.csv",
            edges_file=args.input + "/graph_edges.csv",
            properties_file=args.input +"/graph_properties.csv"
        )
        dataset.process()
        train_dataset = dataset
        val_dataset = dataset
        test_dataset = dataset

        train_dataloader = GraphDataLoader(train_dataset, batch_size=args.batch, shuffle=True)
        val_dataloader = GraphDataLoader(val_dataset, batch_size=args.batch, shuffle=False)
        test_dataloader = GraphDataLoader(test_dataset, batch_size=args.batch, shuffle=False)
        print("Done loading dataloaders.")

        in_feats = dataset.dim_nfeats
        num_classes = dataset.n_classes

        if args.model == "GCN":
            model = MultiLabelGCN(in_feats, args.hdim, num_classes, args.n_layers, 0.3)
        elif args.model == "GIN":
            model = MultiLabelGIN(in_feats, args.hdim, num_classes, args.n_layers, 0.3)
        elif args.model == "PNA":
            degs, _ = train_dataset.calculate_degs()
            delta = degs.float().log().mean().item()
            aggregators = ['mean', 'max', 'min', 'std']
            scalers = ['identity', 'amplification', 'attenuation']
            model = MultiLabelPNA(in_feats, args.hdim, num_classes, aggregators, scalers, delta, args.n_layers, 0.3, "node")
        else:
            raise ValueError(f"Unsupported model type: {args.model}")

    else:
        raise ValueError("Classification type must be either 'graph' or 'node'.")

    model = model.to(device)

    lr = args.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_f1 = -1
    best_epoch = -1
    num_epochs = args.epochs
    test_acc = test_prec = test_recall = test_f1 = test_cm = None

    start_time = time.time()

    for epoch in range(num_epochs):
        train_loss, (train_acc, train_f1) = train_epoch(model, optimizer, train_dataloader, classification_type)
        print(f"[Train Epoch {epoch+1}] Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")

        val_loss, (val_acc, val_f1) = validate_epoch(
            model, val_dataloader, classification_type)
        print(f"[Val   Epoch {epoch+1}] Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")
        if val_f1 >= best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            test_acc, test_prec, test_recall, test_f1, test_cm = test_epoch(model, test_dataloader, classification_type)
            print(f"\n[TEST RESULTS] Acc: {test_acc:.4f} | Prec: {test_prec:.4f} | Recall: {test_recall:.4f} | F1: {test_f1:.4f}")


    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\nTotal elapsed time: {elapsed:.2f} seconds")

    # Logging
    with open(args.output, "a") as output_file:
        output_file.write(f"\n\nModel: {args.model}\n")
        # output_file.write(f"h_dim: {args.hdim}\n")
        # output_file.write(f"epochs: {args.epochs}\n")
        # output_file.write(f"num layers: {args.n_layers}\n")
        # output_file.write(f"best epoch: {best_epoch}\n")
        output_file.write(f"Test Accuracy: {test_acc:.4f}, Precision: {test_prec:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}\n")
        output_file.write("Confusion Matrix:\n" + str(test_cm) + "\n")
        output_file.write(f"Elapsed Time: {elapsed:.2f} seconds\n")
    
if __name__ == "__main__":
    main()
