import numpy as np
import math
import copy

'''
This file collects classes and methods useful to build and deal with the Clustering Feature Trees mentioned in the paper.
See the paper "BIRCH: An Efficient Data Clustering Method for Very Large Databases" for further explications.
'''

# Clustering Feature
class CF:
    def __init__(self, N=0, LS=None, SS=None):
        self.N = N # number of datapoints represented by the CL
        self.LS = LS # linear sum of the N datapoints
        self.SS = SS # squared sum of the N datapoints
        

    def add_datapoint(self, datapoint): # datapoint is a numpy.ndarray of length num_features
        if self.LS is None: # the CF was empty, no datapoint was present
            self.LS = copy.deepcopy(datapoint)
            self.SS = np.sum(datapoint ** 2)
        else:
            self.LS += datapoint
            self.SS += np.sum(datapoint ** 2)
        self.N += 1


    def merge_CF(self, other_CF):
        copy_other_CF = copy.deepcopy(other_CF) # withouth this, references are kept with consequent problems
        if self.LS is None: # the CF was empty, no datapoint was present
            self.N = copy_other_CF.N
            self.LS = copy_other_CF.LS
            self.SS = copy_other_CF.SS
        else:
            self.N += copy_other_CF.N
            self.LS += copy_other_CF.LS
            self.SS += copy_other_CF.SS
            


# Actual set storing datapoints
class Subcluster:
    def __init__(self, threshold_T):
        self.CF = CF()
        self.datapoints = [] # will contain the set of datapoints
        self.diameter = None
        self.threshold_T = threshold_T
        self.parent = None
    

    def add_datapoint(self, datapoint): # datapoint is a numpy.ndarray of length num_features
        if self.diameter == None: # the Subcluster was empty
            self.CF.add_datapoint(datapoint)
            self.datapoints.append(datapoint)
            self.diameter = 0
        else:
            CF_tmp = copy.deepcopy(self.CF)
            CF_tmp.add_datapoint(datapoint)

            # Compute the new possible value of the diameter...
            num = 2 * CF_tmp.N * CF_tmp.SS - 2 * np.sum(CF_tmp.LS ** 2)
            den = CF_tmp.N * (CF_tmp.N - 1)
            diameter_tmp = math.sqrt(num / den)
            
            # ...and see if it is still ok
            if diameter_tmp < self.threshold_T:
                self.CF = copy.deepcopy(CF_tmp)
                self.datapoints.append(datapoint)
                self.diameter = diameter_tmp
                return True # insertion was successful
            else: # if adding the new datapoint would mean to get a value of a diameter equal or greater than threshold_T
                return False # insertion was not successful



# Node of the Clustering Feature tree
class CFNode:
    def __init__(self, is_leafnode, max_num_entries_leafnode_L, branching_factor_B):
        self.is_leafnode = is_leafnode
        self.CF = CF()
        self.CF_list = [] # made of elements in the form [CF, Subcluster] (for leafnode) or [CF, CFNode] (for non-leafnode). Be aware that the two elements (CF and Subcluster/CFNode) are independent
        self.parent = None

        if self.is_leafnode:
            self.max_num_entries_leafnode_L = max_num_entries_leafnode_L
            # Be aware that the following pointers have been added to improve the scanning process. The order of the leafnodes in the
                # chain of pointers is "random"
            self.pointer_next = None # pointer to the next leafnode
            self.pointer_prev = None # pointer to the previous leafnode

        else:
            self.branching_factor_B = branching_factor_B


    def get_depth(self) -> int: # depth = 0 is only for the root
        depth = 0
        node = self
        while node.parent is not None:
            depth+=1
            node = node.parent

        return depth



# Clustering Feature Tree
class CFTree:
    def __init__(self, branching_factor_B, threshold_T, max_num_entries_leafnode_L, distance_metric):
        self.root = CFNode(is_leafnode=True, max_num_entries_leafnode_L = max_num_entries_leafnode_L,
                           branching_factor_B = branching_factor_B)
        
        self.distance_metric = distance_metric
        self.branching_factor_B = branching_factor_B
        self.max_num_entries_leafnode_L = max_num_entries_leafnode_L
        self.threshold_T = threshold_T


    def insert(self, datapoint): # datapoint is a numpy.ndarray
        CF_datapoint = CF()
        CF_datapoint.add_datapoint(datapoint)

        current_node = self.root

        # If the tree is empty
        if not self.root.CF_list:
            new_subcluster = Subcluster(self.threshold_T)
            new_subcluster.add_datapoint(datapoint)
            current_node.CF_list.append([copy.deepcopy(new_subcluster.CF), new_subcluster])
            current_node.CF.add_datapoint(datapoint)
            new_subcluster.parent = self.root
            return
        
        # Here the tree is not empty. Descend the tree until the most suitable leafnode is found 
        while not current_node.is_leafnode:
            best_child = self.choose_childnode(current_node, CF_datapoint)
            current_node = best_child
        
        # At this point current_node is the leafnode CFNode where, somewhere, we would like to insert the datapoint

        # Find the most suitable Subcluster among those in the selected leafnode current_node
        min_distance = self.distance_metric(CF_1 = CF_datapoint, CF_2 = current_node.CF_list[0][0])
        current_subcluster_choice = 0 # current_subcluster_choice is a number identifying the Subcluster in the leafnode
        for subcluster_i in range(1, len(current_node.CF_list)):
            CFi = current_node.CF_list[subcluster_i][0]
            min_distance_tmp = self.distance_metric(CF_1 = CF_datapoint, CF_2 = CFi)
            if min_distance_tmp < min_distance:
                min_distance = min_distance_tmp
                current_subcluster_choice = subcluster_i

        # Try to insert the new datapoint in the subcluster       
        insertion_successful = current_node.CF_list[current_subcluster_choice][1].add_datapoint(datapoint) # current_node is still a leafnode

        if insertion_successful or len(current_node.CF_list) < self.max_num_entries_leafnode_L:
            if insertion_successful: # update the CF in the CF_list
                current_node.CF_list[current_subcluster_choice][0].add_datapoint(datapoint)

            # Insertion failed but there is enough space in the leafnode (i.e., len(current_node.CF_list) < self.max_num_entries_leafnode_L)
            if not insertion_successful and len(current_node.CF_list) < self.max_num_entries_leafnode_L:
                new_subcluster = Subcluster(self.threshold_T)
                new_subcluster.add_datapoint(datapoint)
                current_node.CF_list.append([copy.deepcopy(new_subcluster.CF), new_subcluster])
                new_subcluster.parent = current_node

            # Either with previous if or with insertion successful, we need to update the CFs at all levels
            while current_node is not self.root:  
                current_node.CF.add_datapoint(datapoint)
                current_node_parent = current_node.parent
                index_child = self.get_index_child(current_node, current_node_parent)
                current_node_parent.CF_list[index_child][0] = copy.deepcopy(current_node.CF)
                current_node = current_node.parent

            # Here current_node is the root
            current_node.CF.add_datapoint(datapoint)
            
        else: # insertion failed and there is no space in the leafnode for another subcluster
            new_subcluster = Subcluster(self.threshold_T)
            new_subcluster.add_datapoint(datapoint)
            current_node.CF_list.append([copy.deepcopy(new_subcluster.CF), new_subcluster]) 

            # Split the old leafnode into two new leafnodes and recursively update
            new_leafnode_1, new_leafnode_2 = self.split_node(node = current_node, are_leaves = True)
            self.recursive_update(current_node, new_leafnode_1, new_leafnode_2)


    # SIDE METHOD to computer the best childnode, to descend the tree
    def choose_childnode(self, current_node, CF_datapoint):
        min_distance = self.distance_metric(CF_1 = CF_datapoint, CF_2 = current_node.CF_list[0][0])
        current_child_choice = current_node.CF_list[0][1] # pointer to the child node

        # Find the closest child in terms of distance between CFs
        for child_i in range(1, len(current_node.CF_list)):
            CFi = current_node.CF_list[child_i][0]
            min_distance_tmp = self.distance_metric(CF_1 = CF_datapoint, CF_2 = CFi)

            if min_distance_tmp < min_distance:
                min_distance = min_distance_tmp
                current_child_choice = current_node.CF_list[child_i][1]

        return current_child_choice


    # SIDE METHOD to get the index of a child in the parent's list of children
    def get_index_child(self, current_node, current_node_parent):
        for index in range(len(current_node_parent.CF_list)):
            if current_node_parent.CF_list[index][1] == current_node:
                return index


    # SIDE METHOD to split a node
    def split_node(self, node, are_leaves):

        # Search the greatest distance among the objects, either subclusters or nodes
        max_distance = None
        first = None
        second = None

        for i in range(len(node.CF_list)):
            CF_i = node.CF_list[i][0]
            for j in range(i + 1, len(node.CF_list)): # distance is symmetric; do not consider distance bewteen same objects
                CF_j = node.CF_list[j][0]
                max_distance_tmp = self.distance_metric(CF_1 = CF_i, CF_2 = CF_j)
                if max_distance is None or max_distance_tmp > max_distance:
                    max_distance = max_distance_tmp
                    first = i
                    second = j

        # Initialise new nodes
        if are_leaves:
            new_node_1 = CFNode(is_leafnode = True, max_num_entries_leafnode_L = self.max_num_entries_leafnode_L,
                                branching_factor_B = self.branching_factor_B)

            new_node_2 = CFNode(is_leafnode = True, max_num_entries_leafnode_L = self.max_num_entries_leafnode_L,
                                branching_factor_B = self.branching_factor_B)

            # pointer_next and pointer_prev are handled. Be aware that then nodes (leafnodes and internal nodes) will be rearranged with splits.
            # I just need the pointers to give me a faster scan, not to have the "order" of the leafnodes
            prev_node = node.pointer_prev
            next_node = node.pointer_next

            new_node_1.pointer_prev = prev_node
            if prev_node is not None:
                prev_node.pointer_next = new_node_1

            new_node_1.pointer_next = new_node_2
            new_node_2.pointer_prev = new_node_1

            new_node_2.pointer_next = next_node
            if next_node is not None:
                next_node.pointer_prev = new_node_2            
        
        else:
            new_node_1 = CFNode(is_leafnode = False, max_num_entries_leafnode_L = self.max_num_entries_leafnode_L,
                                branching_factor_B = self.branching_factor_B)
            new_node_2 = CFNode(is_leafnode = False, max_num_entries_leafnode_L = self.max_num_entries_leafnode_L,
                                branching_factor_B = self.branching_factor_B)
        
        new_node_1.CF_list.append(node.CF_list[first])
        new_node_1.CF.merge_CF(new_node_1.CF_list[0][0])
        new_node_1.CF_list[0][1].parent = new_node_1

        new_node_2.CF_list.append(node.CF_list[second])
        new_node_2.CF.merge_CF(new_node_2.CF_list[0][0])
        new_node_2.CF_list[0][1].parent = new_node_2

        if first < second:
            del node.CF_list[first]
            del node.CF_list[second - 1]  # index adjusted after first deletion
        else:
            del node.CF_list[second]
            del node.CF_list[first - 1]

        # Redistribute the objects among the two nodes
        for object_i in range(len(node.CF_list)): # depending on the value of are_leaves, the object is either a Subcluster or a CFNode
            CF_subcluster = node.CF_list[object_i][0]
            CF_seed_1 = new_node_1.CF_list[0][0]
            CF_seed_2 = new_node_2.CF_list[0][0]

            distance_to_1 = self.distance_metric(CF_1 = CF_subcluster, CF_2 = CF_seed_1)
            distance_to_2 = self.distance_metric(CF_1 = CF_subcluster, CF_2 = CF_seed_2)
            if distance_to_1 > distance_to_2:
                new_node_2.CF_list.append(node.CF_list[object_i])
                new_node_2.CF.merge_CF(node.CF_list[object_i][0])
                new_node_2.CF_list[-1][1].parent = new_node_2
            else:
                new_node_1.CF_list.append(node.CF_list[object_i])
                new_node_1.CF.merge_CF(node.CF_list[object_i][0])
                new_node_1.CF_list[-1][1].parent = new_node_1
            
        return new_node_1, new_node_2
            

    # SIDE METHOD to recursively update the tree from the bottom to the top
    def recursive_update(self, current_node, new_node_1, new_node_2):
        if current_node is self.root:

            # If the current node is the root, create a new root
            new_root = CFNode(is_leafnode=False, max_num_entries_leafnode_L=self.max_num_entries_leafnode_L,
                            branching_factor_B=self.branching_factor_B)
            new_root.CF_list.append([copy.deepcopy(new_node_1.CF), new_node_1])
            new_root.CF.merge_CF(new_root.CF_list[0][0])
            new_root.CF_list.append([copy.deepcopy(new_node_2.CF), new_node_2])
            new_root.CF.merge_CF(new_root.CF_list[1][0])
            self.root = new_root
            new_node_1.parent = self.root
            new_node_2.parent = self.root

        else:
            parent_node = current_node.parent
            new_node_1.parent = parent_node
            new_node_2.parent = parent_node
            
            # Remove the old node entry from the parent
            parent_node.CF_list = [entry for entry in parent_node.CF_list if entry[1] != current_node]

            # Add the new nodes to the parent
            parent_node.CF_list.append([copy.deepcopy(new_node_1.CF), new_node_1])
            parent_node.CF_list.append([copy.deepcopy(new_node_2.CF), new_node_2])

            # Update the CF of the parent node
            parent_node.CF = CF()  # reset the CF
            for cf, _ in parent_node.CF_list:
                parent_node.CF.merge_CF(cf)

            # Check if the parent node needs to be split
            if len(parent_node.CF_list) > self.branching_factor_B:
                new_internal_node_1, new_internal_node_2 = self.split_node(parent_node, are_leaves = False)
                self.recursive_update(parent_node, new_internal_node_1, new_internal_node_2)

            else: # the parent node has enough space, but we still need to updates the CF in the higher levels
                current_node = parent_node
                parent_node = current_node.parent

                while parent_node is not None:  
                    index_child = self.get_index_child(current_node, parent_node)
                    parent_node.CF_list[index_child][0] = copy.deepcopy(current_node.CF)

                    parent_node.CF = CF()  # reset the CF
                    for cf, _ in parent_node.CF_list:
                        parent_node.CF.merge_CF(cf)

                    current_node = parent_node
                    parent_node = current_node.parent
                

    # To get the "first" leafnode. Will be useful to reconstruct the chain (defined by the pointers prev and next) of leafnodes
    def get_a_leafnode(self):
        current_node = self.root
        while current_node.is_leafnode is False:
            current_node = current_node.CF_list[0][1] # recursively get the "first" leafnode
        return current_node



def display_tree(node, depth=0):
    indent = '    ' * depth
    if node.is_leafnode:
        print(f"{indent}Depth {depth}: Leaf Node:")
        print(f"{indent}  N: {node.CF.N}")
        print(f"{indent}  LS: {node.CF.LS}")
        print(f"{indent}  SS: {node.CF.SS}\n")

        for i, subcluster in enumerate(node.CF_list):
            print(f"{indent}  Subcluster {i}:")
            print(f"{indent}    CF N: {subcluster[0].N}")
            print(f"{indent}    CF LS: {subcluster[0].LS}")
            print(f"{indent}    CF SS: {subcluster[0].SS}")
            print(f"{indent}    Diameter: {subcluster[1].diameter}\n")
    else:
        print(f"{indent}Depth {depth}: Internal Node:")
        print(f"{indent}  N: {node.CF.N}")
        print(f"{indent}  LS: {node.CF.LS}")
        print(f"{indent}  SS: {node.CF.SS}\n")
        
        for i, child in enumerate(node.CF_list):
            print(f"{indent}  Child {i}:")
            display_tree(child[1], depth + 1)



# # TEST 1
# tree = CFTree(branching_factor_B = 3, threshold_T = 0.5, max_num_entries_leafnode_L = 2, distance_metric = average_intercluster_distance_D2)
# datapoints = np.array([[0, 0, 0], [100, 100, 100], [-1, 2, -0.4], [100.1, 100, 99.8], [1.01, 2.1, 3], [50, 2.1, 3], [5, 90.6, 20], [7, 30.5, -2.3]])

# tree.insert(datapoints[0])
# tree.insert(datapoints[1])
# tree.insert(datapoints[2])
# tree.insert(datapoints[3])
# tree.insert(datapoints[4])
# tree.insert(datapoints[5])
# tree.insert(datapoints[6])
# tree.insert(datapoints[7])

# print("\n\nDISPLAYING CF TREE:\n")
# display_tree(tree.root)

# # node = tree.root.CF_list[0][1]
# # print(node.CF_list[0][1].get_depth())



# # TEST 2
# from sklearn import datasets
# iris = datasets.load_iris()

# tree = CFTree(branching_factor_B = 3, threshold_T = 0.5, max_num_entries_leafnode_L = 2, distance_metric = average_intercluster_distance_D2)

# for i in range(len(iris.data)):
#     tree.insert(iris.data[i])

# print("\n\nDISPLAYING CF TREE:\n")
# display_tree(tree.root)



# # TEST 3
# from sklearn import datasets
# iris = datasets.load_iris()

# tree = CFTree(branching_factor_B = 4, threshold_T = 0.5, max_num_entries_leafnode_L = 3, distance_metric = average_intercluster_distance_D2)

# for i in range(len(iris.data)):
#     tree.insert(iris.data[i])

# print("\n\nDISPLAYING CF TREE:\n")
# display_tree(tree.root)
    