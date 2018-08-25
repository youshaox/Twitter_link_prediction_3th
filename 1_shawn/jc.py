import csv
import time


data_dir = "../../data/"
with open(data_dir + "train.txt", "r") as f:
     train_data = f.readlines()

# 1. node set
# 2. edge tuple
nodes = []
edges = []
for i in range(len(train_data)):
    #if i%100 == 0:
        #print(i)
    nodes_list = [int(n) for n in train_data[i].split()]
    for node in nodes_list:
        nodes.append(node)
    for node in nodes_list[1:]:
        edges.append((nodes_list[0],node))
nodes = set(nodes)
# ordered edges by the source
sorted_edges = sorted(edges, key=lambda tup: tup[0])


# General node class
class Node:
    def __init__(self, node_id, neighbour_id_set):
        self.node_id = node_id
        self.neighbour_id_set = neighbour_id_set
    def get_node_id(self):
        return self.get_node_id
    def get_neighbour_id_set(self):
        return self.neighbour_id_set

def find_neighbours(id):
    
    """
    find all the neighbours of node by id.
    1. All the sink node of node id will be appended as the neighbor first.
    2. All the source node of node id will also be added as neighbor then.
    3. return the neighbor set.
    """
    neighbour_set = set()
    for edge in sorted_edges:
        if edge[0] == id:
            # add the sink node of node id
            neighbour_set.add(edge[1])
        elif edge[1] == id:
            # add the source node of node id
            neighbour_set.add(edge[0])
    return neighbour_set

def get_jaccard_coefficient(node_x, node_y):
    """
    in: node_x::Node object
    in: node_y::Node object
    return: jaccard's cofficient::numeric
    """
    score = 0.0
    neigbours_set_of_node_x = node_x.get_neighbour_id_set()
    neigbours_set_of_node_y = node_y.get_neighbour_id_set()
    union_neighbours = neigbours_set_of_node_x | neigbours_set_of_node_y
    common_neighbours = neigbours_set_of_node_x & neigbours_set_of_node_y
    if len(union_neighbours)==0:
        return 0.0
    return(len(common_neighbours)/len(union_neighbours))

with open(data_dir + "test-public.txt", "r") as f:
     test_data = f.readlines()
test_data = [i.split() for i in test_data[1:]]

def predict():
    """
    make the prediction using the jaccard's coefficient
    """
    result = []
    for line in test_data:
        # converse to integer
        point_x = int(line[1].strip())
        point_y = int(line[2].strip())
        node_x = Node(point_x,find_neighbours(point_x))
        node_y = Node(point_y,find_neighbours(point_y))
        jaccard_coefficient = get_jaccard_coefficient(node_x, node_y)
        result.append((line[0], jaccard_coefficient))
    return result
result = predict()

def nowtime():
    return time.strftime("%Y%m%d-%H%M", time.localtime())

"""
Description: Save prediction result to files
Input: (1) result
       (2) filename
Output: 
"""
def save_prediction_to_csv(result,filename):
    headers = ['id','Prediction']

    with open(filename + str(nowtime()) + ".csv", 'w', encoding = 'utf8') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(headers)
        f_csv.writerows(result)

save_prediction_to_csv(result, "shawn_jc_")
