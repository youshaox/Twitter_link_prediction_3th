import networkx as nx

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
print("Finish read all the data")
graph = nx.Graph()
graph.add_nodes_from(nodes)
graph.add_edges_from(edges)
print("Finish initilise the graph.")
def get_jaccard_coefficient(graph, node_x, node_y):
    score = 0.0
    common_neighbors = sorted(nx.common_neighbors(graph, node_x, node_y))
    union_neighbors = set(sorted(nx.all_neighbors(graph, node_x)) + sorted(nx.all_neighbors(graph, node_y)))
    print(len(common_neighbors))
    print(len(union_neighbors))
#     node_y_neighbors = sorted(nx.all_neighbors(graph, node_y))
#     union_list = 
    return(len(common_neighbors)/len(union_neighbors))
print(get_jaccard_coefficient(graph,2184483,1300190))
