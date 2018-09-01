import pandas as pd
import pickle
import math
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm



data_dir = "../../data/"


def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(data_dir + name + '.pkl', 'rb') as f:
        return pickle.load(f)

# load data
Old_Labeled_data = load_obj("SBdata")
BasicFeatures = load_obj("pre_features-v2")
pre_features = BasicFeatures

def transform_data(Old_Labeled_data):
    labeled_edges = []
    for Old_Labeled_edge in Old_Labeled_data:
        label = int(Old_Labeled_edge[1])
        source = int(Old_Labeled_edge[0][0])
        sink = int(Old_Labeled_edge[0][1])
        labeled_edges.append((source, sink, label))
    return labeled_edges
labeled_edges = transform_data(Old_Labeled_data)


# Function
#Salton Similarity
def salton_similarity(node1, node2):
    n1 = pre_features[node1]
    n2 = pre_features[node2]
    common_neighors = list(set(n1[2]).intersection(n2[2]))
    inter = len(common_neighors)
    degree_out_flow = n1[6]
    degree_in_flow = n2[4]
    
    if inter == 0:
        return 0
    else:
        try:
            sqrt_of_degree = math.sqrt(degree_out_flow * degree_in_flow)
            salton = inter / sqrt_of_degree
            probability = 1 /(1 - math.log(salton)*0.2)
            return probability
        except:
            return 0

#Cosine
def Cosine(Node1, Node2):
    n1 = pre_features[Node1]
    n2 = pre_features[Node2]
    common_neighors = list(set(n1[2]).intersection(n2[2]))
    lm = len(common_neighors)
    if lm == 0:
        return 0
    else:
        return (0.0+lm)/(len(n1[2])*len(n2[2]))

def get_jaccard_coefficient(source, sink):
    """
    in: source::Node object
    in: sink::Node object
    return: jaccard's cofficient::numeric
    """
    # transform
    neighbours_of_source_list = BasicFeatures[source][2]
    neighbours_of_sink_list = BasicFeatures[sink][2]
    
    neigbours_set_of_source = set(neighbours_of_source_list)
    neigbours_set_of_sink = set(neighbours_of_sink_list)
    union_neighbours = neigbours_set_of_source | neigbours_set_of_sink
    common_neighbours = neigbours_set_of_source & neigbours_set_of_sink
    if len(union_neighbours)==0:
        return 0.0
    return(len(common_neighbours)/len(union_neighbours))

def get_preferential_attachment(source, sink):
    # transform
    neighbours_of_source_list = BasicFeatures[source][2]
    neighbours_of_sink_list = BasicFeatures[sink][2]
    
    neigbours_set_of_source = set(neighbours_of_source_list)
    neigbours_set_of_sink = set(neighbours_of_sink_list)
    
    return len(neigbours_set_of_source)*len(neigbours_set_of_sink)

def get_adamic_adar(source, sink):
    # transform
    neighbours_of_source_list = BasicFeatures[source][2]
    neighbours_of_sink_list = BasicFeatures[sink][2]

    neigbours_set_of_source = set(neighbours_of_source_list)
    neigbours_set_of_sink = set(neighbours_of_sink_list)
    common_neighbours = neigbours_set_of_source & neigbours_set_of_sink
    # get the summation
    score = 0
    for common_node in common_neighbours:
        if math.log(len(BasicFeatures[common_node][2])) == 0:
            return 0.0
        score = score + 1/math.log(len(BasicFeatures[common_node][2]))
    return score

def get_resource_allocation(source, sink):
    neighbours_of_source_list = BasicFeatures[source][2]
    neighbours_of_sink_list = BasicFeatures[sink][2]
#     print(neighbours_of_source_list)
#     print(neighbours_of_sink_list)
    neigbours_set_of_source = set(neighbours_of_source_list)
    neigbours_set_of_sink = set(neighbours_of_sink_list)
    
    common_neighbours = neigbours_set_of_source & neigbours_set_of_sink
#     print(common_neighbours)
    score=0
    for common_node in common_neighbours:
        # number of the neighbours of the common_node
        try:
            single_common_node_score = 1/BasicFeatures[common_node][0]
        except:
            single_common_node_score=0
        score = score + single_common_node_score
    return score
    

# how similar are the outbound neighbors of source to sink
# either JA, PA, AA
def get_outbound_similarity_score(source, sink, metric):
    # get the outbound_node of source
    outbound_node_for_source_set = set(BasicFeatures[source][5])
    summation = 0
    for outbound_node_for_source in outbound_node_for_source_set:
        summation =summation + metric(sink,outbound_node_for_source)
    if len(outbound_node_for_source_set) == 0:
        return 0
    score = 1/len(outbound_node_for_source_set)*summation
    return score

# either JA, PA, AA
def get_inbound_similarity_score(source, sink, metric):
    # get the inbound_node of sink
    inbound_node_for_sink_set = set(BasicFeatures[sink][3])
    summation = 0
    for inbound_node_for_sink in inbound_node_for_sink_set:
        summation =summation + metric(source,inbound_node_for_sink)
    if len(inbound_node_for_sink_set) == 0:
        return 0
    score = 1/len(inbound_node_for_sink_set)*summation
    return score

def get_common_neighbours(node1, node2):
    try:
        n1 = pre_features[node1]
        n2 = pre_features[node2]
        common_neighors = list(set(n1[2]).intersection(n2[2]))
        return common_neighors
    except:
        return 0

def get_training_df(final_edges):
    training_df = pd.DataFrame()
    for edge in tqdm(final_edges, mininterval=60):
        source = edge[0]
        sink = edge[1]
        label = edge[2]
        common_neighbours = get_common_neighbours(source,sink)
        num_of_neighbours_source=BasicFeatures[source][0]
        num_of_in_neighbours_source=BasicFeatures[source][4]
        num_of_out_neighbours_source=BasicFeatures[source][6]

        num_of_neighbours_sink=BasicFeatures[sink][0]
        num_of_in_neighbours_sink=BasicFeatures[sink][4]
        num_of_out_neighbours_sink=BasicFeatures[sink][6]
        
        num_of_neighbours_sum=BasicFeatures[source][0] + BasicFeatures[sink][0]
        num_of_in_neighbours_sum=BasicFeatures[source][4] + BasicFeatures[sink][4]
        num_of_out_neighbours_sum=BasicFeatures[source][6] + BasicFeatures[sink][6]
        
        salton_similarity_score = salton_similarity(source, sink)
        cosine = Cosine(source, sink)
        jaccard_coefficient = get_jaccard_coefficient(source, sink)
        preferential_attachment = get_preferential_attachment(source, sink)
        adamic_adar = get_adamic_adar(source, sink)
        resource_allocation = get_resource_allocation(source, sink)

        salton_similarity_score_out = get_outbound_similarity_score(source, sink, salton_similarity)
        cosine_out = get_outbound_similarity_score(source, sink, Cosine)
        jaccard_coefficient_out = get_outbound_similarity_score(source, sink, get_jaccard_coefficient)
        preferential_attachment_out = get_outbound_similarity_score(source, sink, get_preferential_attachment)
        adamic_adar_out = get_outbound_similarity_score(source, sink, get_adamic_adar)
        resource_allocation_out = get_outbound_similarity_score(source, sink, get_resource_allocation)

#         salton_similarity_score_in = get_inbound_similarity_score(source, sink, salton_similarity)
#         cosine_in = get_inbound_similarity_score(source, sink, Cosine)
#         jaccard_coefficient_in = get_inbound_similarity_score(source, sink, get_jaccard_coefficient)
#         preferential_attachment_in = get_inbound_similarity_score(source, sink, get_preferential_attachment)
#         adamic_adar_in = get_inbound_similarity_score(source, sink, get_adamic_adar)
#         resource_allocation_in = get_inbound_similarity_score(source, sink, get_resource_allocation)

# add the basic features
        df_row = pd.DataFrame([
                               source, 
                               sink, 
                               label,
                               num_of_neighbours_source,
                               num_of_in_neighbours_source,
                               num_of_out_neighbours_source,
                               num_of_neighbours_sink,
                               num_of_in_neighbours_sink,
                               num_of_out_neighbours_sink,
                               num_of_neighbours_sum,
                               num_of_in_neighbours_sum,
                               num_of_out_neighbours_sum,      
                               salton_similarity_score, 
                               cosine, 
                               jaccard_coefficient,
                               preferential_attachment, 
                               adamic_adar, 
                               resource_allocation,
                               salton_similarity_score_out,
                               cosine_out,
                               jaccard_coefficient_out,
                               preferential_attachment_out,
                               adamic_adar_out,
                               resource_allocation_out
#                                salton_similarity_score_in,
#                                cosine_in,
#                                jaccard_coefficient_in,
#                                preferential_attachment_in,
#                                adamic_adar_in,
#                                resource_allocation_in
                              ]).T
        training_df = training_df.append(df_row)
    training_df.rename(columns={
        0:'source', 
        1:'sink', 
        2:'label',
        3:'num_of_neighbours_source',
        4:'num_of_in_neighbours_source',
        5:'num_of_out_neighbours_source',
        6:'num_of_neighbours_sink',
        7:'num_of_in_neighbours_sink',
        8:'num_of_out_neighbours_sink',
        9:'num_of_neighbours_sum',
        10:'num_of_in_neighbours_sum',
        11:'num_of_out_neighbours_sum',      
        12:'salton_similarity_score', 
        13:'cosine', 
        14:'jaccard_coefficient',
        15:'preferential_attachment', 
        16:'adamic_adar', 
        17:'resource_allocation',
        18:'salton_similarity_score_out',
        19:'cosine_out',
        20:'jaccard_coefficient_out',
        21:'preferential_attachment_out',
        22:'adamic_adar_out',
        23:'resource_allocation_out'
#         25:'salton_similarity_score_in',
#         26:'cosine_in',
#         27:'jaccard_coefficient_in',
#         28:'preferential_attachment_in',
#         29:'adamic_adar_in',
#         30:'resource_allocation_in'           
    },inplace=True)
    training_df[['source', 'sink', 'label']] = training_df[['source', 'sink', 'label']].astype(int)
    return training_df

# data 需要为array
def rescale_min_max(data): 
    """
    min-max normalisation
    """
    scaler = MinMaxScaler()
    scaler.fit(data)
    result = scaler.transform(data)
    return pd.DataFrame(result)

def standardise(data):
    """remove the mean and transform to unit variance"""
    scaler = StandardScaler()
    scaler.fit(data)
    result = scaler.transform(data)
    return pd.DataFrame(result)


# normalise the training data
training_df = get_training_df(labeled_edges)
final_labels_df = training_df.iloc[:,2]
measurement_to_normal = training_df.iloc[:,3:30]
# 使用标准化
final_training_data_df = standardise(measurement_to_normal)
final_training_data_df.rename(columns={
        0:'num_of_neighbours_source',
        1:'num_of_in_neighbours_source',
        2:'num_of_out_neighbours_source',
        3:'num_of_neighbours_sink',
        4:'num_of_in_neighbours_sink',
        5:'num_of_out_neighbours_sink',
        6:'num_of_neighbours_sum',
        7:'num_of_in_neighbours_sum',
        8:'num_of_out_neighbours_sum',      
        9:'salton_similarity_score', 
        10:'cosine', 
        11:'jaccard_coefficient',
        12:'preferential_attachment', 
        13:'adamic_adar', 
        14:'resource_allocation',
        15:'salton_similarity_score_out',
        16:'cosine_out',
        17:'jaccard_coefficient_out',
        18:'preferential_attachment_out',
        19:'adamic_adar_out',
        20:'resource_allocation_out'
#         21:'salton_similarity_score_in',
#         22:'cosine_in',
#         23:'jaccard_coefficient_in',
#         24:'preferential_attachment_in',
#         25:'adamic_adar_in',
#         26:'resource_allocation_in'           
    },inplace=True)
save_obj(final_training_data_df,'final_training_data_df_lr_script')

X=final_training_data_df
# count=0
# get the data and label
y=final_labels_df

# training model
from sklearn.model_selection import train_test_split
X_t, X_test, y_t, y_test = train_test_split(X,y)
X_train, X_validation, y_train, y_validation  = train_test_split(X_t,y_t)
# Gridsearch settings
pipeline = Pipeline([
    ('clf', LogisticRegression())
])
parameters = {
       'clf__penalty': ('l1', 'l2'),
       'clf__C': (0.01, 0.1, 1, 5, 10),
 }
# 1. training_df_10w running
X_train = X_t
y_train = y_t
grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1,
   verbose=1, scoring='roc_auc', cv=3)
grid_search.fit(X_train, y_train)
print('Best score: %0.3f' % grid_search.best_score_)
print('Best parameters set:')
best_parameters = grid_search.best_estimator_.get_params()

for param_name in sorted(parameters.keys()):
    print('\t%s: %r' % (param_name, best_parameters[param_name]))
predictions = grid_search.predict(X_test)
print('Accuracy:', accuracy_score(y_test, predictions))
print('Precision:', precision_score(y_test, predictions))
print('Recall:', recall_score(y_test, predictions))




# make the prediction
from tqdm import tqdm
with open(data_dir + "test-public.txt", "r") as f:
     test_data = f.readlines()
test_data = [i.split() for i in test_data[1:]]
def predict():
    """
    make the prediction using the jaccard's coefficient
    """
    result = []
    for line in tqdm(test_data, mininterval=50):
        # converse to integer
        source = int(line[1].strip())
        sink = int(line[2].strip())
        common_neighbours = get_common_neighbours(source,sink)
        num_of_neighbours_source=BasicFeatures[source][0]
        num_of_in_neighbours_source=BasicFeatures[source][4]
        num_of_out_neighbours_source=BasicFeatures[source][6]

        num_of_neighbours_sink=BasicFeatures[sink][0]
        num_of_in_neighbours_sink=BasicFeatures[sink][4]
        num_of_out_neighbours_sink=BasicFeatures[sink][6]
        
        num_of_neighbours_sum=BasicFeatures[source][0] + BasicFeatures[sink][0]
        num_of_in_neighbours_sum=BasicFeatures[source][4] + BasicFeatures[sink][4]
        num_of_out_neighbours_sum=BasicFeatures[source][6] + BasicFeatures[sink][6]
        
        salton_similarity_score = salton_similarity(source, sink)
        cosine = Cosine(source, sink)
        jaccard_coefficient = get_jaccard_coefficient(source, sink)
        preferential_attachment = get_preferential_attachment(source, sink)
        adamic_adar = get_adamic_adar(source, sink)
        resource_allocation = get_resource_allocation(source, sink)

        salton_similarity_score_out = get_outbound_similarity_score(source, sink, salton_similarity)
        cosine_out = get_outbound_similarity_score(source, sink, Cosine)
        jaccard_coefficient_out = get_outbound_similarity_score(source, sink, get_jaccard_coefficient)
        preferential_attachment_out = get_outbound_similarity_score(source, sink, get_preferential_attachment)
        adamic_adar_out = get_outbound_similarity_score(source, sink, get_adamic_adar)
        resource_allocation_out = get_outbound_similarity_score(source, sink, get_resource_allocation)

#         salton_similarity_score_in = get_inbound_similarity_score(source, sink, salton_similarity)
#         cosine_in = get_inbound_similarity_score(source, sink, Cosine)
#         jaccard_coefficient_in = get_inbound_similarity_score(source, sink, get_jaccard_coefficient)
#         preferential_attachment_in = get_inbound_similarity_score(source, sink, get_preferential_attachment)
#         adamic_adar_in = get_inbound_similarity_score(source, sink, get_adamic_adar)
#         resource_allocation_in = get_inbound_similarity_score(source, sink, get_resource_allocation)

#         df_row = pd.DataFrame([cosine, jaccard_coefficient, preferential_attachment, adamic_adar, resource_allocation]).T
        X_test = pd.DataFrame([
                               num_of_neighbours_source,
                               num_of_in_neighbours_source,
                               num_of_out_neighbours_source,
                               num_of_neighbours_sink,
                               num_of_in_neighbours_sink,
                               num_of_out_neighbours_sink,
                               num_of_neighbours_sum,
                               num_of_in_neighbours_sum,
                               num_of_out_neighbours_sum,      
                               salton_similarity_score, 
                               cosine, 
                               jaccard_coefficient,
                               preferential_attachment, 
                               adamic_adar, 
                               resource_allocation,
                               salton_similarity_score_out,
                               cosine_out,
                               jaccard_coefficient_out,
                               preferential_attachment_out,
                               adamic_adar_out,
                               resource_allocation_out
#                                salton_similarity_score_in,
#                                cosine_in,
#                                jaccard_coefficient_in,
#                                preferential_attachment_in,
#                                adamic_adar_in,
#                                resource_allocation_in
                              ]).T
        single_result = grid_search.predict(X_test)[0]
        print(single_result)
        result.append((line[0], single_result))
    return result
result = predict()


# save the result

import csv
import time
'''
Description: get time
Input: 
Output: time
''' 
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
save_prediction_to_csv(result, "shawn_lr_ds")