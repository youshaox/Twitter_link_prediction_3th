import pandas as pd
import pickle
import math
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score
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

training_data_v1 = load_obj("un_normalised_final_training_data_df_rf")
BasicFeatures = load_obj("pre_features-v2")
pre_features = BasicFeatures
final_training_data_df = training_data_v1.iloc[:,3:30]
final_labels_df = training_data_v1.iloc[:,2]


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
# training_df = get_training_df(labeled_edges)
final_training_data_df_former = training_data_v1.iloc[:,3:30]
final_labels_df = training_data_v1.iloc[:,2]
training_df= training_data_v1

final_labels_df = training_df.iloc[:,2]
final_training_data_df = final_training_data_df_former.iloc[:,9:15]
# 使用标准化
# final_training_data_df = rescale_min_max(measurement_to_normal)




X=final_training_data_df
# count=0
# get the data and label
y=final_labels_df

# training model
from sklearn.model_selection import train_test_split
X_t, X_test, y_t, y_test = train_test_split(X,y)
X_train, X_validation, y_train, y_validation  = train_test_split(X_t,y_t)

# Gridsearch settings
rf = RandomForestClassifier()
X_train = X_t
y_train = y_t

pipeline = Pipeline([
       ('clf', RandomForestClassifier(criterion='entropy'))
   ])
parameters = {
       'clf__n_estimators': (50, 100),
       'clf__max_depth': (5, 10, 20),
       'clf__min_samples_split': (10, 500),
       'clf__min_samples_leaf': (10, 100, 1000, 5000)
}
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
#         common_neighbours = get_common_neighbours(source,sink)
#         num_of_neighbours_source=BasicFeatures[source][0]
#         num_of_in_neighbours_source=BasicFeatures[source][4]
#         num_of_out_neighbours_source=BasicFeatures[source][6]

#         num_of_neighbours_sink=BasicFeatures[sink][0]
#         num_of_in_neighbours_sink=BasicFeatures[sink][4]
#         num_of_out_neighbours_sink=BasicFeatures[sink][6]
        
#         num_of_neighbours_sum=BasicFeatures[source][0] + BasicFeatures[sink][0]
#         num_of_in_neighbours_sum=BasicFeatures[source][4] + BasicFeatures[sink][4]
#         num_of_out_neighbours_sum=BasicFeatures[source][6] + BasicFeatures[sink][6]
        
        salton_similarity_score = salton_similarity(source, sink)
        cosine = Cosine(source, sink)
        jaccard_coefficient = get_jaccard_coefficient(source, sink)
        preferential_attachment = get_preferential_attachment(source, sink)
        adamic_adar = get_adamic_adar(source, sink)
        resource_allocation = get_resource_allocation(source, sink)

#         salton_similarity_score_out = get_outbound_similarity_score(source, sink, salton_similarity)
#         cosine_out = get_outbound_similarity_score(source, sink, Cosine)
#         jaccard_coefficient_out = get_outbound_similarity_score(source, sink, get_jaccard_coefficient)
#         preferential_attachment_out = get_outbound_similarity_score(source, sink, get_preferential_attachment)
#         adamic_adar_out = get_outbound_similarity_score(source, sink, get_adamic_adar)
#         resource_allocation_out = get_outbound_similarity_score(source, sink, get_resource_allocation)

#         salton_similarity_score_in = get_inbound_similarity_score(source, sink, salton_similarity)
#         cosine_in = get_inbound_similarity_score(source, sink, Cosine)
#         jaccard_coefficient_in = get_inbound_similarity_score(source, sink, get_jaccard_coefficient)
#         preferential_attachment_in = get_inbound_similarity_score(source, sink, get_preferential_attachment)
#         adamic_adar_in = get_inbound_similarity_score(source, sink, get_adamic_adar)
#         resource_allocation_in = get_inbound_similarity_score(source, sink, get_resource_allocation)

#         df_row = pd.DataFrame([cosine, jaccard_coefficient, preferential_attachment, adamic_adar, resource_allocation]).T
        X_test = pd.DataFrame([
#                                num_of_neighbours_source,
#                                num_of_in_neighbours_source,
#                                num_of_out_neighbours_source,
#                                num_of_neighbours_sink,
#                                num_of_in_neighbours_sink,
#                                num_of_out_neighbours_sink,
#                                num_of_neighbours_sum,
#                                num_of_in_neighbours_sum,
#                                num_of_out_neighbours_sum,      
                               salton_similarity_score, 
                               cosine, 
                               jaccard_coefficient,
                               preferential_attachment, 
                               adamic_adar, 
                               resource_allocation
#                                salton_similarity_score_out,
#                                cosine_out,
#                                jaccard_coefficient_out,
#                                preferential_attachment_out,
#                                adamic_adar_out,
#                                resource_allocation_out,
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
save_prediction_to_csv(result, "shawn_rf_6-no-small")
