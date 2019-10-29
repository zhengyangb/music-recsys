import numpy as np
from annoy import AnnoyIndex
import pickle
import os
import time
from tqdm import tqdm_notebook as tqdm
import numpy as np
import matplotlib.pyplot as plt

# Please import your matrix
track_vec = pickle.load(open('./item_vec.pkl', 'rb'))
test_user_vec = pickle.load(open('./user_vec.pkl', 'rb'))


# Example: Build a AnnoyTree
f = len(track_vec[0])
t = AnnoyIndex(f, metric='dot')
for i in range(len(track_vec)):
    t.add_item(i, track_vec[i])

t.build(30)

# Build exhaustive search


def find_nearest_exhaustive(data, queries, k):
    if len(data.shape) == 1:
        data = np.array([x for x in data])
    n_items = data.shape[0]
    n_feat = data.shape[1]
    n_queries = len(queries)
    
    def single_query(query):
        start = time.time()
        if type(query) is not np.ndarray:
            query = np.array(query)
        res = np.argsort(-data.dot(query))[:k]
        interval = time.time() - start
        return interval, res
    times = []
    results = []
    for i in tqdm(range(n_queries)):
        interval, res = single_query(queries[i])
        times.append(interval)
        results.append(res)
    mean_time = sum(times) / len(times)
    print('-' * 26)
    print('Exhaustive Brute-force Search\n')
    print('Mean Query Search: %.6f' % mean_time)
    
    return mean_time, results    

bf_mean_time, bf_results = find_nearest_exhaustive(track_vec, test_user_vec, 500)


# Wrap the algorithm object 
# so it our code could support more algorithms in the future

def wrap_with(obj, method, mapping):
    '''
    obj: the model that can respond to the query
    method: the name of the query method
    mapping: what input be mapped
    '''
    get_map = lambda x: [x[mapping[i]] for i in range(len(mapping))]
    def wrapped(*args, **kwrds):
        return obj.__getattribute__(method)(*get_map(args))
    return wrapped


def find_nearest_algo(data, queries, true_label, model_wrapped, k, extra_para):
    if len(data.shape) == 1:
        data = np.array([x for x in data])
    n_items = data.shape[0]
    n_feat = data.shape[1]
    n_queries = len(queries)
    def single_query(query):
        start = time.time()
        res = model_wrapped(query, k, extra_para)
        interval = time.time() - start
        return interval, res
    def get_recall(predict, truth):
        return len([x for x in predict if x in truth]) / len(truth)
    times = []
    recalls = []
    for i in tqdm(range(n_queries)):
        interval, res = single_query(queries[i])
        recall = get_recall(res, true_label[i])
        times.append(interval)
        recalls.append(recall)
    mean_time = sum(times) / len(times)
    mean_recall = sum(recalls) / len(recalls)
    print('-' * 26)
    print('Algorithm with k\' = %d\n' % k)
    print('Mean Query Search Time: %.6f' % mean_time)
    print('Mean Recall: %.6f' % mean_recall)
    
    return mean_time, mean_recall 


# ### Make the plot

def make_plot(tree_list):
    results = []
    for tree in tqdm(tree_list):
        res = {'tree': tree}
        t.build(tree)
        annoy10_wrapped = wrap_with(t, 'get_nns_by_vector', [0, 1, 2])
        num_query_list = []
        recall_list = []
        for para in [100, 500, 1000, 5000, 10000, 50000, 100000, 500000]:
            algo100_time, algo100_recall = find_nearest_algo(track_vec, test_user_vec, bf_results, annoy10_wrapped, 500, para)
            num_query_list.append(1/algo100_time)
            recall_list.append(algo100_recall)
        plt.plot(recall_list, num_query_list, label = 'Annoy (num_tree = {})'.format(tree))
        plt.ylabel('Queries per second (1/s)')
        plt.xlabel('Recall')
        plt.title('Recall-Queries per second (1/s) tradeoff - up and to the right is better')
        plt.yscale('log')
        plt.legend()
        plt.grid()
        res['recall'] = recall_list
        res['time'] = num_query_list
        results.append(res)
    return results

tree_list = [1, 5, 10, 30]
plot_data_try = make_plot(tree_list)

