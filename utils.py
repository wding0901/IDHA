import numpy as np
import dgl


def minibatch(*tensors, **kwargs):
    batch_size = kwargs.get('batch_size', 2048)

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def random_sampling(src_nodes, sample_num, neighbor_table):
    results = []
    for sid in src_nodes:
        # 从节点的邻居中进行有放回地进行采样
        res = np.random.choice(neighbor_table[sid], size=(sample_num,))
        results.append(res)
    return np.asarray(results).flatten()  # flatten转成一维数组


def random_multihop_sampling(src_nodes, sample_nums, neighbor_table):
    sampling_result = [src_nodes]
    for k, hopk_num in enumerate(sample_nums):
        hopk_result = random_sampling(sampling_result[k], hopk_num, neighbor_table)
        sampling_result.append(hopk_result)
    return sampling_result


def pinsage_samling(pinsage_args, user_item_graph, src_nodes, sample_nums, ctype):
    sampling_result = [src_nodes]
    num_traversals = pinsage_args[0]
    termination_prob = pinsage_args[1]
    num_random_walks = pinsage_args[2]
    G = dgl.heterograph({
        ('A', 'AB', 'B'): user_item_graph.nonzero(),
        ('B', 'BA', 'A'): user_item_graph.T.nonzero()})
    for k, hopk_num in enumerate(sample_nums):
        if ctype is 'user':
            sampler = dgl.sampling.PinSAGESampler(G, 'A', 'B', num_traversals, termination_prob, num_random_walks, hopk_num)
        else:
            sampler = dgl.sampling.PinSAGESampler(G, 'B', 'A', num_traversals, termination_prob, num_random_walks, hopk_num)

        for i in range(3):
            neighbours, _ = sampler(sampling_result[k]).all_edges(form='uv')
            if neighbours.shape[0] == len(sampling_result[k]) * hopk_num:
                break
            print("sample fail")

        sampling_result.append(neighbours.numpy())
    return sampling_result


def rw_samling(rw_args, adj_mat, src_nodes, sample_nums):
    sampling_result = [src_nodes]
    num_traversals = rw_args[0]
    termination_prob = rw_args[1]
    num_random_walks = rw_args[2]
    G = dgl.from_scipy(adj_mat)

    for k, hopk_num in enumerate(sample_nums):
        sampler = dgl.sampling.RandomWalkNeighborSampler(G, num_traversals, termination_prob, num_random_walks, hopk_num)

        for i in range(3):
            neighbours, _ = sampler(sampling_result[k]).all_edges(form='uv')
            if neighbours.shape[0] == len(sampling_result[k]) * hopk_num:
                break
            print("sample fail")

        sampling_result.append(neighbours.numpy())
    return sampling_result
