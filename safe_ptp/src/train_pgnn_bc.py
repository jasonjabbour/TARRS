 
# def get_tg_dataset():

#     data_list = []
#     dists_list = []
#     dists_removed_list = []
#     links_train_list = []
#     links_val_list = []
#     links_test_list = []
#     for i, data in enumerate(dataset):
#         if 'link' in args.task:
#             get_link_mask(data, args.remove_link_ratio, resplit=True,
#                             infer_link_positive=True if args.task == 'link' else False)
#         links_train_list.append(data.mask_link_positive_train)
#         links_val_list.append(data.mask_link_positive_val)
#         links_test_list.append(data.mask_link_positive_test)
#         if args.task=='link':
#             dists_removed = precompute_dist_data(data.mask_link_positive_train, data.num_nodes,
#                                                     approximate=args.approximate)
#             dists_removed_list.append(dists_removed)
#             data.dists = torch.from_numpy(dists_removed).float()
#             data.edge_index = torch.from_numpy(duplicate_edges(data.mask_link_positive_train)).long()

#         else:
#             dists = precompute_dist_data(data.edge_index.numpy(), data.num_nodes, approximate=args.approximate)
#             dists_list.append(dists)
#             data.dists = torch.from_numpy(dists).float()
#         if remove_feature:
#             data.x = torch.ones((data.x.shape[0],1))
#         data_list.append(data)

        
#     return data_list

# def duplicate_edges(edges):
#     return np.concatenate((edges, edges[::-1,:]), axis=-1)

# # Step 1 create your graph and make sure to get the features correct they have them as shape (100,100)

# # you have this funciton 

# # Step 2: Convert to torchgeometric data

# dataset = load_tg_dataset(dataset_name)

# def nx_to_tg_data(graphs, features, edge_labels=None):
#     data_list = []
#     for i in range(len(graphs)):
#         feature = features[i]
#         graph = graphs[i].copy()
#         # Updated:
#         graph.remove_edges_from(nx.selfloop_edges(graph))

#         # relabel graphs
#         keys = list(graph.nodes)
#         vals = range(graph.number_of_nodes())
#         mapping = dict(zip(keys, vals))
#         nx.relabel_nodes(graph, mapping, copy=False)

#         x = np.zeros(feature.shape)
#         graph_nodes = list(graph.nodes)
#         for m in range(feature.shape[0]):
#             x[graph_nodes[m]] = feature[m]
#         x = torch.from_numpy(x).float()

#         # get edges
#         edge_index = np.array(list(graph.edges))
#         edge_index = np.concatenate((edge_index, edge_index[:,::-1]), axis=0)
#         edge_index = torch.from_numpy(edge_index).long().permute(1,0)

#         data = Data(x=x, edge_index=edge_index)
#         # get edge_labels
#         if edge_labels[0] is not None:
#             edge_label = edge_labels[i]
#             mask_link_positive = np.stack(np.nonzero(edge_label))
#             data.mask_link_positive = mask_link_positive
#         data_list.append(data)
#     return data_list

# # 3. compute dist?

# def precompute_dist_data(edge_index, num_nodes, approximate=0):
#         '''
#         Here dist is 1/real_dist, higher actually means closer, 0 means disconnected
#         :return:
#         '''
#         graph = nx.Graph()
#         edge_list = edge_index.transpose(1,0).tolist()
#         graph.add_edges_from(edge_list)

#         n = num_nodes
#         dists_array = np.zeros((n, n))
#         # dists_dict = nx.all_pairs_shortest_path_length(graph,cutoff=approximate if approximate>0 else None)
#         # dists_dict = {c[0]: c[1] for c in dists_dict}
#         dists_dict = all_pairs_shortest_path_length_parallel(graph,cutoff=approximate if approximate>0 else None)
#         for i, node_i in enumerate(graph.nodes()):
#             shortest_dist = dists_dict[node_i]
#             for j, node_j in enumerate(graph.nodes()):
#                 dist = shortest_dist.get(node_j, -1)
#                 if dist!=-1:
#                     # dists_array[i, j] = 1 / (dist + 1)
#                     dists_array[node_i, node_j] = 1 / (dist + 1)
#         return dists_array


# def all_pairs_shortest_path_length_parallel(graph,cutoff=None,num_workers=4):
#     nodes = list(graph.nodes)
#     random.shuffle(nodes)
#     if len(nodes)<50:
#         num_workers = int(num_workers/4)
#     elif len(nodes)<400:
#         num_workers = int(num_workers/2)

#     pool = mp.Pool(processes=num_workers)
#     results = [pool.apply_async(single_source_shortest_path_length_range,
#             args=(graph, nodes[int(len(nodes)/num_workers*i):int(len(nodes)/num_workers*(i+1))], cutoff)) for i in range(num_workers)]
#     output = [p.get() for p in results]
#     dists_dict = merge_dicts(output)
#     pool.close()
#     pool.join()
#     return dists_dict


# def single_source_shortest_path_length_range(graph, node_range, cutoff):
#     dists_dict = {}
#     for node in node_range:
#         dists_dict[node] = nx.single_source_shortest_path_length(graph, node, cutoff)
#     return dists_dict


# # 4. Select anchor

# for i,data in enumerate(data_list):
#     preselect_anchor(data, layer_num=args.layer_num, anchor_num=args.anchor_num, device='cpu')
#     data = data.to(device)
#     data_list[i] = data

  
# # Training Loop
# def train():

#     # 5. Initialize model

#     # model
#     input_dim = num_features
#     output_dim = args.output_dim
#     model = locals()[args.model](input_dim=input_dim, feature_dim=args.feature_dim,
#                 hidden_dim=args.hidden_dim, output_dim=output_dim,
#                 feature_pre=args.feature_pre, layer_num=args.layer_num, dropout=args.dropout).to(device)


#     # loss
#     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
#     if 'link' in args.task:
#         loss_func = nn.BCEWithLogitsLoss()
#         out_act = nn.Sigmoid()


#     for epoch in range(2000):
#         if epoch==200:
#             for param_group in optimizer.param_groups:
#                 param_group['lr'] /= 10
#         model.train()
#         optimizer.zero_grad()
#         shuffle(data_list)
#         effective_len = len(data_list)//args.batch_size*len(data_list)
#         for id, data in enumerate(data_list[:effective_len]):
#             if args.permute:
#                 preselect_anchor(data, layer_num=args.layer_num, anchor_num=args.anchor_num, device=device)
#             out = model(data)

#             if epoch == 0 or epoch == (args.epoch_num -1): 
#                 # Call the visualization function
#                 visualize_embeddings(out, title=f"t-SNE Visualization of Node Embeddings for Graph {id+1}")

#             # get_link_mask(data,resplit=False)  # resample negative links
#             edge_mask_train = np.concatenate((data.mask_link_positive_train, data.mask_link_negative_train), axis=-1)
#             nodes_first = torch.index_select(out, 0, torch.from_numpy(edge_mask_train[0,:]).long().to(device))
#             nodes_second = torch.index_select(out, 0, torch.from_numpy(edge_mask_train[1,:]).long().to(device))
#             pred = torch.sum(nodes_first * nodes_second, dim=-1)
#             label_positive = torch.ones([data.mask_link_positive_train.shape[1],], dtype=pred.dtype)
#             label_negative = torch.zeros([data.mask_link_negative_train.shape[1],], dtype=pred.dtype)
#             label = torch.cat((label_positive,label_negative)).to(device)
#             loss = loss_func(pred, label)

#             # update
#             loss.backward()
#             if id % args.batch_size == args.batch_size-1:
#                 if args.batch_size>1:
#                     # if this is slow, no need to do this normalization
#                     for p in model.parameters():
#                         if p.grad is not None:
#                             p.grad /= args.batch_size
#                 optimizer.step()
#                 optimizer.zero_grad()

# if __name__ == '__main__':

#     data_list = get_tg_dataset()
#     train()
