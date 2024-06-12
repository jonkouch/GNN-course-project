import time
import sys
import os
from tqdm import tqdm
import numpy as np
import warnings
import shutil
import json
import torch
import torch.nn as nn
import argparse


from models.GraphMixer import GraphMixer
from models.TGAT import TGAT
from models.modules import MergeLayer
from models.DyGFormer import DyGFormer

from utils.utils import set_random_seed, convert_to_gpu, get_parameter_sizes, create_optimizer
from utils.utils import get_neighbor_sampler, NegativeEdgeSampler
# from utils.utils import add_edges_and_update_timestamps, update_edge_list_and_timestamps
from utils.evaluate_models_utils import evaluate_model_link_prediction
from utils.metrics import get_link_prediction_metrics
from utils.DataLoader import get_idx_data_loader, get_link_prediction_data
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import get_link_prediction_args
import networkx as nx
import random
from collections import defaultdict
from dynamic_laser.laser import LaserDynamicTransform
import math


def main():

    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser(description='Training link prediction model.')
    parser.add_argument('--model_name', type=str, default='DyGFormer', help='Name of the model to use.')
    parser.add_argument('--filter_loss', type=int, default=1, help='Whether to filter out high-focus nodes and edges.')
    parser.add_argument('--laser_snapshots', type=int, default=0, help='Number of snapshots to use for laser.')
    parser.add_argument('--test_laser_snapshots', type=int, default=0, help='Number of snapshots to use for testing laser.')
    parser.add_argument('--dataset_name', type=str, default='lastfm', help='Name of the dataset to use.')       

    arg = parser.parse_args()
    arg = vars(arg)

    print(arg)

    # get arguments

    args = get_link_prediction_args(args=['--model_name', arg['model_name'], '--num_epochs', '10', '--num_runs', '5', '--dataset_name', arg['dataset_name'],
                                           '--filter_loss', str(arg['filter_loss']), '--drop_node_prob', '1',
                                             '--laser_snapshots', str(arg['laser_snapshots']), '--test_laser_snapshots', str(arg['test_laser_snapshots'])])

    if args.laser_snapshots:
        if args.dataset_name == 'CanParl':
            args.laser_snapshots = 22
        elif args.dataset_name == 'wikipedia':
            args.laser_snapshots = 20
        elif args.dataset_name == 'lastfm':
            args.laser_snapshots = 360

    if args.test_laser_snapshots:
        if args.dataset_name == 'CanParl':
            args.test_laser_snapshots = 8
        elif args.dataset_name == 'wikipedia':
            args.test_laser_snapshots = 9
        elif args.dataset_name == 'lastfm':
            args.test_laser_snapshots = 70
    
    print(f'running with drop_nodes = {args.filter_loss}, prob = {args.drop_node_prob}')
    print(f'add_focus_edges = {args.add_focus_edges}, add_prob = {args.add_probability}')
    print(f'laser_snapshots = {args.laser_snapshots}, test_laser_snapshots = {args.test_laser_snapshots}')


    # get data for training, validation and testing
    node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data = \
        get_link_prediction_data(dataset_name=args.dataset_name, val_ratio=args.val_ratio, test_ratio=args.test_ratio, add_super_node=False)

    # initialize training neighbor sampler to retrieve temporal graph
    train_neighbor_sampler = get_neighbor_sampler(data=train_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                  time_scaling_factor=args.time_scaling_factor, seed=0)

    # initialize validation and test neighbor sampler to retrieve temporal graph
    full_neighbor_sampler = get_neighbor_sampler(data=full_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                 time_scaling_factor=args.time_scaling_factor, seed=1)

    # initialize negative samplers, set seeds for validation and testing so negatives are the same across different runs
    # in the inductive setting, negatives are sampled only amongst other new nodes
    # train negative edge sampler does not need to specify the seed, but evaluation samplers need to do so
    train_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=train_data.src_node_ids, dst_node_ids=train_data.dst_node_ids)
    val_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids, seed=0)
    new_node_val_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=new_node_val_data.src_node_ids, dst_node_ids=new_node_val_data.dst_node_ids, seed=1)
    test_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids, seed=2)
    new_node_test_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=new_node_test_data.src_node_ids, dst_node_ids=new_node_test_data.dst_node_ids, seed=3)

    # get data loaders
    train_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(train_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(val_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    new_node_val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(new_node_val_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(test_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    new_node_test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(new_node_test_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)

    val_metric_all_runs, new_node_val_metric_all_runs, test_metric_all_runs, new_node_test_metric_all_runs = [], [], [], []

    for run in range(3):

        set_random_seed(seed=run)

        args.seed = run
        # name = input('Unique model name: ')
        name = ""
        args.save_model_name = f'{args.model_name}_seed{args.seed}_{name}'

        run_start_time = time.time()
        tqdm.write(f"********** Run {run + 1} starts. **********")
        tqdm.write(f'configuration is {args}')


        if args.model_name == 'GraphMixer':
            dynamic_backbone = GraphMixer(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                          time_feat_dim=args.time_feat_dim, num_tokens=args.num_neighbors, num_layers=args.num_layers, dropout=args.dropout, device=args.device)
            
        elif args.model_name == 'DyGFormer':
            dynamic_backbone = DyGFormer(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                         time_feat_dim=args.time_feat_dim, channel_embedding_dim=args.channel_embedding_dim, patch_size=args.patch_size,
                                         num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout,
                                         max_input_sequence_length=args.max_input_sequence_length, device=args.device)
        
        else:
            raise ValueError(f"Wrong value for model_name {args.model_name}!")
        
        link_predictor = MergeLayer(input_dim1=node_raw_features.shape[1], input_dim2=node_raw_features.shape[1],
                                    hidden_dim=node_raw_features.shape[1], output_dim=1)
        model = nn.Sequential(dynamic_backbone, link_predictor)
        tqdm.write(f'model -> {model}')
        tqdm.write(f'model name: {args.model_name}, #parameters: {get_parameter_sizes(model) * 4} B, '
                    f'{get_parameter_sizes(model) * 4 / 1024} KB, {get_parameter_sizes(model) * 4 / 1024 / 1024} MB.')

        optimizer = create_optimizer(model=model, optimizer_name=args.optimizer, learning_rate=args.learning_rate, weight_decay=args.weight_decay)

        model = convert_to_gpu(model, device=args.device)

        save_model_folder = f"./trained_models/{args.model_name}/{args.dataset_name}"
        shutil.rmtree(save_model_folder, ignore_errors=True)
        os.makedirs(save_model_folder, exist_ok=True)

        early_stopping = EarlyStopping(patience=args.patience, save_model_folder=save_model_folder,
                                       save_model_name=args.save_model_name, model_name=args.model_name)

        loss_func = nn.BCELoss()

        drop_prob = args.drop_node_prob

        lasers = []
        val_lasers = []
        test_lasers = []
        for i in range(args.laser_snapshots):
            G = nx.Graph()
            edges_in_time_window = np.vstack([train_data.src_node_ids[math.floor(i/args.laser_snapshots * len(train_data.src_node_ids)): math.floor((i+1)/args.laser_snapshots * len(train_data.src_node_ids))],
                                                train_data.dst_node_ids[math.floor(i/args.laser_snapshots * len(train_data.src_node_ids)): math.floor((i+1)/args.laser_snapshots * len(train_data.src_node_ids))]]).T
            G.add_edges_from(edges_in_time_window)
            laser = LaserDynamicTransform(G, 3, edges_in_time_window)
            lasers.append(laser)

        for i in range(args.test_laser_snapshots):
            G = nx.Graph()
            edges_in_time_window = np.vstack([val_data.src_node_ids[math.floor(i/args.test_laser_snapshots * len(val_data.src_node_ids)): math.floor((i+1)/args.test_laser_snapshots * len(val_data.src_node_ids))],
                                                val_data.dst_node_ids[math.floor(i/args.test_laser_snapshots * len(val_data.src_node_ids)): math.floor((i+1)/args.test_laser_snapshots * len(val_data.src_node_ids))]]).T
            G.add_edges_from(edges_in_time_window)
            val_laser = LaserDynamicTransform(G, 3, edges_in_time_window)
            val_lasers.append(val_laser)

        for i in range(args.test_laser_snapshots):
            G = nx.Graph()
            edges_in_time_window = np.vstack([test_data.src_node_ids[math.floor(i/args.test_laser_snapshots * len(test_data.src_node_ids)): math.floor((i+1)/args.test_laser_snapshots * len(test_data.src_node_ids))],
                                                test_data.dst_node_ids[math.floor(i/args.test_laser_snapshots * len(test_data.src_node_ids)): math.floor((i+1)/args.test_laser_snapshots * len(test_data.src_node_ids))]]).T
            G.add_edges_from(edges_in_time_window)
            test_laser = LaserDynamicTransform(G, 3, edges_in_time_window)
            test_lasers.append(test_laser)


        
        for epoch in range(args.num_epochs):
            

            model.train()
            # training, only use training graph
            model[0].set_neighbor_sampler(train_neighbor_sampler)


            # store train losses and metrics
            train_losses, train_metrics = [], []
            train_idx_data_loader_tqdm = tqdm(train_idx_data_loader, ncols=120)

            old_laser = -1
            for batch_idx, train_data_indices in enumerate(train_idx_data_loader_tqdm):
                added_edges_indices = []
                batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times = \
                    train_data.src_node_ids[train_data_indices], train_data.dst_node_ids[train_data_indices], \
                    train_data.node_interact_times[train_data_indices]

                _, batch_neg_dst_node_ids = train_neg_edge_sampler.sample(size=len(batch_src_node_ids))
                batch_neg_src_node_ids = batch_src_node_ids

                batch_neg_timestamps = batch_node_interact_times.copy()

                if args.laser_snapshots:
                    curr_laser = (batch_idx * args.laser_snapshots) // len(train_idx_data_loader)

                    if old_laser == -1:
                        old_laser = curr_laser
                        laser = lasers[curr_laser]
                        rewirings = laser.create_rewirings()
                        old_rewirings = rewirings

                    elif curr_laser != old_laser:
                        old_laser = curr_laser
                        laser = lasers[curr_laser]
                        rewirings = laser.create_rewirings()
                        old_rewirings = rewirings
                    
                    else:
                        rewirings = old_rewirings


                    combinations = None
                    for i in range(len(rewirings)):
                        if len(rewirings[i]) == 0:
                            continue
                            
                        # concatenate the combinations
                        if combinations is None:
                            combinations = rewirings[i]
                        else:
                            combinations = np.concatenate((combinations, rewirings[i]), axis=0)

                    high, low = batch_node_interact_times[0], batch_node_interact_times[-1]

                    if high - low > 0:
                        timestamps = np.random.randint(batch_node_interact_times[0], batch_node_interact_times[-1], size=combinations.shape[0])
                    else:
                        timestamps = high * np.ones(combinations.shape[0])
                    to_add = np.random.rand(combinations.shape[0]) < len(batch_node_interact_times)/combinations.shape[0]
                    combinations = combinations[to_add]
                    timestamps = timestamps[to_add]


                    added_edges_indices = []
                    # merge the two lists and keep a mask of the original edges
                    i, j = 0, 0
                    new_src_node_ids, new_dst_node_ids, new_node_interact_times = [], [], []

                    while i < len(batch_src_node_ids) and j < len(combinations):
                        if batch_node_interact_times[i] < timestamps[j]:
                            new_src_node_ids.append(batch_src_node_ids[i])
                            new_dst_node_ids.append(batch_dst_node_ids[i])
                            new_node_interact_times.append(batch_node_interact_times[i])
                            i += 1
                        else:
                            new_src_node_ids.append(combinations[j][0])
                            new_dst_node_ids.append(combinations[j][1])
                            new_node_interact_times.append(timestamps[j])
                            added_edges_indices.append(len(new_src_node_ids) - 1)
                            j += 1

                    while i < len(batch_src_node_ids):
                        new_src_node_ids.append(batch_src_node_ids[i])
                        new_dst_node_ids.append(batch_dst_node_ids[i])
                        new_node_interact_times.append(batch_node_interact_times[i])
                        i += 1

                    while j < len(combinations):
                        new_src_node_ids.append(combinations[j][0])
                        new_dst_node_ids.append(combinations[j][1])
                        new_node_interact_times.append(timestamps[j])
                        added_edges_indices.append(len(new_src_node_ids) - 1)
                        j += 1
                        
                    batch_src_node_ids = np.array(new_src_node_ids)
                    batch_dst_node_ids = np.array(new_dst_node_ids)
                    batch_node_interact_times = np.array(new_node_interact_times)


                loss_mask = torch.tensor([True] * len(batch_src_node_ids), device=args.device)
                if args.laser_snapshots:
                    loss_mask[added_edges_indices] = False

                if args.model_name in ['GraphMixer']:
                    # get temporal embedding of source and destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_src_node_embeddings, batch_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                            dst_node_ids=batch_dst_node_ids,
                                                                            node_interact_times=batch_node_interact_times,
                                                                            num_neighbors=args.num_neighbors,
                                                                            time_gap=args.time_gap)

                    # get temporal embedding of negative source and negative destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                            dst_node_ids=batch_neg_dst_node_ids,
                                                                            node_interact_times=batch_neg_timestamps,
                                                                            num_neighbors=args.num_neighbors,
                                                                            time_gap=args.time_gap)
                
                elif args.model_name in ['DyGFormer']:
                    # get temporal embedding of source and destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_src_node_embeddings, batch_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                          dst_node_ids=batch_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times)

                    # get temporal embedding of negative source and negative destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                          dst_node_ids=batch_neg_dst_node_ids,
                                                                          node_interact_times=batch_neg_timestamps)
                

                # get positive and negative probabilities, shape (batch_size, )
                positive_probabilities = model[1](input_1=batch_src_node_embeddings, input_2=batch_dst_node_embeddings).squeeze(dim=-1).sigmoid()
                positive_probabilities = positive_probabilities[loss_mask]

                negative_probabilities = model[1](input_1=batch_neg_src_node_embeddings, input_2=batch_neg_dst_node_embeddings).squeeze(dim=-1).sigmoid()

                predicts = torch.cat([positive_probabilities, negative_probabilities], dim=0)
                labels = torch.cat([torch.ones_like(positive_probabilities), torch.zeros_like(negative_probabilities)], dim=0)

                loss = loss_func(input=predicts, target=labels)

                # train_losses.append(loss.item())
                # train_metrics.append(get_link_prediction_metrics(predicts=predicts, labels=labels))

                if args.filter_loss:
                    # Identify and zero out high-focus nodes and edges
                    src_node_gradients = torch.autograd.grad(loss, batch_src_node_embeddings, retain_graph=True)[0][loss_mask]
                    src_node_gradient_magnitudes = torch.norm(src_node_gradients, dim=1).cpu().numpy()

                    dst_node_gradients = torch.autograd.grad(loss, batch_dst_node_embeddings, retain_graph=True)[0][loss_mask]
                    dst_node_gradient_magnitudes = torch.norm(dst_node_gradients, dim=1).cpu().numpy()

                    negative_src_node_gradients = torch.autograd.grad(loss, batch_neg_src_node_embeddings, retain_graph=True)[0]
                    negative_src_node_gradient_magnitudes = torch.norm(negative_src_node_gradients, dim=1).cpu().numpy()

                    negative_dst_node_gradients = torch.autograd.grad(loss, batch_neg_dst_node_embeddings, retain_graph=True)[0]
                    negative_dst_node_gradient_magnitudes = torch.norm(negative_dst_node_gradients, dim=1).cpu().numpy()

                    node_gradient_magnitudes = np.concatenate([src_node_gradient_magnitudes, dst_node_gradient_magnitudes,
                                                                negative_src_node_gradient_magnitudes, negative_dst_node_gradient_magnitudes])
                                                                

                    mean_node_gradient = np.mean(node_gradient_magnitudes)
                    std_node_gradient = np.std(node_gradient_magnitudes)
                    node_focus_threshold = mean_node_gradient + 2 * std_node_gradient

                    mean_src_node_gradient = np.mean(src_node_gradient_magnitudes)
                    std_src_node_gradient = np.std(src_node_gradient_magnitudes)
                    src_node_focus_threshold = mean_src_node_gradient + 2 * std_src_node_gradient

                    mean_dst_node_gradient = np.mean(dst_node_gradient_magnitudes)
                    std_dst_node_gradient = np.std(dst_node_gradient_magnitudes)
                    dst_node_focus_threshold = mean_dst_node_gradient + 2 * std_dst_node_gradient

                    high_focus_src_indices = set(np.where(src_node_gradient_magnitudes > src_node_focus_threshold)[0].tolist())
                    high_focus_dst_indices = set(np.where(dst_node_gradient_magnitudes > dst_node_focus_threshold)[0].tolist())
                    high_focus_neg_src_indices = set(np.where(negative_src_node_gradient_magnitudes > node_focus_threshold)[0].tolist())
                    high_focus_neg_dst_indices = set(np.where(negative_dst_node_gradient_magnitudes > node_focus_threshold)[0].tolist())
                    

                if args.filter_loss:
                    # Generate random tensor for the high-focus nodes
                    random_tensor_src = torch.rand(len(high_focus_src_indices), device=labels.device)
                    random_tensor_dst = torch.rand(len(high_focus_dst_indices), device=labels.device)
                    random_tensor_neg_src = torch.rand(len(high_focus_neg_src_indices), device=labels.device)
                    random_tensor_neg_dst = torch.rand(len(high_focus_neg_dst_indices), device=labels.device)

                    # Create mask by comparing random tensor to drop probability
                    mask_high_focus_src = random_tensor_src >= drop_prob
                    mask_high_focus_dst = random_tensor_dst >= drop_prob
                    mask_high_focus_neg_src = random_tensor_neg_src >= drop_prob
                    mask_high_focus_neg_dst = random_tensor_neg_dst >= drop_prob

                    # Create a full mask for labels
                    mask = torch.ones_like(labels, dtype=torch.bool)

                    # Apply mask to high-focus nodes
                    mask[list(high_focus_src_indices)] = mask_high_focus_src
                    mask[list(high_focus_dst_indices)] = mask_high_focus_dst
                    mask[list(high_focus_neg_src_indices)] = mask_high_focus_neg_src
                    mask[list(high_focus_neg_dst_indices)] = mask_high_focus_neg_dst
                    

                    filtered_loss = loss_func(input=predicts[mask], target=labels[mask])
                    


                if args.filter_loss:
                    train_losses.append(filtered_loss.item())
                    train_metrics.append(get_link_prediction_metrics(predicts=predicts[mask], labels=labels[mask]))

                    optimizer.zero_grad()
                    filtered_loss.backward()
                    optimizer.step()

                
                else:
                    train_losses.append(loss.item())
                    train_metrics.append(get_link_prediction_metrics(predicts=predicts, labels=labels))

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                train_idx_data_loader_tqdm.set_description(f'Epoch: {epoch + 1}, train for the {batch_idx + 1}-th batch, train loss: {loss.item()}')


            if not args.test_laser_snapshots:
                test_lasers = None
                val_lasers = None

            val_losses, val_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                     model=model,
                                                                     neighbor_sampler=full_neighbor_sampler,
                                                                     evaluate_idx_data_loader=val_idx_data_loader,
                                                                     evaluate_neg_edge_sampler=val_neg_edge_sampler,
                                                                     evaluate_data=val_data,
                                                                     loss_func=loss_func,
                                                                     num_neighbors=args.num_neighbors,
                                                                     time_gap=args.time_gap,
                                                                     lasers = val_lasers, num_snapshots = args.test_laser_snapshots)


            new_node_val_losses, new_node_val_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                                       model=model,
                                                                                       neighbor_sampler=full_neighbor_sampler,
                                                                                       evaluate_idx_data_loader=new_node_val_idx_data_loader,
                                                                                       evaluate_neg_edge_sampler=new_node_val_neg_edge_sampler,
                                                                                       evaluate_data=new_node_val_data,
                                                                                       loss_func=loss_func,
                                                                                       num_neighbors=args.num_neighbors,
                                                                                       time_gap=args.time_gap, lasers = val_lasers,
                                                                                       num_snapshots = args.test_laser_snapshots)

            
            # Summarize epoch and losses
            epoch_info = (f'Epoch: {epoch + 1}, Learning Rate: {optimizer.param_groups[0]["lr"]}\n'
                        f'Train Loss: {np.mean(train_losses):.4f}, Validate Loss: {np.mean(val_losses):.4f}, '
                        f'New Node Validate Loss: {np.mean(new_node_val_losses):.4f}')
            tqdm.write(epoch_info)

            # Detailed metrics
            metrics_info = ""
            for metric_name in train_metrics[0].keys():
                metrics_info += f'Train {metric_name}: {np.mean([train_metric[metric_name] for train_metric in train_metrics]):.4f}, '
            for metric_name in val_metrics[0].keys():
                metrics_info += f'Validate {metric_name}: {np.mean([val_metric[metric_name] for val_metric in val_metrics]):.4f}, '
            for metric_name in new_node_val_metrics[0].keys():
                metrics_info += f'New Node Validate {metric_name}: {np.mean([new_node_val_metric[metric_name] for new_node_val_metric in new_node_val_metrics]):.4f}, '

            # Trim the trailing comma and space
            if metrics_info.endswith(", "):
                metrics_info = metrics_info[:-2]

            tqdm.write(metrics_info)


            # perform testing once after test_interval_epochs
            if (epoch + 1) % args.test_interval_epochs == 0:
                test_losses, test_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                           model=model,
                                                                           neighbor_sampler=full_neighbor_sampler,
                                                                           evaluate_idx_data_loader=test_idx_data_loader,
                                                                           evaluate_neg_edge_sampler=test_neg_edge_sampler,
                                                                           evaluate_data=test_data,
                                                                           loss_func=loss_func,
                                                                           num_neighbors=args.num_neighbors,
                                                                           time_gap=args.time_gap, lasers = test_lasers,
                                                                           num_snapshots = args.test_laser_snapshots)


                new_node_test_losses, new_node_test_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                                             model=model,
                                                                                             neighbor_sampler=full_neighbor_sampler,
                                                                                             evaluate_idx_data_loader=new_node_test_idx_data_loader,
                                                                                             evaluate_neg_edge_sampler=new_node_test_neg_edge_sampler,
                                                                                             evaluate_data=new_node_test_data,
                                                                                             loss_func=loss_func,
                                                                                             num_neighbors=args.num_neighbors,
                                                                                             time_gap=args.time_gap, lasers = test_lasers,
                                                                                             num_snapshots = args.test_laser_snapshots)

                test_info = f'Test Loss: {np.mean(test_losses):.4f}, New Node Test Loss: {np.mean(new_node_test_losses):.4f}'
                tqdm.write(test_info)

                test_metrics_info = ""
                for metric_name in test_metrics[0].keys():
                    test_metrics_info += f'Test {metric_name}: {np.mean([test_metric[metric_name] for test_metric in test_metrics]):.4f}, '
                for metric_name in new_node_test_metrics[0].keys():
                    test_metrics_info += f'New Node Test {metric_name}: {np.mean([new_node_test_metric[metric_name] for new_node_test_metric in new_node_test_metrics]):.4f}, '

                # Trim the trailing comma and space
                if test_metrics_info.endswith(", "):
                    test_metrics_info = test_metrics_info[:-2]

                tqdm.write(test_metrics_info)


            # select the best model based on all the validate metrics
            val_metric_indicator = []
            for metric_name in val_metrics[0].keys():
                val_metric_indicator.append((metric_name, np.mean([val_metric[metric_name] for val_metric in val_metrics]), True))
            early_stop = early_stopping.step(val_metric_indicator, model)

            if early_stop:
                break

        # load the best model
        early_stopping.load_checkpoint(model)

        # evaluate the best model
        tqdm.write(f'get final performance on dataset {args.dataset_name}...')

        val_losses, val_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                    model=model,
                                                                    neighbor_sampler=full_neighbor_sampler,
                                                                    evaluate_idx_data_loader=val_idx_data_loader,
                                                                    evaluate_neg_edge_sampler=val_neg_edge_sampler,
                                                                    evaluate_data=val_data,
                                                                    loss_func=loss_func,
                                                                    num_neighbors=args.num_neighbors,
                                                                    time_gap=args.time_gap, lasers = val_lasers,
                                                                    num_snapshots = args.test_laser_snapshots)

        new_node_val_losses, new_node_val_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                                    model=model,
                                                                                    neighbor_sampler=full_neighbor_sampler,
                                                                                    evaluate_idx_data_loader=new_node_val_idx_data_loader,
                                                                                    evaluate_neg_edge_sampler=new_node_val_neg_edge_sampler,
                                                                                    evaluate_data=new_node_val_data,
                                                                                    loss_func=loss_func,
                                                                                    num_neighbors=args.num_neighbors,
                                                                                    time_gap=args.time_gap, lasers = val_lasers,
                                                                                    num_snapshots = args.test_laser_snapshots)


        test_losses, test_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                   model=model,
                                                                   neighbor_sampler=full_neighbor_sampler,
                                                                   evaluate_idx_data_loader=test_idx_data_loader,
                                                                   evaluate_neg_edge_sampler=test_neg_edge_sampler,
                                                                   evaluate_data=test_data,
                                                                   loss_func=loss_func,
                                                                   num_neighbors=args.num_neighbors,
                                                                   time_gap=args.time_gap, lasers = test_lasers,
                                                                   num_snapshots = args.test_laser_snapshots)
        

        new_node_test_losses, new_node_test_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                                     model=model,
                                                                                     neighbor_sampler=full_neighbor_sampler,
                                                                                     evaluate_idx_data_loader=new_node_test_idx_data_loader,
                                                                                     evaluate_neg_edge_sampler=new_node_test_neg_edge_sampler,
                                                                                     evaluate_data=new_node_test_data,
                                                                                     loss_func=loss_func,
                                                                                     num_neighbors=args.num_neighbors,
                                                                                     time_gap=args.time_gap, lasers = test_lasers,
                                                                                     num_snapshots = args.test_laser_snapshots)
        # store the evaluation metrics at the current run
        val_metric_dict, new_node_val_metric_dict, test_metric_dict, new_node_test_metric_dict = {}, {}, {}, {}


        tqdm.write(f'validate loss: {np.mean(val_losses):.4f}')
        for metric_name in val_metrics[0].keys():
            average_val_metric = np.mean([val_metric[metric_name] for val_metric in val_metrics])
            tqdm.write(f'validate {metric_name}, {average_val_metric:.4f}')
            val_metric_dict[metric_name] = average_val_metric

            tqdm.write(f'new node validate loss: {np.mean(new_node_val_losses):.4f}')
            for metric_name in new_node_val_metrics[0].keys():
                average_new_node_val_metric = np.mean([new_node_val_metric[metric_name] for new_node_val_metric in new_node_val_metrics])
                tqdm.write(f'new node validate {metric_name}, {average_new_node_val_metric:.4f}')
                new_node_val_metric_dict[metric_name] = average_new_node_val_metric

        tqdm.write(f'test loss: {np.mean(test_losses):.4f}')
        for metric_name in test_metrics[0].keys():
            average_test_metric = np.mean([test_metric[metric_name] for test_metric in test_metrics])
            tqdm.write(f'test {metric_name}, {average_test_metric:.4f}')
            test_metric_dict[metric_name] = average_test_metric

        tqdm.write(f'new node test loss: {np.mean(new_node_test_losses):.4f}')
        for metric_name in new_node_test_metrics[0].keys():
            average_new_node_test_metric = np.mean([new_node_test_metric[metric_name] for new_node_test_metric in new_node_test_metrics])
            tqdm.write(f'new node test {metric_name}, {average_new_node_test_metric:.4f}')
            new_node_test_metric_dict[metric_name] = average_new_node_test_metric

        single_run_time = time.time() - run_start_time
        tqdm.write(f'Run {run + 1} cost {single_run_time:.2f} seconds.')

        val_metric_all_runs.append(val_metric_dict)
        new_node_val_metric_all_runs.append(new_node_val_metric_dict)
        test_metric_all_runs.append(test_metric_dict)
        new_node_test_metric_all_runs.append(new_node_test_metric_dict)



        # save model result
        result_json = {
                "validate metrics": {metric_name: f'{val_metric_dict[metric_name]:.4f}' for metric_name in val_metric_dict},
                "new node validate metrics": {metric_name: f'{new_node_val_metric_dict[metric_name]:.4f}' for metric_name in new_node_val_metric_dict},
                "test metrics": {metric_name: f'{test_metric_dict[metric_name]:.4f}' for metric_name in test_metric_dict},
                "new node test metrics": {metric_name: f'{new_node_test_metric_dict[metric_name]:.4f}' for metric_name in new_node_test_metric_dict}
        }

        result_json = json.dumps(result_json, indent=4)

        save_result_folder = f"./saved_results/{args.model_name}/{args.dataset_name}"
        if args.filter_loss:
            save_result_folder += f'_filtered_{args.drop_node_prob}'
        if args.laser_snapshots:
            save_result_folder += f'_laser_{args.laser_snapshots}'
        if args.test_laser_snapshots:
            save_result_folder += f'_test_laser_{args.test_laser_snapshots}'

        os.makedirs(save_result_folder, exist_ok=True)
        save_result_path = os.path.join(save_result_folder, f"{args.save_model_name}.json")

        with open(save_result_path, 'w') as file:
            file.write(result_json)

    # Assemble the results in a dictionary
    result_json = {
        "validate metrics": {metric_name: f'{np.mean([val_metric[metric_name] for val_metric in val_metric_all_runs]):.4f}'
                            for metric_name in val_metric_all_runs[0].keys()},
        "new node validate metrics": {metric_name: f'{np.mean([new_node_val_metric[metric_name] for new_node_val_metric in new_node_val_metric_all_runs]):.4f}'
                                    for metric_name in new_node_val_metric_all_runs[0].keys()},
        "test metrics": {metric_name: f'{np.mean([test_metric[metric_name] for test_metric in test_metric_all_runs]):.4f}'
                        for metric_name in test_metric_all_runs[0].keys()},
        "new node test metrics": {metric_name: f'{np.mean([new_node_test_metric[metric_name] for new_node_test_metric in new_node_test_metric_all_runs]):.4f}'
                                for metric_name in new_node_test_metric_all_runs[0].keys()}
    }

    # Define the folder path and ensure it exists
    save_result_folder = f"./saved_results/{args.model_name}/{args.dataset_name}"
    if args.filter_loss:
        save_result_folder += f'_filtered_{args.drop_node_prob}'
    if args.laser_snapshots:
        save_result_folder += f'_laser_{args.laser_snapshots}'
    if args.test_laser_snapshots:
        save_result_folder += f'_test_laser_{args.test_laser_snapshots}'    

    os.makedirs(save_result_folder, exist_ok=True)
    save_result_path = os.path.join(save_result_folder, f"{args.save_model_name}.json")

    # Save the results to a JSON file
    with open(save_result_path, 'w') as file:
        json.dump(result_json, file, indent=4)

    # Using tqdm to print the average metrics with standard deviation for runs
    metrics_summary = []
    final_dict = {}
    for metric_name in val_metric_all_runs[0].keys():
        avg = np.mean([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs])
        std = np.std([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs], ddof=1)
        metrics_summary.append(f'Validate {metric_name}: {avg:.4f} ± {std:.4f}')
        final_dict[f'Validate {metric_name}'] = f'{avg:.4f} +- {std:.4f}'

    for metric_name in new_node_val_metric_all_runs[0].keys():
        avg = np.mean([new_node_val_metric_single_run[metric_name] for new_node_val_metric_single_run in new_node_val_metric_all_runs])
        std = np.std([new_node_val_metric_single_run[metric_name] for new_node_val_metric_single_run in new_node_val_metric_all_runs], ddof=1)
        metrics_summary.append(f'New Node Validate {metric_name}: {avg:.4f} ± {std:.4f}')
        final_dict[f'New Node Validate {metric_name}'] = f'{avg:.4f} +- {std:.4f}'

    for metric_name in test_metric_all_runs[0].keys():
        avg = np.mean([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs])
        std = np.std([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs], ddof=1)
        metrics_summary.append(f'Test {metric_name}: {avg:.4f} ± {std:.4f}')
        final_dict[f'Test {metric_name}'] = f'{avg:.4f} +- {std:.4f}'

    for metric_name in new_node_test_metric_all_runs[0].keys():
        avg = np.mean([new_node_test_metric_single_run[metric_name] for new_node_test_metric_single_run in new_node_test_metric_all_runs])
        std = np.std([new_node_test_metric_single_run[metric_name] for new_node_test_metric_single_run in new_node_test_metric_all_runs], ddof=1)
        metrics_summary.append(f'New Node Test {metric_name}: {avg:.4f} ± {std:.4f}')
        final_dict[f'New Node Test {metric_name}'] = f'{avg:.4f} +- {std:.4f}'

    # Print all metrics using tqdm for a nice output
    for line in tqdm(metrics_summary, desc="Metric Summaries"):
        tqdm.write(line)

    save_path = f"./saved_results/{args.model_name}/{args.dataset_name}/{args.model_name}"

    if args.filter_loss:
        save_path += f'_filtered_{args.drop_node_prob}'
    if args.laser_snapshots:
        save_path += f'_laser_{args.laser_snapshots}'
    if args.test_laser_snapshots:
        save_path += f'_test_laser_{args.test_laser_snapshots}'

    json.dump(final_dict, open(save_path + '.json', 'w'), indent=4)
    


if __name__ == "__main__":
    main()