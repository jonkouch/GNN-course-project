import time
import sys
import os
import numpy as np
import warnings
import json
from tqdm import tqdm
import torch.nn as nn

from models.TGAT import TGAT
from models.GraphMixer import GraphMixer

from models.modules import MergeLayer
from utils.utils import set_random_seed, convert_to_gpu, get_parameter_sizes
from utils.utils import get_neighbor_sampler, NegativeEdgeSampler
from utils.evaluate_models_utils import evaluate_model_link_prediction
from utils.DataLoader import get_idx_data_loader, get_link_prediction_data
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import get_link_prediction_args

if __name__ == "__main__":

    warnings.filterwarnings('ignore')

    # get arguments
    args = get_link_prediction_args(args=['--model_name', 'GraphMixer', '--num_epochs', '1'])

    # get data for training, validation and testing
    node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data = \
        get_link_prediction_data(dataset_name=args.dataset_name, val_ratio=args.val_ratio, test_ratio=args.test_ratio)

    # initialize validation and test neighbor sampler to retrieve temporal graph
    full_neighbor_sampler = get_neighbor_sampler(data=full_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                 time_scaling_factor=args.time_scaling_factor, seed=1)

    # initialize negative samplers, set seeds for validation and testing so negatives are the same across different runs
    # in the inductive setting, negatives are sampled only amongst other new nodes
    if args.negative_sample_strategy != 'random':
        val_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids,
                                                   interact_times=full_data.node_interact_times, last_observed_time=train_data.node_interact_times[-1],
                                                   negative_sample_strategy=args.negative_sample_strategy, seed=0)
        new_node_val_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=new_node_val_data.src_node_ids, dst_node_ids=new_node_val_data.dst_node_ids,
                                                            interact_times=new_node_val_data.node_interact_times, last_observed_time=train_data.node_interact_times[-1],
                                                            negative_sample_strategy=args.negative_sample_strategy, seed=1)
        test_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids,
                                                    interact_times=full_data.node_interact_times, last_observed_time=val_data.node_interact_times[-1],
                                                    negative_sample_strategy=args.negative_sample_strategy, seed=2)
        new_node_test_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=new_node_test_data.src_node_ids, dst_node_ids=new_node_test_data.dst_node_ids,
                                                             interact_times=new_node_test_data.node_interact_times, last_observed_time=val_data.node_interact_times[-1],
                                                             negative_sample_strategy=args.negative_sample_strategy, seed=3)
    else:
        val_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids, seed=0)
        new_node_val_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=new_node_val_data.src_node_ids, dst_node_ids=new_node_val_data.dst_node_ids, seed=1)
        test_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids, seed=2)
        new_node_test_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=new_node_test_data.src_node_ids, dst_node_ids=new_node_test_data.dst_node_ids, seed=3)

    # get data loaders
    val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(val_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    new_node_val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(new_node_val_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(test_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    new_node_test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(new_node_test_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)


    val_metric_all_runs, new_node_val_metric_all_runs, test_metric_all_runs, new_node_test_metric_all_runs = [], [], [], []
    for run in range(args.num_runs):

        set_random_seed(seed=run)

        args.seed = run
        args.load_model_name = f'{args.model_name}_seed{args.seed}'
        args.save_result_name = f'{args.negative_sample_strategy}_negative_sampling_{args.model_name}_seed{args.seed}'


        run_start_time = time.time()
        tqdm.write(f"********** Run {run + 1} starts. **********")
        tqdm.write(f'configuration is {args}')

        # create model
        if args.model_name == 'TGAT':
            dynamic_backbone = TGAT(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler,
                                    time_feat_dim=args.time_feat_dim, num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout, device=args.device)
        elif args.model_name == 'GraphMixer':
            dynamic_backbone = GraphMixer(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler,
                                            time_feat_dim=args.time_feat_dim, num_tokens=args.num_neighbors, num_layers=args.num_layers, dropout=args.dropout, device=args.device)
        else:
            raise ValueError(f"Wrong value for model_name {args.model_name}!")
        
        link_predictor = MergeLayer(input_dim1=node_raw_features.shape[1], input_dim2=node_raw_features.shape[1],
                                    hidden_dim=node_raw_features.shape[1], output_dim=1)
        model = nn.Sequential(dynamic_backbone, link_predictor)
        tqdm.write(f'model -> {model}')
        tqdm.write(f'model name: {args.model_name}, #parameters: {get_parameter_sizes(model) * 4} B, '
                    f'{get_parameter_sizes(model) * 4 / 1024} KB, {get_parameter_sizes(model) * 4 / 1024 / 1024} MB.')

        # load the saved model
        load_model_folder = f"./trained_models/{args.model_name}/{args.dataset_name}"
        early_stopping = EarlyStopping(patience=0, save_model_folder=load_model_folder,
                                        save_model_name=args.load_model_name, model_name=args.model_name)
        early_stopping.load_checkpoint(model, map_location='cpu')

        model = convert_to_gpu(model, device=args.device)

        loss_func = nn.BCELoss()

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
                                                                    time_gap=args.time_gap)

        new_node_val_losses, new_node_val_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                                    model=model,
                                                                                    neighbor_sampler=full_neighbor_sampler,
                                                                                    evaluate_idx_data_loader=new_node_val_idx_data_loader,
                                                                                    evaluate_neg_edge_sampler=new_node_val_neg_edge_sampler,
                                                                                    evaluate_data=new_node_val_data,
                                                                                    loss_func=loss_func,
                                                                                    num_neighbors=args.num_neighbors,
                                                                                    time_gap=args.time_gap)



        test_losses, test_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                    model=model,
                                                                    neighbor_sampler=full_neighbor_sampler,
                                                                    evaluate_idx_data_loader=test_idx_data_loader,
                                                                    evaluate_neg_edge_sampler=test_neg_edge_sampler,
                                                                    evaluate_data=test_data,
                                                                    loss_func=loss_func,
                                                                    num_neighbors=args.num_neighbors,
                                                                    time_gap=args.time_gap)



        new_node_test_losses, new_node_test_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                                        model=model,
                                                                                        neighbor_sampler=full_neighbor_sampler,
                                                                                        evaluate_idx_data_loader=new_node_test_idx_data_loader,
                                                                                        evaluate_neg_edge_sampler=new_node_test_neg_edge_sampler,
                                                                                        evaluate_data=new_node_test_data,
                                                                                        loss_func=loss_func,
                                                                                        num_neighbors=args.num_neighbors,
                                                                                        time_gap=args.time_gap)
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
        os.makedirs(save_result_folder, exist_ok=True)
        save_result_path = os.path.join(save_result_folder, f"{args.save_result_name}.json")

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
    os.makedirs(save_result_folder, exist_ok=True)
    save_result_path = os.path.join(save_result_folder, f"{args.save_result_name}.json")

    # Save the results to a JSON file
    with open(save_result_path, 'w') as file:
        json.dump(result_json, file, indent=4)

    # Using tqdm to print the average metrics with standard deviation for runs
    metrics_summary = []
    for metric_name in val_metric_all_runs[0].keys():
        avg = np.mean([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs])
        std = np.std([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs], ddof=1)
        metrics_summary.append(f'Validate {metric_name}: {avg:.4f} ± {std:.4f}')

    for metric_name in new_node_val_metric_all_runs[0].keys():
        avg = np.mean([new_node_val_metric_single_run[metric_name] for new_node_val_metric_single_run in new_node_val_metric_all_runs])
        std = np.std([new_node_val_metric_single_run[metric_name] for new_node_val_metric_single_run in new_node_val_metric_all_runs], ddof=1)
        metrics_summary.append(f'New Node Validate {metric_name}: {avg:.4f} ± {std:.4f}')

    for metric_name in test_metric_all_runs[0].keys():
        avg = np.mean([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs])
        std = np.std([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs], ddof=1)
        metrics_summary.append(f'Test {metric_name}: {avg:.4f} ± {std:.4f}')

    for metric_name in new_node_test_metric_all_runs[0].keys():
        avg = np.mean([new_node_test_metric_single_run[metric_name] for new_node_test_metric_single_run in new_node_test_metric_all_runs])
        std = np.std([new_node_test_metric_single_run[metric_name] for new_node_test_metric_single_run in new_node_test_metric_all_runs], ddof=1)
        metrics_summary.append(f'New Node Test {metric_name}: {avg:.4f} ± {std:.4f}')

    # Print all metrics using tqdm for a nice output
    for line in tqdm(metrics_summary, desc="Metric Summaries"):
        tqdm.write(line)

    sys.exit()