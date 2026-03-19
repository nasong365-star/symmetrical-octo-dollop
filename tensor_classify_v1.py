"""
@Project   : HoRM
@Time      : 2024/03/16
@Author    : Elena (nasong1010@163.com)
@Version   : 1.1 (Baseline + Ablation Experiments)
"""

import random
import time
import os
from args import parameter_parser
from util.loadMatData import generate_partition
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from TrustworthyNet import TrustworthyNet_tensor
from util.utils import get_evaluation_results
from util.loadMatData import features_to_Lap, load_data_2
import matplotlib.pyplot as plt
import umap
import umap.plot


def draw_plt_umap(output_tensor, labels_tensor, dataset_name):
    """Project model outputs to 2D with UMAP and save the figure."""
    output_tensor = output_tensor.detach().cpu().numpy()
    labels_tensor = labels_tensor.detach().cpu().numpy()

    # Create tsne directory if it doesn't exist
    os.makedirs('./tsne', exist_ok=True)

    reducer = umap.UMAP(n_neighbors=10, random_state=42)
    embedding = reducer.fit_transform(output_tensor)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels_tensor, s=8, cmap='rainbow')
    handles, _ = scatter.legend_elements(prop='colors')
    label_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    plt.legend(handles, label_names, loc='upper right')
    plt.savefig('./tsne/umap_scatter_' + dataset_name + '.svg')
    plt.close()

    umap.plot.points(reducer, labels=labels_tensor, cmap="rainbow")
    plt.savefig('./tsne/umap_points_' + dataset_name + '.svg')
    plt.close()

def classifier(train_indices, test_indices, num_samples, feature_list, sim_list, lap_list, lap2_list, labels_np,
               feature_dims_per_view, num_views, num_classes, args, device, dataset):
    """Run semi-supervised classification for multiple repeats and report averaged metrics."""

    all_acc = []
    all_f1_micro = []
    all_f1_macro = []
    begin_time = time.time()
    num_repeats = args.num_repeats

    # Convert labels once and keep them on the selected device.
    labels_tensor = torch.from_numpy(labels_np).long().to(device)
    for repeat_idx in range(num_repeats):
        best_acc = 0
        best_f1_macro = 0
        best_f1_micro = 0
        identity_matrix = np.identity(num_samples)
        ones_matrix = np.ones(num_samples)
        phi = ones_matrix - identity_matrix
        model = TrustworthyNet_tensor(feature_dims_per_view, num_views, num_classes, num_samples, phi, args, device).to(device)
        loss_function = torch.nn.NLLLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        if repeat_idx == 0:
            loss_list = []
            performance_history = []

        # Standard train loop.
        with tqdm(total=args.epoch, desc="Training") as pbar:
            for epoch in range(args.epoch):
                model.train()
                model_outputs = model(feature_list, sim_list, lap_list, lap2_list)
                output = model_outputs[-1]
                output = F.softmax(output, dim=1)
                loss_sup = loss_function(F.log_softmax(model_outputs[-1], dim=1)[train_indices],
                                         labels_tensor[train_indices])

                # Entropy-style regularization across class probabilities.
                loss_consis = torch.Tensor(np.array([0])).to(device)

                for class_idx in range(num_classes):
                    qj = 0
                    for item in output[:, class_idx]:
                        qj += item
                    qj /= num_samples
                    # qj = -qj
                    entropy = qj * torch.log(qj)
                    loss_consis += entropy

                loss = 100 * (loss_sup + args.gamma * loss_consis)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Evaluate on the test split each epoch.
                with torch.no_grad():
                    model.eval()
                    output = model(feature_list, sim_list, lap_list, lap2_list)
                    pred_labels = torch.argmax(output[-1], 1).cpu().detach().numpy()
                    acc, f1_macro, f1_micro = get_evaluation_results(labels_tensor.cpu().detach().numpy()[test_indices],
                                                                     pred_labels[test_indices])
                    if repeat_idx == 0:
                        loss_list.append(loss.item())
                        performance_history.append(acc)
                    if acc >= best_acc:
                        best_acc = acc
                        best_f1_macro = f1_macro
                        best_f1_micro = f1_micro
                        # pbar.set_postfix
                    print({'Loss': '{:.6f}'.format(loss.item()),
                           'ACC': '{:.2f} | {:.2f}'.format(acc * 100, best_acc * 100),
                           'F1_macro': '{:.2f} | {:.2f}'.format(f1_macro * 100, best_f1_macro * 100),
                           'F1_micro': '{:.2f} | {:.2f}'.format(f1_micro * 100, best_f1_micro * 100)})
                pbar.update(1)

            output = F.softmax(output[-1], dim=1)
            # plt.savefig('./heatmap/' + dataset + '.svg')
            # plt.show()
            print('times = ', repeat_idx)
            print("------------------------")
            print("ACC:   {:.2f}".format(acc * 100))
            # print("Std:   {:.2f}".format( * 100))
            print("F1_macro:   {:.2f}".format(f1_macro * 100))
            print("F1_micro:   {:.2f}".format(f1_micro * 100))
        draw_plt_umap(output, labels_tensor, dataset)
        all_acc.append(acc)
        all_f1_macro.append(f1_macro)
        all_f1_micro.append(f1_micro)

    # Append aggregated metrics to the result log.
    filename = "Tensor_classify.txt"
    with open(filename, 'a') as f:
        f.write("============ " + str(dataset) + ",lr = " + str(args.lr) + ",seed = " + str(args.seed) + ",gamma = " + str(args.gamma)
                + ",delta = " + str(args.delta)
                + ",beta = " + str(args.beta) + ",alpha = " + str(args.alpha) + ",fusion_type = " + str(args.fusion_type) + '\n')
        f.write(
            "averageACC = " + str(round(np.mean(all_acc), 4) * 100) + ",Std: " + str(round(np.std(all_acc), 4) * 100) + ",F1_macro: " + str(
                np.mean(all_f1_macro) * 100) + ",F1_std: " + str(np.std(all_f1_macro) * 100) + '\n')

    cost_time = time.time() - begin_time
    print("------------------------")
    print("ACC:   {:.2f}".format(np.mean(all_acc) * 100))
    print("Std:   {:.2f}".format(np.std(all_acc) * 100))
    print("F1_macro:   {:.2f}".format(np.mean(all_f1_macro) * 100))
    print("F1_micro:   {:.2f}".format(np.mean(all_f1_micro) * 100))
    print("Classifier cost_time: {:.2f}s ({:.2f}m)".format(cost_time, cost_time / 60))
    print("------------------------")


if __name__ == '__main__':
    # Record total script runtime.
    script_start_time = time.time()
    print("=" * 50)
    print("Script started at:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(script_start_time)))
    print("=" * 50)

    args = parameter_parser()

    dataset_name_map = {1: "ALOI", 2: "animals", 3: "BBC4view", 4: "Caltech101-7", 5: "Caltech101-20", 6: "Caltech101-all",
                7: "MFeat", 8: "COIL", 9: "GRAZ02", 10: "flower17", 11: "HW", 12: "MNIST",13: "MNIST10k",
                14: "MITIndoor", 15: "NUS-WIDE", 16: "Notting-Hill", 17: "ORL", 18: "scene15",19: "small_Reuters",
                20: "UCI", 21: "WebKB_texas", 22: "WikipediaArticles", 23: "Youtube", 24: "citeseer",
                25: "100leaves", 26: "WebKB", 27: "WebKB_washington", 28: "WebKB_wisconsin", 29: "20newsgroups",
                30: "BBC3view",31: "BBCSports", 32: "3sources", 33: "handwritten", 34: "Hdigit", 35: "MSRC-v1",
                36: "NGs", 37: "WebKB_cornell", 38: "OutdoorScene", 39: "Cora", 40: "BBC4view"}

    if args.dataset_id is not None and args.dataset_id not in dataset_name_map:
        raise ValueError(f"Unsupported dataset_id: {args.dataset_id}")

    selected_dataset_ids = [args.dataset_id] if args.dataset_id is not None else [1]
    device = torch.device('cpu' if args.device == 'cpu' else 'cuda:' + args.device)


    # Run experiments dataset by dataset.
    for dataset_id in selected_dataset_ids:

        dataset = dataset_name_map[dataset_id]
        dataset_start_time = time.time()
        print("\n[DATASET START] load {} dataset...".format(dataset))

        feature_list, labels = load_data_2(dataset, args.data_path)
        n_view = len(feature_list)
        n_feats = [x.shape[1] for x in feature_list]
        n_classes = len(np.unique(labels))
        n = feature_list[0].shape[0]
        print("samples:{}, view size:{}, feature dimensions:{}, class:{}".format(n, n_view, n_feats, n_classes))

        print("construct Laplace matrix...")
        sim_list, lap_list, adj_list, lap2_list = features_to_Lap(feature_list, 5)  # int(n / n_classes * 0.1)

        # Move all view-specific tensors to the target device.
        for i in range(n_view):
            feature_list[i] = torch.from_numpy(feature_list[i] / 1.0).float().to(device)
            sim_list[i] = sim_list[i].to_dense().to(device)
            lap_list[i] = lap_list[i].to_dense().to(device)
            lap2_list[i] = lap2_list[i].to_dense().to(device)

        idx_labeled, idx_unlabeled = generate_partition(labels=labels, ratio=args.ratio)
        gamma_values = [args.gamma]
        # Loop over gamma values for controlled experiments.
        for gamma_value in gamma_values:
            args.gamma = gamma_value
            if args.fix_seed:
                torch.cuda.manual_seed(args.seed)
                torch.cuda.manual_seed_all(args.seed)
                random.seed(args.seed)
                np.random.seed(args.seed)
                torch.manual_seed(args.seed)
            print("*" * 40 + "\nfusion:{},active:{},gamma:{},block:{},epoch:{},thre:{},lr:{},"
                             "gamma:{}\n".format(
                args.fusion_type, args.active, args.gamma, args.block, args.epoch, args.thre, args.lr, args.gamma))
            classifier(idx_labeled, idx_unlabeled, n, feature_list, sim_list, lap_list, lap2_list, labels, n_feats, n_view, n_classes,
                   args, device, dataset)
        
        dataset_end_time = time.time()
        dataset_cost_time = dataset_end_time - dataset_start_time
        print("\n[DATASET END] {} - Elapsed time: {:.2f}s ({:.2f}m)".format(
            dataset, dataset_cost_time, dataset_cost_time / 60))
    
    script_end_time = time.time()
    total_cost_time = script_end_time - script_start_time
    print("\n" + "=" * 50)
    print("Script completed at:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(script_end_time)))
    print("Total elapsed time: {:.2f}s ({:.2f}m)".format(total_cost_time, total_cost_time / 60))
    print("=" * 50)
