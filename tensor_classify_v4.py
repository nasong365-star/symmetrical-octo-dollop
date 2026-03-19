"""
@Project   : HoRM
@Time      : 2025/01/12
@Last Edited by: Elena (nasong1010@163.com)
@Version   : 4.0 (Convergence Curves)
"""

import random
import time
import os
from datetime import datetime
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
import seaborn as sns


def plot_convergence_curves(loss_list, acc_list, dataset_name, train_acc_list=None, output_dir='./Results/Convergence_Curves'):
    """
    Plot convergence curves: Loss and Accuracy (dual Y-axis).
    
    Parameters:
    - loss_list: list, training loss value for each epoch
    - acc_list: list, test accuracy value for each epoch
    - dataset_name: str, dataset name
    - train_acc_list: list, training accuracy value for each epoch (optional)
    - output_dir: str, output directory
    """
    try:
        print("\nGenerating convergence curves...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create figure and main axis
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Set style
        sns.set_style("whitegrid")
        
        # Plot loss curve on the left Y-axis
        epochs = np.arange(1, len(loss_list) + 1)
        color_loss = '#0052CC'  # vivid blue
        ax1.set_xlabel('Epoch', fontsize=20, fontweight='bold')
        ax1.set_ylabel('Loss', fontsize=20, fontweight='bold', color=color_loss)
        line1 = ax1.plot(epochs, loss_list, color=color_loss, linewidth=3.5, label='Training Loss', marker='o', markersize=5)
        ax1.tick_params(axis='y', labelcolor=color_loss, labelsize=14)
        ax1.tick_params(axis='x', labelsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Create the second Y-axis (right) for accuracy
        ax2 = ax1.twinx()
        color_acc = '#6A1B9A'  # deep violet
        ax2.set_ylabel('Accuracy', fontsize=20, fontweight='bold', color=color_acc)
        line2 = ax2.plot(epochs, acc_list, color='#FF0000', linewidth=3.5, label='Test Accuracy', marker='s', markersize=5)
        ax2.tick_params(axis='y', labelcolor=color_acc, labelsize=14)
        
        # Plot training accuracy if provided
        line3 = []
        if train_acc_list is not None and len(train_acc_list) > 0:
            color_train_acc = '#6A1B9A'  # deep violet
            line3 = ax2.plot(epochs, train_acc_list, color=color_train_acc, linewidth=3.5, label='Training Accuracy', 
                            marker='^', markersize=5, alpha=0.8)
        
        # Set accuracy axis range
        if len(acc_list) > 0:
            all_acc = acc_list.copy()
            if train_acc_list is not None:
                all_acc.extend(train_acc_list)
            acc_min = min(all_acc)
            acc_max = max(all_acc)
            margin = (acc_max - acc_min) * 0.1
            ax2.set_ylim([max(0, acc_min - margin), min(1, acc_max + margin)])
        
        # Add title
        plt.title(f'{dataset_name} - Training Convergence Curves', fontsize=14, fontweight='bold', pad=20)
        
        # Merge legends and place them at the center-right
        lines = line1 + line3 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='center right', 
              fontsize=14, framealpha=0.95, edgecolor='gray', fancybox=True, shadow=True)
        
        fig.tight_layout()
        
        # Save figure
        timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
        output_file = os.path.join(output_dir, f'Convergence_{dataset_name}_{timestamp}.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Convergence curves saved to: {output_file}")
        
        plt.close()
        return True
        
    except Exception as e:
        print(f"ERROR while plotting convergence curves: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def classifier(train_indices, test_indices, num_samples, feature_list, sim_list, lap_list, lap2_list, labels_np,
               feature_dims_per_view, num_views, num_classes, args, device, dataset=None):

    all_acc = []
    all_f1_micro = []
    all_f1_macro = []
    begin_time = time.time()
    num_repeats = args.num_repeats
    labels_tensor = torch.from_numpy(labels_np).long().to(device)
    
    # Store data for convergence curves
    loss_history = []
    acc_history = []
    train_acc_history = []
    
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
            loss_history = []
            performance_history = []
        with tqdm(total=args.epoch, desc="Training") as pbar:
            for epoch in range(args.epoch):
                model.train()
                model_outputs = model(feature_list, sim_list, lap_list, lap2_list)
                output = model_outputs[-1]
                output = F.softmax(output, dim=1)
                loss_sup = loss_function(F.log_softmax(model_outputs[-1], dim=1)[train_indices], labels_tensor[train_indices])

                loss_consis = torch.Tensor(np.array([0])).to(device)

                for clu in range(num_classes):
                    qj = 0
                    for item in output[:, clu]:
                        qj += item
                    qj /= num_samples
                    entropy = qj * torch.log(qj)
                    loss_consis += entropy

                loss = 100 * (loss_sup + args.gamma * loss_consis)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    model.eval()
                    output = model(feature_list, sim_list, lap_list, lap2_list)
                    pred_labels = torch.argmax(output[-1], 1).cpu().detach().numpy()
                    acc, f1_macro, f1_micro = get_evaluation_results(labels_tensor.cpu().detach().numpy()[test_indices],
                                                                     pred_labels[test_indices])
                    # Compute training-set accuracy
                    train_acc, _, _ = get_evaluation_results(labels_tensor.cpu().detach().numpy()[train_indices],
                                                             pred_labels[train_indices])
                    if repeat_idx == 0:
                        performance_history.append(acc)
                        loss_history.append(loss.item())
                        acc_history.append(acc)
                        train_acc_history.append(train_acc)
                    if acc >= best_acc:
                        best_acc = acc
                        best_f1_macro = f1_macro
                        best_f1_micro = f1_micro
                    print({'Loss': '{:.6f}'.format(loss.item()),
                           'ACC': '{:.2f} | {:.2f}'.format(acc * 100, best_acc * 100),
                           'F1_macro': '{:.2f} | {:.2f}'.format(f1_macro * 100, best_f1_macro * 100),
                           'F1_micro': '{:.2f} | {:.2f}'.format(f1_micro * 100, best_f1_micro * 100)})
                pbar.update(1)

            output = F.softmax(output[-1], dim=1)
            # sns.heatmap(output, cmap="BuGn")
            # sns.heatmap(output.mm(output.T), cmap="BuGn")

            # plt.savefig('./heatmap/' + dataset + '.svg')
            # plt.show()
            print('times = ', repeat_idx)
            print("------------------------")
            print("ACC:   {:.2f}".format(acc * 100))
            print("F1_macro:   {:.2f}".format(f1_macro * 100))
            print("F1_micro:   {:.2f}".format(f1_micro * 100))
        
        all_acc.append(acc)
        all_f1_macro.append(f1_macro)
        all_f1_micro.append(f1_micro)

    # ========== Plot convergence curves ==========
    if dataset is not None and len(loss_history) > 0:
        print("\n========== Generating convergence curves ==========")
        plot_convergence_curves(
            loss_list=loss_history,
            acc_list=acc_history,
            train_acc_list=train_acc_history,
            dataset_name=dataset,
            output_dir='./Results/Convergence_Curves'
        )
        print("========== Convergence curves complete ==========\n")

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
    print("cost_time:  {:.2f}s ({:.2f}m)".format(cost_time, cost_time / 60))
    print("------------------------")

if __name__ == '__main__':

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


    for dataset_id in selected_dataset_ids:

        dataset = dataset_name_map[dataset_id]
        print("load {} dataset...".format(dataset))

        feature_list, labels = load_data_2(dataset, args.data_path)
        num_views = len(feature_list)
        feature_dims_per_view = [x.shape[1] for x in feature_list]
        num_classes = len(np.unique(labels))
        num_samples = feature_list[0].shape[0]
        print("samples:{}, view size:{}, feature dimensions:{}, class:{}".format(num_samples, num_views, feature_dims_per_view, num_classes))

        print("construct Laplace matrix...")
        sim_list, lap_list, adj_list, lap2_list = features_to_Lap(feature_list, 5)  # int(num_samples / num_classes * 0.1)

        for i in range(num_views):
            feature_list[i] = torch.from_numpy(feature_list[i] / 1.0).float().to(device)
            sim_list[i] = sim_list[i].to_dense().to(device)
            lap_list[i] = lap_list[i].to_dense().to(device)
            lap2_list[i] = lap2_list[i].to_dense().to(device)

        train_indices, test_indices = generate_partition(labels=labels, ratio=args.ratio)
        gamma_values = [args.gamma]
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
            classifier(train_indices, test_indices, num_samples, feature_list, sim_list, lap_list, lap2_list, labels, feature_dims_per_view, num_views, num_classes,
                   args, device, dataset)

