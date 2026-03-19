"""
@Project   : HoRM
@Time      : 2025/01/12
@Last Edited by: Na Song(nasong1010@163.com)
@Version   : 3.0 (Enhanced t-SNE 3D Visualization)
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
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE

def visualize_tsne_3d(features, labels, dataset_name, accuracy, output_dir='./Results/Visualizations_tsne'):
    """
    Apply t-SNE to visualize fused multi-view features in 3D.

    Parameters:
    - features: numpy array, shape (n_samples, hidden_size) - fused feature matrix
    - labels: numpy array, shape (n_samples,) - sample labels
    - dataset_name: str, dataset name
    - accuracy: float, classification accuracy (used in plot title)
    - output_dir: str, output directory
    """
    try:
        print("\nRunning t-SNE 3D visualization...")

        # Ensure inputs are numpy arrays
        if torch.is_tensor(features):
            features = features.cpu().detach().numpy()
        if torch.is_tensor(labels):
            labels = labels.cpu().detach().numpy()
        
        # Ensure features is a 2D array
        if len(features.shape) == 1:
            features = features.reshape(-1, 1)

        # Apply t-SNE (3D)
        print(f"Feature shape: {features.shape}")
        print("Reducing to 3D with t-SNE (this may take a minute)...")
        
        perplexity_val = min(30, features.shape[0] // 3)
        tsne = TSNE(n_components=3, random_state=42, perplexity=perplexity_val, 
                   max_iter=1000, verbose=0)
        features_3d = tsne.fit_transform(features)
        
        # Create 3D visualization
        print("Creating 3D visualization...")
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Get unique labels and color palette (ColorBrewer)
        unique_labels = np.unique(labels)

        # ColorBrewer Set1 (9 colors) - recommended for Nature/Science
        colorbrewer_set1 = [
            (0.89, 0.10, 0.11),  # red
            (0.22, 0.49, 0.72),  # blue
            (0.30, 0.69, 0.29),  # green
            (0.60, 0.31, 0.64),  # purple
            (1.00, 0.50, 0.00),  # orange
            (1.00, 1.00, 0.20),  # yellow
            (0.65, 0.33, 0.16),  # brown
            (0.97, 0.51, 0.75),  # pink
            (0.63, 0.63, 0.63),  # grey
        ]
        
        # ColorBrewer Set2 (8 colors) - for more than 9 classes
        if len(unique_labels) > 9:
            colorbrewer_set2 = [
                (0.40, 0.76, 0.65),  # teal
                (0.99, 0.55, 0.38),  # orange
                (0.55, 0.63, 0.80),  # blue
                (0.91, 0.30, 0.24),  # red
                (0.63, 0.85, 0.90),  # light blue
                (1.00, 0.85, 0.18),  # yellow
                (0.70, 0.70, 0.70),  # grey
                (0.80, 0.73, 0.55),  # tan
            ]
            colors = [colorbrewer_set2[i % len(colorbrewer_set2)] for i in range(len(unique_labels))]
        else:
            colors = [colorbrewer_set1[i % len(colorbrewer_set1)] for i in range(len(unique_labels))]
        
        # Plot scatter for each class
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(features_3d[mask, 0], features_3d[mask, 1], features_3d[mask, 2],
                      c=[colors[i]], alpha=0.7, s=30, edgecolors='k', linewidth=0.3,
                      label=f'Class {int(label)}')
        
        # Get axis ranges
        x_min, x_max = features_3d[:, 0].min(), features_3d[:, 0].max()
        y_min, y_max = features_3d[:, 1].min(), features_3d[:, 1].max()
        z_min, z_max = features_3d[:, 2].min(), features_3d[:, 2].max()

        # Add axis lines
        ax.plot([x_min, x_max], [y_min, y_min], [z_min, z_min], 'r-', linewidth=2, alpha=0.6)  # X axis
        ax.plot([x_min, x_min], [y_min, y_max], [z_min, z_min], 'g-', linewidth=2, alpha=0.6)  # Y axis
        ax.plot([x_min, x_min], [y_min, y_min], [z_min, z_max], 'b-', linewidth=2, alpha=0.6)  # Z axis

        # Add grid lines
        x_ticks = np.linspace(x_min, x_max, 5)
        y_ticks = np.linspace(y_min, y_max, 5)
        z_ticks = np.linspace(z_min, z_max, 5)

        # Grid lines along X
        for y_val in y_ticks:
            for z_val in z_ticks:
                ax.plot([x_min, x_max], [y_val, y_val], [z_val, z_val], 'gray',
                       linewidth=0.5, alpha=0.2)

        # Grid lines along Y
        for x_val in x_ticks:
            for z_val in z_ticks:
                ax.plot([x_val, x_val], [y_min, y_max], [z_val, z_val], 'gray',
                       linewidth=0.5, alpha=0.2)

        # Grid lines along Z
        for x_val in x_ticks:
            for y_val in y_ticks:
                ax.plot([x_val, x_val], [y_val, y_val], [z_min, z_max], 'gray',
                       linewidth=0.5, alpha=0.2)

        # Set axis limits
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_zlim([z_min, z_max])

        # Remove tick labels
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        # Set pane background
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        
        ax.xaxis.pane.set_edgecolor('lightgray')
        ax.yaxis.pane.set_edgecolor('lightgray')
        ax.zaxis.pane.set_edgecolor('lightgray')
        
        ax.xaxis.pane.set_alpha(0.1)
        ax.yaxis.pane.set_alpha(0.1)
        ax.zaxis.pane.set_alpha(0.1)
        
        # Add title
        plt.title(f'{dataset_name} - t-SNE 3D Fused Features (ACC: {accuracy:.2f}%)',
                 fontsize=14, pad=20)

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Save figure
        timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
        output_file = os.path.join(output_dir, f'tSNE_3D_{dataset_name}_{timestamp}.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"t-SNE 3D visualization saved to: {output_file}")
        
        plt.close()
        return True
        
    except Exception as e:
        print(f"ERROR during t-SNE visualization: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def classifier(train_indices, test_indices, num_samples, feature_list, sim_list, lap_list, lap2_list, labels_np, feature_dims_per_view, num_views, num_classes, args, device, dataset):

    all_acc = []
    all_f1_micro = []
    all_f1_macro = []
    begin_time = time.time()
    num_repeats = args.num_repeats
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
        if (repeat_idx == 0):
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
                    if repeat_idx == 0:
                        performance_history.append(acc)
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
        
        # ========== t-SNE 3D Visualization ==========
        # Visualize fused multi-view features from model output.
        # Each color represents a different class label.
        print("\n========== Generating t-SNE 3D visualization ==========")
        visualize_tsne_3d(
            features=output.detach(),
            labels=labels_tensor,
            dataset_name=dataset,
            accuracy=acc * 100,
            output_dir='./Results/Visualizations_tsne'
        )
        print("========== t-SNE 3D visualization complete ==========\n")
        
        all_acc.append(acc)
        all_f1_macro.append(f1_macro)
        all_f1_micro.append(f1_micro)

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
    print("cost_time:  {:.2f}".format(cost_time))
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
