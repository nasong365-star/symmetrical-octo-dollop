"""
@Project   : HoRM
@Time      : 2025/01/12
@Author    : Elena (nasong1010@163.com)
@Version   : 2.0 (Clustering Ablation)

This script keeps loss ablation toggles in the per-view accumulation block.
Default behavior disables Self2 reconstruction loss.
"""

from __future__ import print_function, division

import random
import time

from tqdm import tqdm
from util.loadMatData import features_to_Lap, load_data_2
from args import parameter_parser
from util.clusteringPerformance import StatisticClustering
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from TrustworthyNet import TrustworthyNet_tensor
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def draw_plt(output_, labels, dataset_name):
    x_tsne = TSNE(n_components=2, learning_rate=100, random_state=42).fit_transform(output_)
    plt.figure(figsize=(8, 6))

    scatter = plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c=labels, s=8, cmap='rainbow')
    handles, _ = scatter.legend_elements(prop='colors')
    labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    plt.legend(handles, labels, loc='upper right')
    plt.savefig('./tsne/' + dataset_name + '.svg')
    plt.close()

def clustering(
    n,
    adj_list,
    feature_list,
    sim_list,
    lap_list,
    lap2_list,
    labels,
    n_feats,
    n_view,
    n_classes,
    args,
    device,
    dataset_name,
    used_seed,
    lambda_feature_relation,
    lambda_graph_topology,
    lambda_consistency,
):
    # network architecture
    iden = np.identity(n)
    one = np.ones(n)
    phi = one - iden
    model = TrustworthyNet_tensor(n_feats, n_view, n_classes, n, phi, args, device).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.90, 0.92), eps=0.01, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.3, patience=15,
                                                           min_lr=1e-8)
    start = time.time()
    acc_max = 0.0
    criterion = nn.MSELoss()

    with tqdm(total=args.epoch, desc="Training") as pbar:
        for epoch_idx in range(args.epoch):
            model.train()

            z_list = model(feature_list, sim_list, lap_list, lap2_list)
            output = z_list[-1]
            output = F.softmax(output, dim=1)

            loss_feature_relation_recon = torch.Tensor(np.array([0])).to(device)
            loss_graph_topology_recon = torch.Tensor(np.array([0])).to(device)

            loss_consis = torch.Tensor(np.array([0])).to(device)

            for clu in range(n_classes):
                qj = 0
                for item in output[:, clu]:
                    qj += item
                qj /= n
                entropy = qj * torch.log(qj)
                loss_consis += entropy


            for k in range(n_view):
                for j in range(args.block):
                    loss_feature_relation_recon += criterion(z_list[j].mm(z_list[j].t()), feature_list[k].mm(feature_list[k].t()))
                    # Ablation default: disable Self2 reconstruction loss in v2.
                    # loss_graph_topology_recon += criterion(z_list[j].mm(z_list[j].t()), adj_list[k])

            loss = (
                lambda_feature_relation * loss_feature_relation_recon
                + lambda_graph_topology * loss_graph_topology_recon
                + lambda_consistency * loss_consis
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(loss)

            feature_relation_recon_loss = loss_feature_relation_recon.cpu().detach().numpy()
            graph_topology_recon_loss = loss_graph_topology_recon.cpu().detach().numpy()
            consis_loss = loss_consis.cpu().detach().numpy()
            train_loss = loss.cpu().detach().numpy()

            output_np = output.detach().cpu().numpy()
            [acc, nmi, purity, ari, fscore, precision, recall] = StatisticClustering(output_np, labels, n_classes)

            now = []
            for item in [acc, nmi, purity, ari, fscore, precision, recall]:
                now.append(str(round(item[0] * 100,1)) + "(" + str(round(item[1] * 100,1)) + ")")
            if (acc[0] > acc_max):
                acc_max = acc[0]
                epoch_best = epoch_idx
                best_result = now
            print({
                "Feature recon loss": "{:.6f}".format(feature_relation_recon_loss[0]),
                "Graph recon loss": "{:.6f}".format(graph_topology_recon_loss[0]),
                "Entropy reg loss": "{:.6f}".format(consis_loss[0]),
                'Loss': '{:.6f}'.format(train_loss[0]),
                'ACC': '{:.2f} | {:.2f}'.format(acc[0] * 100, acc_max * 100)
            })
            pbar.update(1)
    draw_plt(output_np, labels, dataset_name)
    end = time.time()
    score_names = ['ACC', 'NMI', 'Purity', 'ARI', 'Fscore', 'Precision', 'Recall']
    filename = "Tensor_cluster.txt"
    with open(filename, 'a') as f:
        f.write("=========== " + str(dataset_name) + ",lr = " + str(args.lr) + ",seed = " + str(used_seed) + ",delta = " + str(args.delta)
            + ",alpha = " + str(args.alpha) + ",beta = " + str(args.beta)
            + ",lambda_feature_relation = " + str(lambda_feature_relation)
            + ",lambda_graph_topology = " + str(lambda_graph_topology)
            + ",lambda_consistency = " + str(lambda_consistency))
        f.write("ACC = " + str(acc_max * 100) + '\n' + "best_score:" + str(dict(zip(score_names, best_result))) + '\n')
    print("------------------------")
    print("best_score:", dict(zip(score_names, best_result)))
    print("epoch_best:", np.around(epoch_best, decimals=4))
    print("running_time:", end - start)
    print("------------------------")

if __name__ == '__main__':

    args = parameter_parser()


    dataset_name_map = {1: '100leaves', 2: '20newsgroups', 3: '3sources', 4: 'ALOI', 5: 'animals',
                6: 'BBC3view', 7: 'BBCSport2view', 8: 'COIL', 9: 'Caltech101-20',
                10: 'Caltech101-7', 11: 'Caltech101-all', 12: 'Cifar10_10k_batch_1', 13: 'citeseer',
                14: 'GRAZ02', 15: 'HW', 16: 'Hdigit', 17: 'MFeat', 18: 'MITIndoor', 19: 'MNIST',
                20: 'MNIST10k', 21: 'MSRC-v1', 22: 'NGs', 23: 'Notting-Hill', 24: 'NoisyMNIST-30000',
                25: 'NUS-WIDE', 26: 'ORL', 27: 'scene15', 28: 'small_Reuters', 29: 'UCI',
                30: 'WebKB_texas', 31: 'Wikipedia', 32: 'Youtube', 33: "flower17", 34: "YaleB", 35: "handwritten",
                36: "Reuters", 37: 'WebKB', 38: "WebKB_washington", 39: "WebKB_cornell", 40: "WebKB_wisconsin",
                41: 'BBC4view'}

    if args.dataset_clustering_id is not None and args.dataset_clustering_id not in dataset_name_map:
        raise ValueError(f"Unsupported dataset_clustering_id: {args.dataset_clustering_id}")

    selected_dataset_ids = [args.dataset_clustering_id] if args.dataset_clustering_id is not None else [3]
    device = torch.device('cpu' if args.device == 'cpu' else 'cuda:' + args.device)

    for dataset_id in selected_dataset_ids:
        dataset_name = dataset_name_map[dataset_id]

        print("load {} dataset...".format(dataset_name))
        feature_list, labels = load_data_2(dataset_name, './datasets/')

        n_view = len(feature_list)
        n_feats = [x.shape[1] for x in feature_list]
        n_classes = len(np.unique(labels))
        n = feature_list[0].shape[0]
        print("samples:{}, view size:{}, feature dimensions:{}, class:{}".format(n, n_view, n_feats, n_classes))

        print("construct Laplace matrix...")
        sim_list, lap_list, adj_list, lap2_list = features_to_Lap(feature_list, 5)
        for i in range(n_view):
            feature_list[i] = torch.from_numpy(feature_list[i] / 1.0).float().to(device)
            sim_list[i] = sim_list[i].to_dense().to(device)
            lap_list[i] = lap_list[i].to_dense().to(device)
            lap2_list[i] = lap2_list[i].to_dense().to(device)
            adj_list[i] = adj_list[i].to_dense().to(device)

        lambda_feature_relation_values = [0.1]
        lambda_graph_topology_values = [0.1]
        lambda_consistency_values = [1]
        weight_decay_values = [0.15]
        for weight_decay_value in weight_decay_values:
            for lambda_graph_topology_value in lambda_graph_topology_values:
                for lambda_feature_relation_value in lambda_feature_relation_values:
                    for lambda_consistency_value in lambda_consistency_values:
                        args.weight_decay = weight_decay_value
                        seed_value = args.clustering_seed if args.clustering_seed is not None else args.seed
                        if args.fix_seed:
                            torch.cuda.manual_seed(seed_value)
                            torch.cuda.manual_seed_all(seed_value)
                            random.seed(seed_value)
                            np.random.seed(seed_value)
                            torch.manual_seed(seed_value)
                            torch.backends.cudnn.deterministic = True
                            torch.backends.cudnn.benchmark = False
                        print("*" * 40 + "\nfusion:{},active:{},gamma:{},block:{},epoch:{},thre:{},"
                                         "lr:{},lambda_feature_relation:{},lambda_graph_topology:{},lambda_consistency:{}\n".format(
                            args.fusion_type, args.active, args.gamma, args.block, args.epoch, args.thre, args.lr,
                            lambda_feature_relation_value, lambda_graph_topology_value, lambda_consistency_value))
                        clustering(
                            n,
                            adj_list,
                            feature_list,
                            sim_list,
                            lap_list,
                            lap2_list,
                            labels,
                            n_feats,
                            n_view,
                            n_classes,
                            args,
                            device,
                            dataset_name,
                            seed_value,
                            lambda_feature_relation_value,
                            lambda_graph_topology_value,
                            lambda_consistency_value,
                        )

