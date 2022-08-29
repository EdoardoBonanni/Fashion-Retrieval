import torch
import numpy as np
import os
import torch.nn as nn
import torch.nn.functional as F
from src.loss_function import hash_labels, TripletSemiHardLoss
from tqdm import tqdm
import faiss, random
import ntm_manip
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot_grad_flow(named_parameters, filename):
    """Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow"""
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if p.requires_grad and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().cpu().detach().numpy())
            max_grads.append(p.grad.abs().max().cpu().detach().numpy())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    layers = [layers[i].replace("weight", "") for i in range(len(layers))]
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation=45)
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.tight_layout()
    # plt.show()
    # plt.savefig(filename + '.png')


def train(train_loader, model_ntm_feature, optimizer, args, train):
    avg_total_loss = 0

    model_ntm_feature.train()
    # list_index_triplets = [index for index in range(0, int(len(train_loader.dataset.triplets)/train_loader.batch_size))]
    # list_index_triplets = random.sample(list_index_triplets, 1000)

    for i, (imgs, one_hots, labels, indicator) in enumerate(tqdm(train_loader)):
        # if i in list_index_triplets:
        a = True
        if a:
            indicator = indicator.float()
            for key in one_hots.keys():
                one_hots[key] = one_hots[key].float()
            if not args.use_cpu:
                for key in imgs.keys():
                    imgs[key] = imgs[key].cuda()
                    one_hots[key] = one_hots[key].cuda()
                    labels[key] = labels[key].cuda()
                indicator = indicator.cuda()

            model_ntm_feature.zero_grad()

            feat_manip, input_memory, feats, cls_outs, cls_outs_manip = model_ntm_feature(indicator, imgs, labels, train_loader, optimizer, args, train)

            # attribute prediction loss
            cls_loss = 0
            for j in range(len(train_loader.dataset.attr_num)):
                for key in imgs.keys():
                    cls_loss += F.cross_entropy(cls_outs[key][j], labels[key][:, j], ignore_index=-1)
                cls_loss += F.cross_entropy(cls_outs_manip[j], labels['pos'][:, j], ignore_index=-1)

            # label_triplet_loss
            hashs = {}
            for key in imgs.keys():
                hashs[key] = hash_labels(labels[key])

            label_triplet_loss = TripletSemiHardLoss(torch.cat((hashs['ref'], hashs['pos'], hashs['neg']), 0),
                                                     torch.cat((F.normalize(torch.cat(feats['ref'], 1)),
                                                                F.normalize(feat_manip),
                                                                F.normalize(torch.cat(feats['neg'], 1))), 0),
                                                     margin=args.margin)

            # manipulation_triplet_loss
            criterion_c = nn.TripletMarginLoss(margin=args.margin)
            manip_triplet_loss = criterion_c(F.normalize(feat_manip),
                                             F.normalize(torch.cat(feats['pos'], 1)),
                                             F.normalize(torch.cat(feats['neg'], 1))
                                             )
            total_loss = args.weight_cls * cls_loss + args.weight_label_trip * label_triplet_loss + args.weight_manip_trip * manip_triplet_loss

            total_loss.backward()

            optimizer.step()
            # model_ntm_feature.model_ntm.memory.init_mem_bias(input_memory)

            # plot_grad_flow(list(model_ntm_feature.named_parameters()), 'Feature_' + str(i))
            avg_total_loss += total_loss.data

    return avg_total_loss / (i+1)


def eval(gallery_loader, query_loader, model_ntm_feature, optimizer, args, file_root, eval):
    model_ntm_feature.eval()

    gt_labels = np.loadtxt(os.path.join(file_root, 'gt_test.txt'), dtype=int)
    # list_index_query = [index for index in range(0, int(len(query_loader.dataset)/query_loader.batch_size))]
    # list_index_query = random.sample(list_index_query, 5)

    gallery_feat = []
    query_fused_feats = []
    with torch.no_grad():
        # indexing the gallery
        for i, (img, _) in enumerate(tqdm(gallery_loader)):
            if not args.use_cpu:
                img = img.cuda()

            dis_feat, _ = model_ntm_feature.model_feature(img)
            gallery_feat.append(F.normalize(torch.cat(dis_feat, 1)).squeeze().cpu().numpy())

        # load the queries
        for i, (img, indicator) in enumerate(tqdm(query_loader)):
            # if i in list_index_query:
            a = True
            if a:
                indicator = indicator.float()
                if not args.use_cpu:
                    img = img.cuda()
                    indicator = indicator.cuda()

                feat_manip, input_memory, feats, cls_outs, cls_outs_manip = model_ntm_feature(indicator, img, None, None, optimizer, args, eval)

                query_fused_feats.append(F.normalize(feat_manip).cpu().numpy())
                # model_ntm_feature.model_ntm.memory.init_mem_bias(input_memory)


    gallery_feat = np.concatenate(gallery_feat, axis=0).reshape(-1, args.dim_chunk * len(gallery_loader.dataset.attr_num))
    fused_feat = np.array(np.concatenate(query_fused_feats, axis=0)).reshape(-1, args.dim_chunk * len(gallery_loader.dataset.attr_num))
    dim = args.dim_chunk * len(gallery_loader.dataset.attr_num)  # dimension
    num_query = fused_feat.shape[0]  # number of queries

    database = gallery_feat
    queries = fused_feat
    index = faiss.IndexFlatL2(dim)
    index.add(database)
    k = 30
    _, knn = index.search(queries, k)

    # load the GT labels for all gallery images
    label_data = np.loadtxt(os.path.join(file_root, 'labels_test.txt'), dtype=int)

    # compute top@k acc
    hits = 0
    for q in tqdm(range(num_query)):
        neighbours_idxs = knn[q]
        for n_idx in neighbours_idxs:
            if (label_data[n_idx] == gt_labels[q]).all():
                hits += 1
                break
    print('Top@{k} accuracy: {acc}'.format(k=k, acc=hits/num_query))

    return hits/num_query