# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import argparse
import pickle as pkl
import os, random
import numpy as np
from tqdm import tqdm
import faiss
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from src.dataloader import Data, DataQuery, DataQueryManip, DataQueryManipMethod2
from src.model import Extractor, MemoryBlock
from src.argument_parser import add_base_args, add_eval_args
from src.utils import split_labels,  compute_NDCG, get_target_attr
import src.constants as C
import utilities
from PIL import Image, ImageFile
from copy import deepcopy
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_base_args(parser)
    add_eval_args(parser)
    args = parser.parse_args()
    if not args.use_cpu and not torch.cuda.is_available():
        print('Warning: Using CPU')
        args.use_cpu = True
    else:
        gpu_id = 0
        # torch.cuda.set_device(0)
        # torch.cuda.set_device(1)
        torch.cuda.set_device(gpu_id)

    # file_root = args.file_root
    # img_root_path = args.img_root
    file_root = "splits/Shopping100k"
    img_root_path = "../Shopping100k/Images/"
    ckpt_dir = "models/Shopping100k"
    memory_dir = ckpt_dir + "/initialized_memory_block"
    feat_dir = 'eval_out'
    batch_size = 1
    save_matrix = False
    training_model_path = "best_loss_100"
    reset_memory = False
    num_image_query = 500
    number_of_manipulation = 100
    k = 30

    if reset_memory:
        print('Reset Memory')
    else:
        print('No reset memory')
    print('Number of manipulations: ' + str(number_of_manipulation))

    # load dataset
    print('Loading gallery...')
    gallery_data = Data(file_root, img_root_path,
                        transforms.Compose([
                            transforms.Resize((C.TARGET_IMAGE_SIZE, C.TARGET_IMAGE_SIZE)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=C.IMAGE_MEAN, std=C.IMAGE_STD)
                        ]), mode='test')

    gallery_loader = torch.utils.data.DataLoader(gallery_data, batch_size=batch_size, shuffle=False,
                                                 sampler=torch.utils.data.SequentialSampler(gallery_data),
                                                 num_workers=args.num_threads,
                                                 drop_last=False)

    model_feature = Extractor(gallery_data.attr_num, backbone=args.backbone, dim_chunk=args.dim_chunk)

    model = Extractor(gallery_data.attr_num, backbone=args.backbone, dim_chunk=args.dim_chunk)
    memory = MemoryBlock(gallery_data.attr_num)

    if gpu_id == 0:
        model.load_state_dict(torch.load(ckpt_dir + "/manip/extractor_" + training_model_path + ".pkl"))
    else:
        model.load_state_dict(torch.load(ckpt_dir + "/manip/extractor_" + training_model_path + ".pkl",  map_location={'cuda:0': 'cuda:1'}))

    if gpu_id == 0:
        memory.load_state_dict(torch.load(ckpt_dir + "/manip/memory_" + training_model_path + ".pkl"))
    else:
        memory.load_state_dict(torch.load(ckpt_dir + "/manip/memory_" + training_model_path + ".pkl",  map_location={'cuda:0': 'cuda:1'}))

    if not os.path.exists(args.feat_dir):
        os.makedirs(args.feat_dir)

    if not args.use_cpu:
        model.cuda()
        memory.cuda()

    model.eval()
    memory.eval()

    #indexing the gallery
    gallery_feat = []
    if save_matrix:
        with torch.no_grad():
            for i, (img, _) in enumerate(tqdm(gallery_loader)):
                if not args.use_cpu:
                    img = img.cuda()

                dis_feat, _ = model(img)
                gallery_feat.append(F.normalize(torch.cat(dis_feat, 1)).squeeze().cpu().numpy())

            # np.save(os.path.join(feat_dir, 'gallery_feats_multiple_manip.npy'), np.concatenate(gallery_feat, axis=0))
            with open(os.path.join(feat_dir, 'gallery_feats_multiple_manip.pkl'), 'wb') as handle:
                pkl.dump(gallery_feat, handle, protocol=pkl.HIGHEST_PROTOCOL)
            print('Saved indexed features at {dir}/gallery_feats.npy'.format(dir=feat_dir))
    else:
        # gallery_feat = np.load(feat_dir + "/gallery_feats_multiple_manip.npy")
        with open(os.path.join(feat_dir, 'gallery_feats_multiple_manip.pkl'), 'rb') as handle:
            gallery_feat = pkl.load(handle)

    init_gallery_feat = deepcopy(gallery_feat)
    #indexing the query
    query_inds = np.loadtxt(os.path.join(file_root, args.query_inds), dtype=int)
    gt_labels = np.loadtxt(os.path.join(file_root, args.gt_labels), dtype=int)
    ref_idxs = np.loadtxt(os.path.join(file_root, args.ref_ids), dtype=int)

    assert (query_inds.shape[0] == gt_labels.shape[0]) and (query_inds.shape[0] == ref_idxs.shape[0])

    print('Loading test queries...')
    query_data = DataQueryManipMethod2(file_root, img_root_path,
                           args.ref_ids, args.query_inds,
                           transforms.Compose([
                               transforms.Resize((C.TARGET_IMAGE_SIZE, C.TARGET_IMAGE_SIZE)),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=C.IMAGE_MEAN, std=C.IMAGE_STD)
                           ]), num_image_query, mode='test')
    query_loader = torch.utils.data.DataLoader(query_data, batch_size=batch_size, shuffle=False,
                                               sampler=torch.utils.data.SequentialSampler(query_data),
                                               num_workers=args.num_threads,
                                               drop_last=False)

    label_data = np.loadtxt(os.path.join(file_root, 'labels_test.txt'), dtype=int)
    mapping_attributes = np.loadtxt(os.path.join(file_root, 'mapping_attributes.txt'), dtype=str)
    if reset_memory:
        f1 = open('manipulation_results/method 2/k-' + str(k) + '/results_with_' + str(number_of_manipulation) + '_manipulations_and_' + str(num_image_query) + '_images_reset_k' + str(k) + '.txt', 'w')
    else:
        f1 = open('manipulation_results/method 2/k-' + str(k) + '/results_with_' + str(number_of_manipulation) + '_manipulations_and_' + str(num_image_query) + '_images_k' + str(k) + '.txt', 'w')
    num_attributes = 0
    categories = []
    for x in query_loader.dataset.attr_num:
        categories.append((num_attributes, num_attributes + x - 1))
        num_attributes = num_attributes + x

    categories_names = ['Category', 'Collar', 'Color', 'Fabric', 'Fastening', 'Fit', 'Gender', 'Neckline', 'Pattern', 'Pocket',
                        'Sleeve', 'Sport']
    dict_categories = dict(zip(categories_names, categories))
    all_indicators = []
    labels_images_manipulated = []
    all_images_query = []
    all_images_target = []
    total_accuracy = []

    print('Forward')
    for z in range(0, num_image_query):
        accuracy = 0
        if gpu_id == 0:
            memory.load_state_dict(torch.load(ckpt_dir + "/manip/memory_" + training_model_path + ".pkl"))
        else:
            memory.load_state_dict(torch.load(ckpt_dir + "/manip/memory_" + training_model_path + ".pkl",  map_location={'cuda:0': 'cuda:1'}))

        print('Image ' + str(z))
        images_query = []
        images_target = []

        # img query
        ref_id_query = int(query_loader.dataset.ref_idxs[query_loader.dataset.idx_query_images[query_loader.dataset.idx_image_query]])
        img_query = Image.open(os.path.join(query_loader.dataset.img_root_path, query_loader.dataset.img_data[ref_id_query]))
        img_query = img_query.convert('RGB')

        if query_loader.dataset.img_transform:
            img_query = query_loader.dataset.img_transform(img_query)
        if not args.use_cpu:
            img_query = img_query.cuda()
        img_query = torch.reshape(img_query, (1, img_query.shape[0], img_query.shape[1], img_query.shape[2]))

        image_name_query = query_loader.dataset.img_data[ref_id_query]
        index_image_query = query_loader.dataset.img_data.index(query_loader.dataset.query_images[z])
        label_data_query = label_data[index_image_query]
        label_data_query_manipulated = deepcopy(label_data_query)
        images_query.append(image_name_query)

        # img target
        ref_id_target = int(query_loader.dataset.ref_idxs[query_loader.dataset.idx_target_images[query_loader.dataset.idx_image_query]])
        img_target = Image.open(os.path.join(query_loader.dataset.img_root_path, query_loader.dataset.img_data[ref_id_target]))
        img_target = img_target.convert('RGB')

        if query_loader.dataset.img_transform:
            img_target = query_loader.dataset.img_transform(img_target)
        if not args.use_cpu:
            img_target = img_target.cuda()
        img_target = torch.reshape(img_target, (1, img_target.shape[0], img_target.shape[1], img_target.shape[2]))

        image_name_target = query_loader.dataset.img_data[ref_id_target]
        index_image_target = query_loader.dataset.img_data.index(query_loader.dataset.target_images[z])
        label_data_target = label_data[index_image_target]
        images_target.append(image_name_target)
        f1.write('Query image: ' + image_name_query + '\n')
        f1.write('Target image: ' + image_name_target + '\n')
        
        with torch.no_grad():
            dis_feat, _ = model(img_query)
            for i in range(0, number_of_manipulation):
                query_fused_feats = []

                indicator = utilities.create_new_indicator_method1(label_data_query_manipulated, categories)
                indicator = indicator.float()

                residual_feat = memory(indicator)
                if i == 0:
                    feat_manip = torch.cat(dis_feat, 1) + residual_feat
                else:
                    feat_manip = feat_manip + residual_feat

                if reset_memory:
                    if gpu_id == 0:
                        memory.load_state_dict(torch.load(ckpt_dir + "/manip/memory_" + training_model_path + ".pkl"))
                    else:
                        memory.load_state_dict(torch.load(ckpt_dir + "/manip/memory_" + training_model_path + ".pkl",  map_location={'cuda:0': 'cuda:1'}))

                query_fused_feats.append(F.normalize(feat_manip).cpu().numpy())
                indicator_cpu = indicator.cpu().detach().numpy()

                index_neg = np.where(indicator_cpu[0] == -1)
                index_pos = np.where(indicator_cpu[0] == 1)
                f1.write('Manipulation ' + str(i + 1) + '\n')
                if np.size(index_neg) > 0:
                    f1.write('Attribute removed: ' + str(mapping_attributes[index_neg[0]][0]) + '\n')
                if np.size(index_pos) > 0:
                    f1.write('Attribute added: ' + str(mapping_attributes[index_pos[0]][0]) + '\n')
                if np.size(index_neg) == 0 and np.size(index_pos) == 0:
                    f1.write('No attributes changed')
                f1.write('\n')

                for index_indicator, attribute in enumerate(indicator_cpu[0]):
                    if attribute == -1:
                        label_data_query_manipulated[index_indicator] = 0
                    elif attribute == 1:
                        label_data_query_manipulated[index_indicator] = 1
                        utilities.check_attributes(label_data_query_manipulated, categories, index_indicator)

            indicator_necessary_transformations, categories_to_modify = utilities.find_necessary_transformations(label_data_query_manipulated, label_data_target, categories)
            for i in range(0, len(indicator_necessary_transformations)):
                query_fused_feats = []

                indicator = indicator_necessary_transformations[i]
                indicator = indicator.float()

                residual_feat = memory(indicator)
                feat_manip = feat_manip + residual_feat

                if reset_memory:
                    if gpu_id == 0:
                        memory.load_state_dict(torch.load(ckpt_dir + "/manip/memory_" + training_model_path + ".pkl"))
                    else:
                        memory.load_state_dict(torch.load(ckpt_dir + "/manip/memory_" + training_model_path + ".pkl",  map_location={'cuda:0': 'cuda:1'}))

                query_fused_feats.append(F.normalize(feat_manip).cpu().numpy())
                indicator_cpu = indicator.cpu().detach().numpy()

                index_neg = np.where(indicator_cpu[0] == -1)
                index_pos = np.where(indicator_cpu[0] == 1)

                f1.write('Manipulation ' + str(number_of_manipulation + i + 1) + '\n')
                if np.size(index_neg) > 0:
                    f1.write('Attribute removed: ' + str(mapping_attributes[index_neg[0]][0]) + '\n')
                if np.size(index_pos) > 0:
                    f1.write('Attribute added: ' + str(mapping_attributes[index_pos[0]][0]) + '\n')
                if np.size(index_neg) == 0 and np.size(index_pos) == 0:
                    f1.write('No attributes changed')
                f1.write('\n')

                for index_indicator, attribute in enumerate(indicator_cpu[0]):
                    if attribute == -1:
                        label_data_query_manipulated[index_indicator] = 0
                    elif attribute == 1:
                        label_data_query_manipulated[index_indicator] = 1
                        utilities.check_attributes(label_data_query_manipulated, categories, index_indicator)

            #evaluate the top@k results
            gallery_feat = np.concatenate(init_gallery_feat, axis=0).reshape(-1, args.dim_chunk * len(gallery_data.attr_num))
            fused_feat = np.array(np.concatenate(query_fused_feats, axis=0)).reshape(-1, args.dim_chunk * len(gallery_data.attr_num))
            dim = args.dim_chunk * len(gallery_data.attr_num)  # dimension
            num_database = gallery_feat.shape[0]  # number of images in database
            num_query = fused_feat.shape[0]  # number of queries

            database = gallery_feat
            queries = fused_feat
            index = faiss.IndexFlatL2(dim)
            index.add(database)
            _, knn = index.search(queries, k)

            #load the GT labels for all gallery images

            #compute top@k acc
            hits = 0
            bool_hit = False
            neighbours_idxs = knn[0]

            if (label_data_query_manipulated == label_data_target).all():
                f1.write('Label manipulated correctly \n')
            else:
                f1.write('Error in label manipulation \n')

            f1.write('Image with similar attributes for knn: ' + query_loader.dataset.img_data[neighbours_idxs[0]] + '\n')
            for n_idx in neighbours_idxs:
                label_data_nidx_copy = deepcopy(label_data[n_idx])
                if (label_data[n_idx] == label_data_target).all():
                    bool_hit = True
                    hits += 1
                    accuracy += 1
                    break

        query_loader.dataset.idx_image_query = query_loader.dataset.idx_image_query + 1
        total_accuracy.append(accuracy)
        f1.write('Query image: ' + image_name_query + '\n')
        f1.write('Target image: ' + image_name_target + '\n')
        if accuracy == 1:
            f1.write('Target image found \n')
        else:
            f1.write('Target image not found \n')
        f1.write('-------------------------------- \n')

    f1.write('\n')
    f1.write('Total accuracy: ' + str(np.mean(total_accuracy)) + ' \n')

    f1.close()
    print(str(np.mean(total_accuracy)))
    print('End')



