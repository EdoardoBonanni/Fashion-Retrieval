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
from src.dataloader import Data, DataQuery, DataQueryManip
from src.model_feature import Extractor, MemoryBlock
from src.argument_parser import add_base_args, add_eval_args
from src.utils import split_labels,  compute_NDCG, get_target_attr
import src.constants as C
import utilities
import ntm_manip
from PIL import Image, ImageFile
from model.model import NTM_extractor
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
        gpu_id = 1
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
    training_model = True
    training_model_path = "best_loss_95"
    reset_memory = True
    num_image_query = 10
    number_of_manipulation = 5
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

    if not training_model:
        print('load ntm memory %s\n' % args.load_init_mem)
        init_weight_memory = np.load(args.load_init_mem)
        init_weight_memory = torch.from_numpy(init_weight_memory).cuda()
        init_weight_memory = init_weight_memory.clone().repeat(batch_size, 1, 1)

        # start training from the pretrained weights if provided
        if args.load_pretrained_extractor:
            print('load %s\n' % args.load_pretrained_extractor)
            if gpu_id == 0:
                model_feature.load_state_dict(torch.load(args.load_pretrained_extractor))
            else:
                model_feature.load_state_dict(torch.load(args.load_pretrained_extractor,  map_location={'cuda:0': 'cuda:1'}))
        else:
            print('Pretrained extractor not provided. Use --load_pretrained_extractor or the model will be randomly initialized.')

        # NTM initialization
        utilities.init_logging()
        # Initialize arguments
        args_ntm = utilities.init_arguments()
        # Initialize random
        utilities.init_seed(args_ntm.seed)
        # Initialize the model
        model_ntm = ntm_manip.init_model(init_weight_memory, batch_size)
        model_ntm.cuda()
        epoch_num = 0

    else:
        print('load ntm memory ' + training_model_path)
        init_weight_memory = np.load(ckpt_dir + "/manip/memory_" + training_model_path + ".npy")[0:batch_size]
        init_weight_memory = torch.from_numpy(init_weight_memory).cuda()

        print('load ' + training_model_path)
        if gpu_id == 0:
            model_feature.load_state_dict(torch.load(ckpt_dir + "/manip/extractor_" + training_model_path + ".pkl"))
        else:
            model_feature.load_state_dict(torch.load(ckpt_dir + "/manip/extractor_" + training_model_path + ".pkl",  map_location={'cuda:0': 'cuda:1'}))

        model_ntm = ntm_manip.init_model(init_weight_memory, batch_size)
        model_ntm.cuda()
        if gpu_id == 0:
            model_ntm.load_state_dict(torch.load(ckpt_dir + "/manip/ntm_" + training_model_path + ".model"))
        else:
            model_ntm.load_state_dict(torch.load(ckpt_dir + "/manip/ntm_" + training_model_path + ".model", map_location={'cuda:0': 'cuda:1'}))

        # model_ntm.memory.init_mem_bias(init_weight_memory)
        epoch_num = training_model_path.split('_')[-1]
        epoch_num = int(epoch_num)


    model_ntm_feature = NTM_extractor(model_feature, model_ntm)
    model_ntm_feature.cuda()
    model_ntm_feature.eval()
    model_ntm_feature.model_ntm.memory.init_mem_bias(init_weight_memory)
    model_ntm_feature.model_ntm.init_sequence(batch_size)
    #indexing the gallery
    gallery_feat = []
    if save_matrix:
        with torch.no_grad():
            for i, (img, _) in enumerate(tqdm(gallery_loader)):
                if not args.use_cpu:
                    img = img.cuda()

                dis_feat, _ = model_ntm_feature.model_feature(img)
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
    query_data = DataQueryManip(file_root, img_root_path,
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
        f1 = open('manipulation_results/method 1/k-' + str(k) + '/results_with_' + str(number_of_manipulation) + '_manipulations_and_' + str(num_image_query) + '_images_reset_k' + str(k) + '.txt', 'w')
    else:
        f1 = open('manipulation_results/method 1/k-' + str(k) + '/results_with_' + str(number_of_manipulation) + '_manipulations_and_' + str(num_image_query) + '_images_k' + str(k) + '.txt', 'w')
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
    all_images = []
    all_feat_manip = []
    final_weights_memory = []
    total_accuracy = []

    print('Forward')
    for z in range(0, num_image_query):
        model_ntm_feature.model_ntm.memory.init_mem_bias(init_weight_memory)
        model_ntm_feature.model_ntm.init_sequence(batch_size)
        print('Image ' + str(z))
        indicators_image = []
        images = []
        feat_manip_array = []

        ref_id = int(query_loader.dataset.ref_idxs[query_loader.dataset.idx_query_images[query_loader.dataset.idx_image_query]])
        img = Image.open(os.path.join(query_loader.dataset.img_root_path, query_loader.dataset.img_data[ref_id]))
        img = img.convert('RGB')

        if query_loader.dataset.img_transform:
            img = query_loader.dataset.img_transform(img)
        if not args.use_cpu:
            img = img.cuda()
        img = torch.reshape(img, (1, img.shape[0], img.shape[1], img.shape[2]))

        image_name = query_loader.dataset.img_data[ref_id]
        index_image_ref = query_loader.dataset.img_data.index(query_loader.dataset.query_images[z])
        label_data_ref = label_data[index_image_ref]
        label_data_ref_copy = deepcopy(label_data_ref)
        images.append(image_name)
        
        with torch.no_grad():
            dis_feat, _ = model_ntm_feature.model_feature(img)
            for i in range(0, number_of_manipulation):
                query_fused_feats = []
                indicator = utilities.create_new_indicator_method1(label_data_ref_copy, categories)
                indicators_image.insert(0, indicator)

                input_memory = None
                residual_feat = ntm_manip.manipulate_ntm(model_ntm_feature, False, indicator, False, input_memory)
                if reset_memory:
                    model_ntm_feature.model_ntm.memory.init_mem_bias(init_weight_memory)
                else:
                    output_memory = ntm_manip.memory_state_tensor(model_ntm_feature)
                    model_ntm_feature.model_ntm.memory.init_mem_bias(output_memory)

                if i == 0:
                    feat_manip = torch.cat(dis_feat, 1) + residual_feat
                    feat_manip_array.insert(0, feat_manip)
                else:
                    feat_manip = feat_manip + residual_feat
                    feat_manip_array.insert(0, feat_manip)

                query_fused_feats.append(F.normalize(feat_manip).cpu().numpy())
                indicator_cpu = indicator.cpu().detach().numpy()

                index_neg = np.where(indicator_cpu[0] == -1)
                index_pos = np.where(indicator_cpu[0] == 1)

                for index_indicator, attribute in enumerate(indicator_cpu[0]):
                    if attribute == -1:
                        label_data_ref_copy[index_indicator] = 0
                    elif attribute == 1:
                        label_data_ref_copy[index_indicator] = 1
                        utilities.check_attributes(label_data_ref_copy, categories, index_indicator)


            labels_images_manipulated.append(label_data_ref)
            final_weights_memory.append(ntm_manip.memory_state_tensor(model_ntm_feature))

        query_loader.dataset.idx_image_query = query_loader.dataset.idx_image_query + 1
        all_images.append(images)
        all_feat_manip.append(feat_manip_array)
        all_indicators.append(indicators_image)

    print('Backward')
    query_loader.dataset.idx_image_query = 0
    for z in range(0, num_image_query):
        accuracy = 0
        if reset_memory:
            input_weight_memory = init_weight_memory
        else:
            input_weight_memory = final_weights_memory[z]
        model_ntm_feature.model_ntm.memory.init_mem_bias(input_weight_memory)
        model_ntm_feature.model_ntm.init_sequence(batch_size)
        if all_images[z][0] is None:
            image_name = 'Image query backward not found in dataset'
        else:
            image_name = 'Image Query ' + str(all_images[z][0])
        print("Image " + str(z))

        label_data_ref = labels_images_manipulated[z]
        label_data_ref_copy = deepcopy(label_data_ref)

        with torch.no_grad():
            for i in range(0, number_of_manipulation):
                query_fused_feats = []
                indicator = all_indicators[z][i]
                for ind, value in enumerate(indicator[0]):
                    if value == 1:
                        indicator[0][ind] = -1
                    elif value == -1:
                        indicator[0][ind] = 1

                input_memory = None
                residual_feat = ntm_manip.manipulate_ntm(model_ntm_feature, False, indicator, False, input_memory)
                if reset_memory:
                    model_ntm_feature.model_ntm.memory.init_mem_bias(init_weight_memory)
                else:
                    output_memory = ntm_manip.memory_state_tensor(model_ntm_feature)
                    model_ntm_feature.model_ntm.memory.init_mem_bias(output_memory)

                if i == 0:
                    feat_manip = all_feat_manip[z][i] + residual_feat
                else:
                    feat_manip = feat_manip + residual_feat

                query_fused_feats.append(F.normalize(feat_manip).cpu().numpy())
                indicator_cpu = indicator.cpu().detach().numpy()

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
            bool_hit = False
            hits = 0
            if z != 0:
                f1.write('\n')
            f1.write('Groundtruth Image: ' + all_images[z][-1] + '\n')

            neighbours_idxs = knn[0]
            f1.write('Image with similar attributes for knn: ' + query_loader.dataset.img_data[neighbours_idxs[0]] + '\n')
            for n_idx in neighbours_idxs:
                label_data_nidx_copy = deepcopy(label_data[n_idx])
                if (label_data[n_idx] == label_data_ref_copy).all():
                    bool_hit = True
                    f1.write('Image with attributes requested: ' + query_loader.dataset.img_data[n_idx] + '\n')
                    hits += 1
                    accuracy += 1
                    break
            if not bool_hit:
                f1.write('Image with attributes requested: not found in top ' + str(k) + '\n')

        query_loader.dataset.idx_image_query = query_loader.dataset.idx_image_query + 1
        total_accuracy.append(accuracy)
        if accuracy == 1:
            f1.write('Groundtruth image found \n')
        else:
            f1.write('Groundtruth image not found \n')
        f1.write('-------------------------------- \n')

    f1.write('\n')
    f1.write('Total accuracy: ' + str(np.mean(total_accuracy)) + ' \n')

    f1.close()
    print(str(np.mean(total_accuracy)))
    print('End')



