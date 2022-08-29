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
    training_model = True
    training_model_path = "best_loss_144"
    num_image_query = 500
    number_of_manipulation = 20
    k = 30

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
        model_ntm = ntm_manip.init_model(batch_size)
        model_ntm.cuda()
        epoch_num = 0

    else:
        print('load ' + training_model_path)
        if gpu_id == 0:
            model_feature.load_state_dict(torch.load(ckpt_dir + "/manip/extractor_" + training_model_path + ".pkl", map_location={'cuda:1': 'cuda:0'}))
        else:
            model_feature.load_state_dict(torch.load(ckpt_dir + "/manip/extractor_" + training_model_path + ".pkl",  map_location={'cuda:0': 'cuda:1'}))

        model_ntm = ntm_manip.init_model(batch_size)
        model_ntm.cuda()
        if gpu_id == 0:
            model_ntm.load_state_dict(torch.load(ckpt_dir + "/manip/ntm_" + training_model_path + ".model", map_location={'cuda:1': 'cuda:0'}))
        else:
            model_ntm.load_state_dict(torch.load(ckpt_dir + "/manip/ntm_" + training_model_path + ".model", map_location={'cuda:0': 'cuda:1'}))

        # model_ntm.memory.init_mem_bias(init_weight_memory)
        epoch_num = training_model_path.split('_')[-1]
        epoch_num = int(epoch_num)


    model_ntm_feature = NTM_extractor(model_feature, model_ntm)
    model_ntm_feature.cuda()
    model_ntm_feature.eval()
    model_ntm_feature.model_ntm.memory.init_mem_bias(torch.zeros([batch_size, 12, 340]).cuda())
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
    f1 = open('manipulation_results/k-' + str(k) + '/results_with_' + str(number_of_manipulation) + '_manipulations_and_' + str(num_image_query) + '_images_forward_retrieval_k' + str(k) + '.txt', 'w')
    f2 = open('manipulation_results/k-' + str(k) + '/results_with_' + str(number_of_manipulation) + '_manipulations_and_' + str(num_image_query) + '_images_backward_retrieval_k' + str(k) + '.txt', 'w')
    num_attributes = 0
    categories = []
    for x in query_loader.dataset.attr_num:
        categories.append((num_attributes, num_attributes + x - 1))
        num_attributes = num_attributes + x

    categories_names = ['Category', 'Collar', 'Color', 'Fabric', 'Fastening', 'Fit', 'Gender', 'Neckline', 'Pattern', 'Pocket',
                        'Sleeve', 'Sport']
    dict_categories = dict(zip(categories_names, categories))
    categories_to_check = ['Category', 'Color', 'Gender']
    all_indicators = []
    labels_images_manipulated = []
    all_images = []
    all_feat_manip = []
    final_weights_memory = []
    accuracy_total = []
    for ind_array in range(0, number_of_manipulation):
        accuracy_total.append([])
    accuracy_forward = []
    for ind_array in range(0, number_of_manipulation):
        accuracy_forward.append([])
    accuracy_backward = []
    for ind_array in range(0, number_of_manipulation):
        accuracy_backward.append([])

    print('Forward')
    for z in range(0, num_image_query):
        total_hit = 0
        accuracy = 0
        valid_manipulations = 0

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
            input_memory = torch.zeros([batch_size, 12, 340]).cuda()

            for i in range(0, input_memory.shape[0]):
                for j in range(0, input_memory.shape[1]):
                    input_memory[i][j] = torch.reshape(dis_feat[j][i], (1, input_memory.shape[2])).cuda()

            for i in range(0, number_of_manipulation):
                query_fused_feats = []
                indicator = utilities.create_new_indicator(label_data_ref_copy, categories)
                indicators_image.insert(0, indicator)

                feat_manip = ntm_manip.manipulate_ntm(model_ntm_feature, None, indicator, False, input_memory)
                feat_manip = torch.reshape(feat_manip, (feat_manip.shape[0], feat_manip.shape[1] * feat_manip.shape[2]))
                input_memory = ntm_manip.memory_state_tensor(model_ntm_feature)

                if i == 0:
                    f1.write('Image Query: ' + image_name + '\n')
                feat_manip_array.insert(0, feat_manip)

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
                hits = 0
                f1.write('\n')
                bool_hit = False
                neighbours_idxs = knn[0]
                index_neg = np.where(indicator_cpu[0] == -1)
                index_pos = np.where(indicator_cpu[0] == 1)
                if len(index_neg) > 0:
                    f1.write('Attribute removed: ' + str(mapping_attributes[index_neg[0]][0]) + '\n')
                if len(index_pos) > 0:
                    f1.write('Attribute added: ' + str(mapping_attributes[index_pos[0]][0]) + '\n')
                if len(index_pos) == 0 and len(index_pos) == 0:
                    f1.write('No attributes changed')

                for index_indicator, attribute in enumerate(indicator_cpu[0]):
                    if attribute == -1:
                        label_data_ref_copy[index_indicator] = 0
                    elif attribute == 1:
                        label_data_ref_copy[index_indicator] = 1
                        utilities.check_attributes(label_data_ref_copy, categories, index_indicator)

                f1.write('Image with similar attributes for knn: ' + query_loader.dataset.img_data[neighbours_idxs[0]] + '\n')
                for n_idx in neighbours_idxs:
                    label_data_nidx_copy = deepcopy(label_data[n_idx])
                    if utilities.check_label_data(label_data_ref_copy, label_data_nidx_copy, dict_categories, categories_to_check):
                        # if (label_data[n_idx] == label_data_ref).all():
                        bool_hit = True
                        f1.write('Image with attributes requested: ' + query_loader.dataset.img_data[n_idx] + '\n')
                        images.insert(0, query_loader.dataset.img_data[n_idx])
                        hits += 1
                        total_hit += 1
                        accuracy_total[i].append(1)
                        accuracy_forward[i].append(1)
                        accuracy += 1
                        valid_manipulations += 1
                        break
                if not bool_hit:
                    f1.write('Image with attributes requested: not found in top ' + str(k) + '\n')

                    bool_found_item = False
                    for ind_label, label in enumerate(label_data):
                        label_copy = deepcopy(label)
                        if utilities.check_label_data(label_data_ref_copy, label_copy, dict_categories, categories_to_check):
                            bool_found_item = True
                            f1.write('But, it\'s found in the entire dataset: ' + query_loader.dataset.img_data[ind_label] + '\n')
                            images.insert(0, query_loader.dataset.img_data[ind_label])
                            accuracy_total[i].append(0)
                            accuracy_forward[i].append(0)
                            valid_manipulations += 1
                            break
                    if not bool_found_item:
                        f1.write('Image not found in the entire dataset \n')
                        images.insert(0, None)
            labels_images_manipulated.append(label_data_ref_copy)
            final_weights_memory.append(ntm_manip.memory_state_tensor(model_ntm_feature))

        query_loader.dataset.idx_image_query = query_loader.dataset.idx_image_query + 1
        all_images.append(images)
        all_feat_manip.append(feat_manip_array)
        if valid_manipulations > 0:
            accuracy = accuracy / number_of_manipulation
            f1.write('Accuracy: ' + str(accuracy) + '\n')
        else:
            f1.write('Accuracy not evaluable \n')
        f1.write('-------------------------------- \n')
        all_indicators.append(indicators_image)

    print('Backward')
    query_loader.dataset.idx_image_query = 0
    for z in range(0, num_image_query):
        accuracy = 0
        total_hit = 0
        valid_manipulations = 0
        input_memory = final_weights_memory[z]
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

                feat_manip = ntm_manip.manipulate_ntm(model_ntm_feature, None, indicator, False, input_memory)
                feat_manip = torch.reshape(feat_manip, (feat_manip.shape[0], feat_manip.shape[1] * feat_manip.shape[2]))
                input_memory = ntm_manip.memory_state_tensor(model_ntm_feature)

                if i == 0:
                    f2.write(image_name + '\n')

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
                hits = 0
                f2.write('\n')
                bool_hit = False
                neighbours_idxs = knn[0]
                index_neg = np.where(indicator_cpu[0] == -1)
                index_pos = np.where(indicator_cpu[0] == 1)
                if len(index_neg) > 0:
                    f2.write('Attribute removed: ' + str(mapping_attributes[index_neg[0]][0]) + '\n')
                if len(index_pos) > 0:
                    f2.write('Attribute added: ' + str(mapping_attributes[index_pos[0]][0]) + '\n')
                if len(index_pos) == 0 and len(index_pos) == 0:
                    f2.write('No attributes changed')

                for index_indicator, attribute in enumerate(indicator_cpu[0]):
                    if attribute == -1:
                        label_data_ref_copy[index_indicator] = 0
                    elif attribute == 1:
                        label_data_ref_copy[index_indicator] = 1
                        utilities.check_attributes(label_data_ref_copy, categories, index_indicator)

                f2.write('Image with similar attributes for knn: ' + query_loader.dataset.img_data[neighbours_idxs[0]] + '\n')
                for n_idx in neighbours_idxs:
                    label_data_nidx_copy = deepcopy(label_data[n_idx])
                    if utilities.check_label_data(label_data_ref_copy, label_data_nidx_copy, dict_categories, categories_to_check):
                        # if (label_data[n_idx] == label_data_ref).all():
                        bool_hit = True
                        f2.write('Image with attributes requested: ' + query_loader.dataset.img_data[n_idx] + '\n')
                        accuracy_total[i].append(1)
                        accuracy_backward[i].append(1)
                        hits += 1
                        total_hit += 1
                        accuracy += 1
                        valid_manipulations += 1
                        break
                if not bool_hit:
                    f2.write('Image with attributes requested: not found in top ' + str(k) + '\n')

                    bool_found_item = False
                    for ind_label, label in enumerate(label_data):
                        label_copy = deepcopy(label)
                        if utilities.check_label_data(label_data_ref_copy, label_copy, dict_categories, categories_to_check):
                            bool_found_item = True
                            accuracy_total[i].append(0)
                            accuracy_backward[i].append(0)
                            valid_manipulations += 1
                            f2.write('But, it\'s found in the entire dataset: ' + query_loader.dataset.img_data[ind_label] + '\n')
                            break
                    if not bool_found_item:
                        f2.write('Image not found in the entire dataset \n')

        query_loader.dataset.idx_image_query = query_loader.dataset.idx_image_query + 1
        f2.write('Groundtruth Image: ' + all_images[z][-1] + '\n')
        if valid_manipulations > 0:
            accuracy = accuracy / number_of_manipulation
            f2.write('Accuracy: ' + str(accuracy) + '\n')
        else:
            f2.write('Accuracy not evaluable \n')
        f2.write('-------------------------------- \n')

    print('Results evaluation .....')
    for ind_array in range(0, number_of_manipulation):
        if accuracy_forward[ind_array]:
            f1.write('Forward accuracy manipulation ' + str(ind_array + 1) + ': ' + str(np.mean(accuracy_forward[ind_array])) + ' \n')
            f2.write('Forward accuracy manipulation ' + str(ind_array + 1) + ': ' + str(np.mean(accuracy_forward[ind_array])) + ' \n')
        else:
            f1.write('Forward accuracy manipulation ' + str(ind_array + 1) + ' cannot be evaluated \n')
            f2.write('Forward accuracy manipulation ' + str(ind_array + 1) + ' cannot be evaluated \n')

        if accuracy_backward[ind_array]:
            f1.write('Backward accuracy manipulation ' + str(ind_array + 1) + ': ' + str(np.mean(accuracy_backward[ind_array])) + ' \n')
            f2.write('Backward accuracy manipulation ' + str(ind_array + 1) + ': ' + str(np.mean(accuracy_backward[ind_array])) + ' \n')
        else:
            f1.write('Backward accuracy manipulation ' + str(ind_array + 1) + ' cannot be evaluated \n')
            f2.write('Backward accuracy manipulation ' + str(ind_array + 1) + ' cannot be evaluated \n')

        if accuracy_total[ind_array]:
            f1.write('Total accuracy manipulation ' + str(ind_array + 1) + ': ' + str(np.mean(accuracy_total[ind_array])) + ' \n')
            f2.write('Total accuracy manipulation ' + str(ind_array + 1) + ': ' + str(np.mean(accuracy_total[ind_array])) + ' \n')
        else:
            f1.write('Total accuracy manipulation ' + str(ind_array + 1) + ' cannot be evaluated \n')
            f2.write('Total accuracy manipulation ' + str(ind_array + 1) + ' cannot be evaluated \n')

        f1.write('\n')
        f2.write('\n')

    accuracy_forward_flat = [item for sublist in accuracy_forward for item in sublist if len(sublist) > 0]
    accuracy_backward_flat = [item for sublist in accuracy_backward for item in sublist if len(sublist) > 0]
    accuracy_total_flat = [item for sublist in accuracy_total for item in sublist if len(sublist) > 0]
    f1.write('Forward accuracy: ' + str(np.mean(accuracy_forward_flat)) + ' \n')
    f2.write('Forward accuracy: ' + str(np.mean(accuracy_forward_flat)) + ' \n')
    f1.write('Backward accuracy: ' + str(np.mean(accuracy_backward_flat)) + ' \n')
    f2.write('Backward accuracy: ' + str(np.mean(accuracy_backward_flat)) + ' \n')
    f1.write('Total accuracy: ' + str(np.mean(accuracy_total_flat)) + ' \n')
    f2.write('Total accuracy: ' + str(np.mean(accuracy_total_flat)) + ' \n')
    f1.close()
    f2.close()
    print('End')



