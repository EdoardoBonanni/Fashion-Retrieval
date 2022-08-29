#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK

import ntm_manip, amzn_manip

import logging
import utilities
import torch
import numpy as np
import argparse
import datetime
import json

import src.constants as C
import os
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as transforms
from src.argument_parser import add_base_args, add_train_args
from src.dataloader import DataTriplet, DataQuery, Data
from src.model_feature import Extractor, MemoryBlock
from model.model import NTM_extractor


def main():
    parser = argparse.ArgumentParser()
    add_base_args(parser)
    add_train_args(parser)
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
    # memory_dir = ckpt_dir + "/initialized_memory_block"
    batch_size = 128
    # batch_size = 1 # for testing
    # batch_size = 64 # batch size iniziale
    num_epochs = 200
    print("Batch_size:", str(batch_size) + ", num_epochs:", str(num_epochs))
    train_bool = True
    resume_training = False
    resume_training_path = "best_loss_1"

    if train_bool:
        print("Train")
    else:
        print("Test")

    if resume_training:
        print("Resume Training")

    # load dataset
    print('Loading dataset...')
    train_data = DataTriplet(file_root, img_root_path, args.triplet_file,
                             transforms.Compose([
                                 transforms.Resize((C.TRAIN_INIT_IMAGE_SIZE, C.TRAIN_INIT_IMAGE_SIZE)),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.CenterCrop(C.TARGET_IMAGE_SIZE),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=C.IMAGE_MEAN, std=C.IMAGE_STD)
                             ]), 'train')

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                               num_workers=args.num_threads,
                                               drop_last=True)

    query_data = DataQuery(file_root, img_root_path,
                           'ref_test.txt', 'indfull_test.txt',
                           transforms.Compose([
                               transforms.Resize((C.TARGET_IMAGE_SIZE, C.TARGET_IMAGE_SIZE)),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=C.IMAGE_MEAN, std=C.IMAGE_STD)
                           ]), mode='test')
    query_loader = torch.utils.data.DataLoader(query_data, batch_size=batch_size, shuffle=False,
                                               sampler=torch.utils.data.SequentialSampler(query_data),
                                               num_workers=args.num_threads,
                                               drop_last=False)

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

    # create the folder to save log, checkpoints and args config
    if not ckpt_dir:
        name = datetime.datetime.now().strftime("%m-%d-%H:%M")
    else:
        name = ckpt_dir
    directory = '{name}'.format(name=name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(os.path.join(directory, 'commandline_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    model_feature = Extractor(train_data.attr_num, backbone=args.backbone, dim_chunk=args.dim_chunk)

    if not resume_training:
        # start training from the pretrained weights if provided
        if args.load_pretrained_extractor:
            print('load %s\n' % args.load_pretrained_extractor)
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
        print('load ntm memory ' + resume_training_path)
        init_weight_memory = np.load(directory + "/manip/memory_" + resume_training_path + ".npy")

        print('load ' + resume_training_path)
        model_feature.load_state_dict(torch.load(directory + "/manip/extractor_" + resume_training_path + ".pkl"))

        model_ntm = ntm_manip.init_model(batch_size)
        model_ntm.cuda()
        model_ntm.load_state_dict(torch.load(directory + "/manip/ntm_" + resume_training_path + ".model"))

        # model_ntm.memory.init_mem_bias(init_weight_memory)
        epoch_num = resume_training_path.split('_')[-1]
        epoch_num = int(epoch_num)

    model_ntm_feature = NTM_extractor(model_feature, model_ntm)
    model_ntm_feature.cuda()
    # model_ntm_feature.model_ntm.memory.init_mem_bias(torch.zeros([batch_size, 12, 340]).cuda())
    model_ntm_feature.model_ntm.memory.init_mem_bias(torch.zeros([batch_size, 12, 340]).cuda())
    model_ntm_feature.model_ntm.init_sequence(batch_size)
    init_weight_memory = None

    if not args.use_cpu:
        model_ntm_feature.cuda()

    # optimizer = torch.optim.Adam(list(model_feature.parameters()) + list(model_ntm.net.parameters()), lr=args.lr,
    #                              betas=(args.momentum, 0.999))
    optimizer = torch.optim.Adam(list(model_ntm_feature.parameters()), lr=args.lr,
                                 betas=(args.momentum, 0.999))
    lr_scheduler_local = lr_scheduler.StepLR(optimizer, args.lr_decay_step, args.lr_decay_rate)

    previous_best_avg_test_acc = 0.0
    previous_best_avg_total_loss = 10000000
    for epoch in range(epoch_num, num_epochs):
        current_epoch = epoch + 1
        if train_bool:
            avg_total_loss = amzn_manip.train(train_loader, model_ntm_feature, optimizer, args, True)
        avg_test_acc = amzn_manip.eval(gallery_loader, query_loader, model_ntm_feature, optimizer, args, file_root, False)

        # result record
        if train_bool:
            print('Epoch %d, Cls_loss: %.4f, test_acc: %.4f\n' % (current_epoch, avg_total_loss, avg_test_acc))
        else:
            print('Epoch %d, test_acc: %.4f\n' % (current_epoch, avg_test_acc))

        if train_bool:
            with open(os.path.join(directory, 'log.txt'), 'a') as f:
                f.write('Epoch %d, Cls_loss: %.4f, test_acc: %.4f\n' % (current_epoch, avg_total_loss, avg_test_acc))

            # store parameters
            if current_epoch % 5 == 0:
                torch.save(model_ntm_feature.model_feature.state_dict(), os.path.join(directory, "manip", "extractor_epoch_%d.pkl" % (current_epoch)))
                save_checkpoint(model_ntm_feature.model_ntm, directory + "/manip", current_epoch, False, False)
                print('Saved checkpoints at {dir}/manip/extractor_epoch_{epoch}.pkl, {dir}/manip/ntm_epoch_{epoch}.model'.format(dir=directory,
                                                                                                                       epoch=current_epoch))

            if avg_test_acc > previous_best_avg_test_acc:
                torch.save(model_ntm_feature.model_feature.state_dict(), os.path.join(directory, "manip", "extractor_best_acc_%d.pkl" % (current_epoch)))
                save_checkpoint(model_ntm_feature.model_ntm, directory + "/manip", current_epoch, False, True)
                print('Best model in {dir}/manip/extractor_best_acc_{epoch}.pkl and {dir}/manip/ntm_best_acc_{epoch}.model'.format(dir=directory,
                                                                                                                 epoch=current_epoch))
                previous_best_avg_test_acc = avg_test_acc

            if avg_total_loss < previous_best_avg_total_loss:
                torch.save(model_ntm_feature.model_feature.state_dict(), os.path.join(directory, "manip", "extractor_best_loss_%d.pkl" % (current_epoch)))
                save_checkpoint(model_ntm_feature.model_ntm, directory + "/manip", current_epoch, True, False)
                print('Best model loss in {dir}/manip/extractor_best_loss_{epoch}.pkl and {dir}/manip/ntm_best_loss_{epoch}.model'.format(dir=directory,
                                                                                                                        epoch=current_epoch))
                previous_best_avg_total_loss = avg_total_loss

        lr_scheduler_local.step()


def save_checkpoint(net, path, current_epoch, best_loss, best_acc):
    if best_loss:
        basename = "{}/ntm_best_loss_{}".format(path, current_epoch)
    elif best_acc:
        basename = "{}/ntm_best_acc_{}".format(path, current_epoch)
    else:
        basename = "{}/ntm_epoch_{}".format(path, current_epoch)
    model_fname = basename + ".model"
    torch.save(net.state_dict(), model_fname)


if __name__ == '__main__':
    main()
