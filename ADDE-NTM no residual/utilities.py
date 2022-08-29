#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK

import argparse
import logging
import time
import random
import re
import sys

import attr
import argcomplete
import torch
import numpy as np


LOGGER = logging.getLogger(__name__)

# Default values for program arguments
RANDOM_SEED = 1000

def get_ms():
    """Returns the current time in miliseconds."""
    return time.time() * 1000


def init_seed(seed=None):
    """Seed the RNGs for predicatability/reproduction purposes."""
    if seed is None:
        seed = int(get_ms() // 1000)

    LOGGER.info("Using seed=%d", seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def progress_clean():
    """Clean the progress bar."""
    print("\r{}".format(" " * 80), end='\r')


def clip_grads(net):
    """Gradient clipping to the range [-10, 10]."""
    parameters = list(filter(lambda p: p.grad is not None, net.parameters()))
    for p in parameters:
        p.grad.data.clamp_(-10, 10)

def init_arguments():
    parser = argparse.ArgumentParser(prog='memory_manipulation.py')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED, help="Seed value for RNGs")
    parser.add_argument('-p', '--param', action='append', default=[],
                        help='Override model params. Example: "-pbatch_size=4 -pnum_heads=2"')

    argcomplete.autocomplete(parser)

    args = parser.parse_args()

    return args


def update_model_params(params, update):
    """Updates the default parameters using supplied user arguments."""

    update_dict = {}
    for p in update:
        m = re.match("(.*)=(.*)", p)
        if not m:
            LOGGER.error("Unable to parse param update '%s'", p)
            sys.exit(1)

        k, v = m.groups()
        update_dict[k] = v

    try:
        params = attr.evolve(params, **update_dict)
    except TypeError as e:
        LOGGER.error(e)
        LOGGER.error("Valid parameters: %s", list(attr.asdict(params).keys()))
        sys.exit(1)

    return params


def init_logging():
    logging.basicConfig(format='[%(asctime)s] [%(levelname)s] [%(name)s]  %(message)s',
                        level=logging.DEBUG)


def check_attributes(label_data_ref, categories, index_manipulation_indicator):
    for i in range(0, len(categories)):
        if categories[i][0] <= index_manipulation_indicator <= categories[i][1]:
            number_of_ones_attribute = np.sum(np.array(label_data_ref[categories[i][0]:categories[i][1] + 1]))
            if number_of_ones_attribute >= 2:
                for j in range(categories[i][0], categories[i][1] + 1):
                    if j != index_manipulation_indicator:
                        label_data_ref[j] = 0
            return

def create_new_indicator(label_ref, categories):
    ones_index = np.where(label_ref == 1)[0]
    categories_of_image = []
    # categories 0, 2
    for i in range(0, len(categories)):
        if i == 0 or i == 2:
            for j in range(0, len(ones_index)):
                if categories[i][0] <= ones_index[j] <= categories[i][1] and categories[i] not in categories_of_image:
                    categories_of_image.append(categories[i])
    if len(categories_of_image) == 0:
        print('Error')
    index_category_of_manipulation = random.randint(0, len(categories_of_image) - 1)
    category_of_manipulation = categories_of_image[index_category_of_manipulation]
    ones_index_category_of_manipulation = ones_index[(category_of_manipulation[0]<=ones_index) & (ones_index<=category_of_manipulation[1])]
    index_attribute_to_remove = ones_index_category_of_manipulation[random.randint(0, len(ones_index_category_of_manipulation) - 1)]
    index_found = False
    while index_found == False:
        index_attribute_to_add = random.randint(category_of_manipulation[0], category_of_manipulation[1])
        if index_attribute_to_add not in ones_index:
            index_found = True
    np_indicator = np.zeros((1, 151))
    np_indicator[0][index_attribute_to_add] = 1
    np_indicator[0][index_attribute_to_remove] = -1
    new_indicator = torch.from_numpy(np_indicator).cuda()
    return new_indicator


def create_new_indicator_method1(label_ref, categories):
    ones_index = np.where(label_ref == 1)[0]
    category_found = False
    while not category_found:
        index_category_of_manipulation = random.randint(0, len(categories) - 1)
        category_of_manipulation = categories[index_category_of_manipulation]
        ones_index_category_of_manipulation = ones_index[(category_of_manipulation[0]<=ones_index) & (ones_index<=category_of_manipulation[1])]
        if len(ones_index_category_of_manipulation) > 0:
            category_found = True
    index_attribute_to_remove = ones_index_category_of_manipulation[random.randint(0, len(ones_index_category_of_manipulation) - 1)]
    index_found = False
    while index_found == False:
        index_attribute_to_add = random.randint(category_of_manipulation[0], category_of_manipulation[1])
        if index_attribute_to_add not in ones_index:
            index_found = True
    np_indicator = np.zeros((1, 151))
    np_indicator[0][index_attribute_to_add] = 1
    np_indicator[0][index_attribute_to_remove] = -1
    new_indicator = torch.from_numpy(np_indicator).cuda()
    return new_indicator


def check_label_data(label_ref, label_nidx, dict_categories, categories_to_check):
    dict_categories_to_check = dict((k, dict_categories[k]) for k in categories_to_check if k in dict_categories)
    new_label_ref = np.zeros(shape=0)
    new_label_nidx = np.zeros(shape=0)
    # for index, value in enumerate(label_ref):
    #     for category in categories_to_check:
    #         if category[0] <= index <= category[1]:
    #             new_label_ref.append(value)
    #             new_label_nidx.append(value)
    for category in dict_categories_to_check:
        new_label_ref = np.concatenate((new_label_ref, np.array(label_ref[dict_categories_to_check[category][0]:dict_categories_to_check[category][1] + 1])), axis=None)
        new_label_nidx = np.concatenate((new_label_nidx, np.array(label_nidx[dict_categories_to_check[category][0]:dict_categories_to_check[category][1] + 1])), axis=None)

    res = True
    # if (new_label_ref == new_label_nidx).all():
    # if np.array_equal(new_label_ref, new_label_nidx):
    #     res = True
    for i in range(0, len(new_label_ref)):
        if new_label_ref[i] == 1 and new_label_nidx[i] == 0:
            res = False
    return res


def find_necessary_transformations(label_data_manipulated, label_data_target, categories):
    indicator_necessary_transformations = []
    categories_to_modify = []
    for i in range(0, len(categories)):
        label_manipulated_current_category = np.array(label_data_manipulated[categories[i][0]:categories[i][1] + 1])
        label_target_current_category = np.array(label_data_target[categories[i][0]:categories[i][1] + 1])
        if (label_manipulated_current_category == label_target_current_category).all():
            continue
        else:
            categories_to_modify.append(categories[i])
            np_indicator = np.zeros((1, 151))
            index_attribute_to_remove = np.where(label_manipulated_current_category == 1)[0]
            index_attribute_to_add = np.where(label_target_current_category == 1)[0]
            if len(index_attribute_to_add) > 0:
                np_indicator[0][index_attribute_to_add + categories[i][0]] = 1
            if len(index_attribute_to_remove) > 0:
                np_indicator[0][index_attribute_to_remove + categories[i][0]] = -1
            new_indicator = torch.from_numpy(np_indicator).cuda()
            indicator_necessary_transformations.append(new_indicator)
    return indicator_necessary_transformations, categories_to_modify