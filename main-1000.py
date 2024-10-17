import os
import csv
import time  # 导入time模块，用于计时
from typing import Dict
from nas_201_api import NASBench201API as API201
import models
from models import NB101Network
import datasets
from nats_bench import create as create_nats
import torch
import torch.nn as nn
import random
import numpy as np
from nasbench import api as api101
from ptflops import get_model_complexity_info
from measures import get_grad_score, get_ntk_n, get_batch_jacobian, get_logdet, get_zenscore
import argparse

# Argument parser
parser = argparse.ArgumentParser(description='ZS-NAS')
parser.add_argument('--searchspace', metavar='ss', type=str, choices=['101', '201', 'nats', 'nats_tss', 'mbnv2', 'resnet'],
                    help='define the target search space of benchmark')
parser.add_argument('--dataset', metavar='ds', type=str, choices=['cifar10', 'cifar100', 'ImageNet16-120', 'imagenet-1k', 'cifar10-valid'],
                    help='select the dataset')
parser.add_argument('--data_path', type=str, default='/home/test0/dataset/',
                    help='the path where you store the dataset')
parser.add_argument('--cutout', type=int, default=0,
                    help='use cutout or not on input data')
parser.add_argument('--batchsize', type=int, default=1024,
                    help='batch size for each input batch')
parser.add_argument('--num_worker', type=int, default=8,
                    help='number of threads for data pipelining')
parser.add_argument('--metric', type=str, choices=['basic', 'ntk', 'lr', 'logdet', 'grad', 'zen', 'IB'],
                    help='define the zero-shot proxy for evaluation')
parser.add_argument('--startnetid', type=int, default=0,
                    help='the index of the first network to be evaluated in the search space. currently only works for nb101')
parser.add_argument('--manualSeed', type=int, default=0,
                    help='random seed')
args = parser.parse_args()


# Initialize and get datasets, model, and loss function
def getmisc(args):
    manualSeed = args.manualSeed
    np.random.seed(manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if args.dataset == "cifar10":
        root = args.data_path
        imgsize = 32
    elif args.dataset == "cifar100":
        root = args.data_path
        imgsize = 32
    elif args.dataset.startswith("imagenet-1k"):
        root = args.data_path + 'ILSVRC/Data/CLS-LOC'
        imgsize = 224
    elif args.dataset.startswith("ImageNet16"):
        root = args.data_path + 'img16/ImageNet16/'
        imgsize = 16

    train_data, test_data, xshape, class_num = datasets.get_datasets(args.dataset, root, args.cutout)

    trainloader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batchsize, shuffle=True, num_workers=args.num_worker)
    testloader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batchsize, shuffle=False, num_workers=args.num_worker)

    ce_loss = nn.CrossEntropyLoss().cuda()
    return imgsize, ce_loss, trainloader, testloader


# Save results to a CSV file
def save_results_to_csv(results, file_path='results.csv'):
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(results)


# Function to evaluate a network in NAS-Bench-101
def search101(nasbench, netid, dataset, imgsize, metric, trainloader, ce_loss):
    unique_hash = list(nasbench.hash_iterator())[netid]
    fixed_metrics, computed_metrics = nasbench.get_metrics_from_hash(unique_hash)

    ops = fixed_metrics['module_operations']
    adjacency = fixed_metrics['module_adjacency']

    network = NB101Network((adjacency, ops))
    network.cuda()

    # For basic metric
    if metric == 'basic':
        acc_metrics = get101acc(computed_metrics)
        macs, params = get_model_complexity_info(network, (3, imgsize, imgsize), as_strings=False,
                                                 print_per_layer_stat=False, verbose=False)
        return [netid, macs, params] + acc_metrics
    return None


def enumerate_networks(args):
    # 获取一些必要的信息和数据集
    imgsize, ce_loss, trainloader, testloader = getmisc(args)

    # 用于存储结果
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)  # 确保结果存储目录存在
    output_file = os.path.join(results_dir, f'{args.searchspace}_{args.dataset}_{args.metric}_results.csv')

    # 计时开始
    start_time = time.time()

    # 打开文件准备保存结果
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['netid', 'macs', 'params', 'train_acc', 'test_acc']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # NAS-Bench-101 的架构评估
        if '101' in args.searchspace.lower():
            assert args.dataset == "cifar10"
            NASBENCH_TFRECORD = '/home/test0/dataset/nasbench/nasbench_full.tfrecord'
            nasbench = api101.NASBench(NASBENCH_TFRECORD)

            # 评估的网络数限制在1000个
            allnethash = list(nasbench.hash_iterator())
            for netid in range(args.startnetid, min(1000, len(allnethash))):
                unique_hash = allnethash[netid]
                fixed_metrics, computed_metrics = nasbench.get_metrics_from_hash(unique_hash)

                # 获取架构的准确率
                def getallacc(data_dict:dict):
                    acc4 = sum(data_dict[4][i]['final_test_accuracy'] for i in range(3)) / 3.0
                    acc12 = sum(data_dict[12][i]['final_test_accuracy'] for i in range(3)) / 3.0
                    acc36 = sum(data_dict[36][i]['final_test_accuracy'] for i in range(3)) / 3.0
                    acc108 = sum(data_dict[108][i]['final_test_accuracy'] for i in range(3)) / 3.0
                    return [acc4, acc12, acc36, acc108]

                acc_metrics = getallacc(computed_metrics)
                ops = fixed_metrics['module_operations']
                adjacency = fixed_metrics['module_adjacency']

                # 构建网络
                network = NB101Network((adjacency, ops))
                network.cuda()

                # 计算架构的 flops 和参数数量
                macs, params = get_model_complexity_info(network, (3, imgsize, imgsize), as_strings=False, print_per_layer_stat=False, verbose=False)

                # 保存结果到 CSV 文件
                writer.writerow({
                    'netid': netid,
                    'macs': macs,
                    'params': params,
                    'train_acc': acc_metrics[0],  # 训练准确率
                    'test_acc': acc_metrics[3]  # 测试准确率
                })
                print(f"Evaluated network {netid}, saved results.")

        # NAS-Bench-201 的架构评估
        elif '201' in args.searchspace.lower():
            api = API201('/home/test0/dataset/nasbench/NAS-Bench-201-v1_1-096897.pth', verbose=False)

            # 评估的网络数限制在1000个
            for netid in range(args.startnetid, 1000):
                network, metric = search201(api, netid, args.dataset)
                network.cuda()

                # 计算架构的 flops 和参数数量
                macs, params = get_model_complexity_info(network, (3, imgsize, imgsize), as_strings=False, print_per_layer_stat=False, verbose=False)

                # 保存结果到 CSV 文件
                writer.writerow({
                    'netid': netid,
                    'macs': macs,
                    'params': params,
                    'train_acc': metric[1],
                    'test_acc': metric[3]
                })
                print(f"Evaluated network {netid}, saved results.")

        # NATS 的架构评估
        elif 'nats' in args.searchspace.lower():
            if 'tss' in args.searchspace.lower():
                api = create_nats('/home/test0/dataset/nasbench/NATS/NATS-tss-v1_0-3ffb9-simple', 'tss', fast_mode=True, verbose=True)
                hpval = '200'
            else:
                api = create_nats('/home/test0/dataset/nasbench/NATS/NATS-sss-v1_0-50262-simple', 'sss', fast_mode=True, verbose=True)
                hpval = '90'

            # 评估的网络数限制在1000个
            for netid in range(args.startnetid, 1000):
                network, metric = search_nats(api, netid, args.dataset, hpval)
                network.cuda()

                # 计算架构的 flops 和参数数量
                macs, params = get_model_complexity_info(network, (3, imgsize, imgsize), as_strings=False, print_per_layer_stat=False, verbose=False)

                # 保存结果到 CSV 文件
                writer.writerow({
                    'netid': netid,
                    'macs': macs,
                    'params': params,
                    'train_acc': metric,
                    'test_acc': metric
                })
                print(f"Evaluated network {netid}, saved results.")

    # 计时结束并输出总时间
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total time to evaluate 1000 networks: {total_time:.2f} seconds")


# Helper function to get accuracy from NAS-Bench-101
def get101acc(data_dict: dict):
    acc4 = (data_dict[4][0]['final_test_accuracy'] + data_dict[4][1]['final_test_accuracy'] + data_dict[4][2]['final_test_accuracy']) / 3.0
    acc12 = (data_dict[12][0]['final_test_accuracy'] + data_dict[12][1]['final_test_accuracy'] + data_dict[12][2]['final_test_accuracy']) / 3.0
    acc36 = (data_dict[36][0]['final_test_accuracy'] + data_dict[36][1]['final_test_accuracy'] + data_dict[36][2]['final_test_accuracy']) / 3.0
    acc108 = (data_dict[108][0]['final_test_accuracy'] + data_dict[108][1]['final_test_accuracy'] + data_dict[108][2]['final_test_accuracy']) / 3.0
    return [acc4, acc12, acc36, acc108]


if __name__ == '__main__':
    enumerate_networks(args)
