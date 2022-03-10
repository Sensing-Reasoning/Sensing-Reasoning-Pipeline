import os
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from datasets import get_dataset, DATASETS
from architectures import ARCHITECTURES, get_architecture
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import StepLR
from torch.nn import  Sigmoid, Softmax
from time import time
import pickle
import datetime
from scipy.stats import norm, binom_test
import numpy as np
from math import ceil
import pickle as pkl
from train_utils import setup_seed
from architectures import get_architecture
from statsmodels.stats.proportion import proportion_confint
import argparse

parser = argparse.ArgumentParser(description='pA generator')
parser.add_argument('--noise_sd', default=0.12, type=float,
                    help="standard deviation of Gaussian noise for data augmentation")
parser.add_argument('--top', default=2, type=int,
                    help="pick top k characters")
parser.add_argument('--gpu', type=str, default = '2', help='folder to save model and training log)')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
test_dataset = get_dataset('word10_main', 'test')

sigma = args.sigma
noise_sd = args.noise_sd
all_path=['../ckpts/model/main/main-1_noise_sd%.2f.pth.tar' % noise_sd]
all_path.extend(['../ckpts/model/character/character-%d_noise_sd%.2f.pth.tar' % (i,noise_sd) for i in range(1,6)])
    
def _count_arr(arr, length):
    counts = np.zeros(length, dtype=int)
    for idx in arr:
          counts[idx] += 1
    return counts

def _ensemble_sample_noise( x, num, batch_size):
    """ Sample the base classifier's prediction under noisy corruptions of the input x.

    :param x: the input [channel x width x height]
    :param num: number of samples to collect
    :param batch_size:
    :return: an ndarray[int] of length num_classes containing the per-class counts
    """

    old_x = x.clone().reshape(1,-1)
    mat = []
    for model_id, path in enumerate(all_path):
        checkpoint = torch.load(path)
        cur_num = num
        if model_id == 0:
            x = old_x.clone().cuda()
            num_class = 10
            model = get_architecture('MLP', 'word10_main', classes = num_class)
        else:
            x = ori_x.clone()[:,28 * 28 * (model_id - 1): 28 * 28 * model_id].cuda()
            num_class = 26
            model = get_architecture('MLP', 'word10_character', classes = num_class)
        
        m = Softmax(dim=1)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()

        setup_seed(0)
        with torch.no_grad():
            counts = np.zeros(num_class, dtype=int)
            for _ in range(ceil(cur_num / batch_size)):
                this_batch_size = min(batch_size, cur_num)
                cur_num -= this_batch_size
                batch = x.repeat((this_batch_size, 1))
                noise = torch.randn_like(batch, device='cuda') * noise_sd
                predictions = model(batch + noise).argmax(1)
                counts += _count_arr(predictions.cpu().numpy(), num_class)

        mat.append(counts)
    mat = np.concatenate(mat, axis=0)
    return mat

def _lower_confidence_bound( NA: int, N: int, alpha: float) -> float:
    """ Returns a (1 - alpha) lower confidence bound on a bernoulli proportion.

    This function uses the Clopper-Pearson method.

    :param NA: the number of "successes"
    :param N: the number of total draws
    :param alpha: the confidence level
    :return: a lower bound on the binomial proportion which holds true w.p at least (1 - alpha) over the samples
    """
    return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]

all_mat = []
all_pick = []

for i in range(len(test_dataset)):

    counts = np.zeros(10 + 26 * 5, dtype=float)
    pick = np.zeros(10 + 26 * 5, dtype=int)
    (x, label) = test_dataset[i]
    before_time = time()
    # sample for determining the topk
    count_N0 = _ensemble_sample_noise(x, 10000, 10000)
    count_N = _ensemble_sample_noise(x, 100000, 50000)
    
    for j in range(10):
        counts[j] = _lower_confidence_bound(count_N[j], 100000, 0.001)
        pick[j] = 1
        
    for j in range(5):
        part = count_N0[10+26*j:10+26*(j+1)]
        ind = np.argpartition(part, -args.top)[-args.top:]
        for k in ind:
            counts[10+26*j + k] = _lower_confidence_bound(count_N[10+26*j + k], 100000, 0.001)
            pick[10+26*j + k] = 1
            
    all_mat.append(counts.reshape(1,-1))
    all_pick.append(pick.reshape(1,-1))
        
    after_time = time()
    time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
    print(i, time_elapsed)
    
all_mat = np.concatenate(all_mat, axis=0)
all_pick = np.concatenate(all_pick, axis=0)

with open('../ckpts/pA/word10_top%d_sigma%.2f.pkl'% (args.top, args.noise_sd),'wb') as f:
    pickle.dump([all_mat,all_pick], f)
