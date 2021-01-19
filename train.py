import argparse
import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import math
import pickle
import cv2
from torch.autograd import Variable
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import sys
import os
import maxflow
import os
import os.path as osp
from model import Res_Deeplab
from dataset import VOCDataSet
from cues_reader import CuesReader
from scipy.ndimage import zoom
from scipy.linalg import fractional_matrix_power
import random
import timeit
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax
import pdb
import time
from tool import *
import torchvision.models as models
import matplotlib.pyplot as plt
sys.path.append('/workspace2/fengjp/Synchronized-BatchNorm-PyTorch/')
import warnings
import pickle
from dist_ops import synchronize
from torch.utils.model_zoo import load_url as load_state_dict_from_url
warnings.filterwarnings("ignore")
start = timeit.default_timer()
VOC_BBOX_LABEL_NAMES = (
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor')
IMG_MEAN = np.array((122.67891434,116.66876762,104.00698793), dtype=np.float32)
EPOCHS = 7
BATCH_SIZE = 16
DATA_DIRECTORY = '/workspace/fjp/data/VOC2012/'
DATA_LIST_PATH = './dataset/list/input_list.txt'
IGNORE_LABEL = 255
INPUT_SIZE = '321,321'
LEARNING_RATE = 5e-4
MOMENTUM = 0.9
NUM_CLASSES = 21
NUM_STEPS = 10582
POWER = 0.9
RANDOM_SEED = 1234
CUES_DIR = './dataset/'
CUES_NAME = 'localization_cues-sal.pickle'
RESTORE_FROM = "./dataset/wide_resnet38_ipabn_lr_256.pth.tar"
SNAPSHOT_DIR = './results/snapshots/'
WEIGHT_DECAY = 0.0005
K = 10
GAMMA = 0.30

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--gamma", type=float, default=GAMMA,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu-ids", type=str, default='0,1',
                        help="choose gpu device.")
    parser.add_argument("--local_rank", type=int, default=0,
                        help="process rank on node.")
    parser.add_argument("--distributed", action="store_true",
                        help="process rank on node.")
    return parser.parse_args()


args = get_arguments()

def crf_inference(img, probs, labels=21, t=10, scale_factor=12):

    h, w = img.shape[:2]
    n_labels = labels

    d = dcrf.DenseCRF2D(w, h, n_labels)

    unary = unary_from_softmax(probs)
    unary = np.ascontiguousarray(unary)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3/scale_factor, compat=3)
    d.addPairwiseBilateral(sxy=80/scale_factor, srgb=13, rgbim=np.copy(img), compat=10)
    Q = d.inference(t)

    return np.array(Q).reshape((n_labels, h, w))

def crf_operation(images,probs):
    batchsize, _, h, w = probs.shape
    probs[probs < 0.0001] = 0.0001
    # unary = np.transpose(probs, [0, 2, 3, 1])

    mean_pixel = np.array([123.0, 117.0, 104.0])
    im = images
    im = zoom(im, (1.0, 1.0, float(h) / im.shape[2], float(w) / im.shape[3]), order=1)
    im = np.transpose(im, [0, 2, 3, 1])
    im = im + mean_pixel[None, None, None, :]
    im = np.ascontiguousarray(im, dtype=np.uint8)
    result = np.zeros(probs.shape)
    for i in range(batchsize):
        result[i] = crf_inference(im[i], probs[i])

    result[result < 0.0001] = 0.0001
    result = result / np.sum(result, axis=1, keepdims=True)

    result = np.log(result)

    return result

def clip(x, min, max):
    x_min = x < min
    x_max = x > max
    y = torch.mul(torch.mul(x, (~x_min).float()), (~x_max).float()) + ((x_min.float()) * min) + (x_max * max).float()
    return y


def constrain_loss(probs, crf):
    probs_smooth = torch.exp(torch.from_numpy(crf)).float().cuda()
    loss = torch.mean(torch.sum(probs_smooth * torch.log(clip(probs_smooth / probs, 0.05, 20)), dim=1))
    return loss

def cacul_knn_matrix_(feature_map, k):
    batchsize, channels, h, w = feature_map.shape
    n = h * w
    # S = torch.zeros(batchsize,n,n)
    knn_matrix = torch.zeros(batchsize, n, n, device='cuda')
    for i in range(batchsize):
        # reshape feature_map: n*channel
        feature = torch.transpose(feature_map[i].reshape(channels, h * w), 0, 1)
        # ||x1-x2||^2
        x1_norm = (feature ** 2).sum(dim=1).view(-1, 1)  # n*1
        x2_norm = x1_norm.view(1, -1)  # 1*n
        dist = (x1_norm + x2_norm - 2.0 * torch.mm(feature, feature.transpose(0, 1))).abs()  # x1_norm + x2_norm : n*n
        # first method
        value, position = torch.topk(dist,k,dim=1,largest=False)
        temp = value[:,-1].unsqueeze(1).repeat(1,n)
        knn_matrix[i] = (dist<=temp).float()-torch.eye(n,n,device='cuda')

    return knn_matrix

def generate_supervision(feature, label, cues, mask, pred, knn_matrix):
    batchsize, class_num, h, w = pred.shape
    Y = torch.zeros(batchsize, class_num, h, w)
    supervision = cues.clone()

    for i in range(batchsize):
        label_class = torch.nonzero(label[i])
        markers_new = np.zeros((h, w))
        markers_new.fill(NUM_CLASSES)
        pos = np.where(cues[i].numpy() == 1)
        markers_new[pos[1], pos[2]] = pos[0]
        markers_new_flat = markers_new.reshape(h*w)
        for c in (label_class):
            c_c = c[0].numpy()
            feature_c = feature[i].reshape(feature.shape[1], h * w).transpose(1, 0)
            pred_c = pred[i][c[0]]
            pred_c_flat = pred_c.flatten()
            g = maxflow.Graph[float]()
            nodes = g.add_nodes(h * w)
            pos = np.where(markers_new_flat == c_c)
            for node_i in pos[0]:
                g.add_tedge(nodes[node_i], 0, 10)
                k_neighbor = np.where(knn_matrix[i][node_i] == 1)
                for neighbor in (k_neighbor[0]):
                    g.add_edge(nodes[node_i], nodes[neighbor], 1, 1)
            pos = np.where(markers_new_flat == NUM_CLASSES)
            for node_i in pos[0]:
                g.add_tedge(nodes[node_i], -np.log10(pred_c_flat[node_i]), -np.log10(1 - pred_c_flat[node_i]))
                k_neighbor = np.where(knn_matrix[i][node_i] == 1)
                for neighbor in (k_neighbor[0]):
                    g.add_edge(nodes[node_i], nodes[neighbor], 1, 1)
            pos = np.where((markers_new_flat!= NUM_CLASSES)&(markers_new_flat!=c_c))
            for node_i in pos[0]:
                g.add_tedge(nodes[node_i], 10, 0)
                k_neighbor = np.where(knn_matrix[i][node_i] == 1)
                for neighbor in (k_neighbor[0]):
                    g.add_edge(nodes[node_i], nodes[neighbor], 1, 1)

            flow = g.maxflow()
            node_ids = np.arange(h*w)
            label_new = g.get_grid_segments(node_ids)
            
            supervision[i][c[0]] = torch.from_numpy(np.where(pred_c>0.7,label_new.astype(int).reshape(h, w),supervision[i][c[0]])).float()

    return supervision

def softmax(preds,min_prob):
    preds_max = torch.max(preds, dim=1, keepdim=True)
    preds_exp = torch.exp(preds - preds_max[0])
    probs = preds_exp / torch.sum(preds_exp, dim=1, keepdim=True) 
    min_prob = torch.ones((probs.shape),device='cuda')*min_prob
    probs = probs + min_prob
    probs = probs / torch.sum(probs, dim=1, keepdim=True)
    return probs


def cal_seeding_loss(pred, label):
    pred_bg = pred[:, 0, :, :]
    labels_bg = label[:, 0, :, :].float().to('cuda')
    pred_fg = pred[:, 1:, :, :]
    labels_fg = label[:, 1:, :, :].float().to('cuda')

    count_bg = torch.sum(torch.sum(labels_bg, dim=2, keepdim=True), dim=1, keepdim=True)
    count_fg = torch.sum(torch.sum(torch.sum(labels_fg, dim=3, keepdim=True), dim=2, keepdim=True), dim=1, keepdim=True)

    sum_bg = torch.sum(torch.sum(labels_bg * torch.log(pred_bg), dim=2, keepdim=True), dim=1, keepdim=True)
    sum_fg = torch.sum(torch.sum(torch.sum(labels_fg * torch.log(pred_fg), dim=3, keepdim=True), dim=2, keepdim=True),
                       dim=1, keepdim=True)
    loss_1 = -(sum_bg / torch.max(count_bg, torch.tensor(0.0001,device='cuda'))).mean()
    loss_2 = -(sum_fg / torch.max(count_fg, torch.tensor(0.0001,device='cuda'))).mean()
    loss_balanced = loss_1 + loss_2
    return loss_balanced

def lr_step(base_lr, i_iter, epoch, max_iter, gamma):
    if (epoch==0) and (i_iter < 200):
    	return (base_lr/200)*(i_iter+1)
    return base_lr * gamma ** (np.floor(float(epoch)/3.0))

def lr_poly(base_lr, iter, epoch, max_iter, power):
    return base_lr*((1-float(iter+max_iter*epoch)/(max_iter*EPOCHS))**(power))


def get_1x_lr_params_NOscale(model):
    b = []


    b.append(model.module.conv1)
    b.append(model.module.bn1)
    b.append(model.module.layer1)
    b.append(model.module.layer2)
    b.append(model.module.layer3)
    b.append(model.module.layer4)

    for i in range(len(b)):
        for j in b[i].modules():
            jj = 0
            for k in j.parameters():
                jj += 1
                if k.requires_grad:
                    yield k


def get_10x_lr_params(model):
    b = []
    b.append(model.module.layer5.parameters())

    for j in range(len(b)):
        for i in b[j]:
            yield i


def adjust_learning_rate(optimizer, i_iter, epoch):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = lr_step(args.learning_rate, i_iter, epoch, args.num_steps, args.gamma)
    optimizer.param_groups[0]['lr'] = lr
    optimizer.param_groups[1]['lr'] = lr * 10

def save_to_pickle(cues_dict, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(cues_dict, handle, protocol=cPickle.HIGHEST_PROTOCOL)

def main():
    """Create the model and start the training."""

    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)
    k = K
    cudnn.enabled = True

    # Create network.
    model = Res_Deeplab(num_classes=args.num_classes)
    model.load_state_dict(load_state_dict_from_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'), strict=False)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.train()

    train_dataset = VOCDataSet(args.data_dir, args.data_list, max_iters=args.num_steps, crop_size=input_size,
                   scale=False, mirror=True, mean=IMG_MEAN, cues_dir=CUES_DIR, cues_name=CUES_NAME)
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    synchronize()
    world_size = torch.distributed.get_world_size()
    args.batch_size_n = int(args.batch_size / world_size)
    model = torch.nn.parallel.DistributedDataParallel(model.to(args.local_rank), device_ids=[args.local_rank], output_device=args.local_rank, broadcast_buffers=False)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    trainloader = data.DataLoader(train_dataset, batch_size=args.batch_size_n, shuffle=False, num_workers=2, pin_memory=True, sampler=train_sampler)
    
    cudnn.benchmark = True

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    optimizer = optim.SGD([{'params': get_1x_lr_params_NOscale(model), 'lr': args.learning_rate},
                           {'params': get_10x_lr_params(model), 'lr': 10 * args.learning_rate}],
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()
    min_prob = torch.tensor(0.0000001,device='cuda')
    for ep in range(EPOCHS):
        train_sampler.set_epoch(ep)
        for i_iter, batch in enumerate(trainloader):
            images, cues, labels, mask = batch
            images = Variable(images).cuda()
            optimizer.zero_grad()
            adjust_learning_rate(optimizer, i_iter, ep)
            feature_2, feature_4, pred = model(images)
            feature_2 = torch.nn.functional.interpolate(feature_2, size=[41, 41], mode='bilinear', align_corners=True)
            feature = torch.cat((feature_2, feature_4), 1)

            knn_matrix = cacul_knn_matrix_(feature, k)
            probs = softmax(pred,min_prob)
            batchsize = probs.shape[0]
            labels = labels.reshape(labels.shape[0], labels.shape[3])  
            crf = crf_operation(images.cpu().detach().numpy(), probs.cpu().detach().numpy())
            crf_constrain_loss = constrain_loss(probs, crf)
            supervision = generate_supervision(feature.cpu().detach().numpy(), labels, cues, mask,
                                               probs.cpu().detach().numpy(), knn_matrix.cpu().detach().numpy())
            seeding_loss = cal_seeding_loss(probs, supervision)
            loss = crf_constrain_loss + seeding_loss
            loss.backward()
            optimizer.step()
            print('epoch=', ep,'iter =', i_iter, 'of', int(args.num_steps/args.batch_size)+1, 'completed, loss = ', loss.data.cpu().numpy())
            if i_iter == int(args.num_steps/args.batch_size) and args.local_rank == 0:
                print('save model ...')
                torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'Epoch_' + str(ep+1) + '.pth'))
                break

    end = timeit.default_timer()
    print(end - start, 'seconds')


if __name__ == '__main__':
    main()
