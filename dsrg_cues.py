import numpy as np
import os, sys
import os.path as osp
import pylab
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import argparse
import cPickle
# import pyDRFI
import torch
import torch.nn as nn
from torch.autograd import Variable
import cv2

# from model import novelmodel,FeatureExtractor

# RESTORE_FROM = './model_vgg_cam_rdc.pth.tar'
SAVE_PATH = './cues-sal/'
def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='evaluate segmentation result')
    parser.add_argument('--voc', dest='voc_dir',
                        help='ground truth dir',
                        default='/workspace2/fengjp/data/JPEGImages/', type=str)
    parser.add_argument('--images', dest='image_ids',
                        help='test ids file path',
                        default='dataset/list/input_list.txt', type=str)
    # parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
    #                     help="Where restore model parameters from.") 

    args = parser.parse_args()
    return args



def preprocess(image, size):
    mean_pixel = np.array([104.008, 116.669, 122.675])

    image = np.array(image)
    H, W, _ = image.shape
    image = zoom(image.astype('float32'), (size / H, size / W, 1.0), order=1)
    image = image - mean_pixel
    image = image.transpose([2, 0, 1])
    return image

def generate_cues(localization, cam_cues, labels):
    cues = np.zeros_like(cam_cues)
    cues[0, :, :] = cam_cues[0, :, :]
    # cues[0, :, :] = bg

    present_class_index = np.where(labels[1:] == 1)[0]
    sum_of_calss = np.sum(localization, axis=(1,2))
    sum_of_present_class = sum_of_calss[labels[1:]==1]
    index = sorted(range(len(sum_of_present_class)), key=lambda k: sum_of_present_class[k], reverse=True)
    for i in index:
        local_index = present_class_index[i]
        # index_map = np.where(localization[local_index] == 1)
        # cues[:, index_map[0], index_map[1]] = 0
        # cues[local_index+1, index_map[0], index_map[1]] = 1.0

        index_map = np.where(cam_cues[local_index+1] == 1)
        cues[:, index_map[0], index_map[1]] = 0
        cues[local_index+1, index_map[0], index_map[1]] = 1.0

    return cues

def save_to_pickle(cues_dict, filename):
    with open(filename, 'wb') as handle:
        cPickle.dump(cues_dict, handle, protocol=cPickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    args = parse_args()

    # model = novelmodel()
    # model_weights = torch.load(args.restore_from)
    # model.load_state_dict(model_weights)
    # model.cuda()

    # DRFI = pyDRFI.pyDRFI()
    # DRFI.load('../drfi_cpp/drfiModelCpp.data')

    cues_dict = {}
    data_file = cPickle.load(open('dataset/localization_cues-sal.pickle'))
    image_ids = [i.strip().split() for i in open(args.image_ids) if not i.strip() == '']
    for (img_name, index) in image_ids:
        if int(index) % 100 == 0:
            print('%s processd'%(index))
        img_id = osp.splitext(img_name)[0]
        img_path = os.path.join(args.voc_dir, img_id+'.jpg')
        
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        H, W, _ = image.shape
        # sal = DRFI.getSalMap(image)
        # sal = zoom(sal, (41.0 / H, 41.0 / W), order=1)
        # threshold = 0.07
        # bg = (sal < threshold)

        # image = Variable(torch.from_numpy(preprocess(image, 321.0)).cuda()).unsqueeze(0).float()
        # myexactor = FeatureExtractor(model)
        # feature,params = myexactor(image)

        heat_maps = np.zeros((4, 20, 41, 41))
        localization = np.zeros((20, 41, 41))

        # for i in range(4): 
        #     feature[i] = feature[i].squeeze(0)
        #     for j in range(20):
        #         w = params[i][j].cpu().detach().numpy()
        #         heat_maps[i,j, :, :] = np.sum((feature[i][j].cpu().detach().numpy()) * w[:, None, None], axis=0)
        #         heat_maps[i,j] = heat_maps[i,j] / np.max(heat_maps[i,j].flat)

        # heat_maps_final = np.zeros((20, 41, 41))

        # for i in range(20):
        #     heat_maps_final[i] = heat_maps[0][i] + (heat_maps[1][i]+heat_maps[2][i]+heat_maps[3][i])/3.0
        #     localization[i, :, :] = (heat_maps_final[i, :, :] > 0.7 * np.max(heat_maps_final[i]))
        cam_cues = np.zeros((21, 41, 41))
        labels = np.zeros((21,))
        labels_i = data_file['%i_labels' % int(index)]
        labels[labels_i] = 1.0
        cues_i = data_file['%i_cues' % int(index)]
        cam_cues[cues_i[0], cues_i[1], cues_i[2]] = 1.0

        cues = generate_cues(localization, cam_cues, labels)

        cues_dict['%i_labels' % int(index)] = labels_i
        cues_dict['%i_cues' % int(index)] = np.where(cues==1)
        # cues
        markers_new = np.zeros((41, 41))
        markers_new.fill(21)
        pos = np.where(cues == 1)
        markers_new[pos[1], pos[2]] = pos[0]
        markers_new = zoom(markers_new, (float(H)/41.0, float(W)/41.0), order=0)
        save_path = osp.join(SAVE_PATH,img_id+'.png')
        cv2.imwrite(save_path, markers_new)
    # save_to_pickle(cues_dict, 'localization_cues-0.7-0.07.pickle')
