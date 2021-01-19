import os
import cv2
import sys
import scipy
import torch
import argparse
import numpy as np
import scipy.ndimage as nd
from torch.autograd import Variable
import torchvision.models as models
from model import Res_Deeplab
from model import DeepLab_LargeFOV
# from deeplabv2 import Res_Deeplab
import pylab
import os.path as osp
# import krahenbuhl2013
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax
import torch.nn as nn
import warnings
# from wider_resnet import WiderResNet38
warnings.filterwarnings("ignore")

DATA_DIRECTORY = '/workspace/fjp/data/VOC2012/'
DATA_LIST_PATH = './dataset/list/val.txt'
NUM_CLASSES = 21
RESTORE_FROM = './results/snapshots/Epoch_7.pth'
RESULT_DIR  = './results/evaluation/'

def get_arguments():
    """Parse all the arguments provide from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--result-dir", type=str, default=RESULT_DIR,
                        help="Where to save the generated mask of the trained model.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--gpu", type=str, default="0,1",
                        help="choose gpu device.")
    return parser.parse_args()

def write_to_png_file(im, f):
    cv2.imwrite(f, im)

def preprocess(image, size, mean_pixel):

    image = np.array(image)

    image = nd.zoom(image.astype('float32'),
                    (size, size, 1.0),
                    order=1)

    # image = image[:, :, [2, 1, 0]]
    image = image - mean_pixel

    image = image.transpose([2, 0, 1])
    return np.expand_dims(image, 0)

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

def main():
    """Create the model and start the evaluation process."""
    args = get_arguments()
    # mean_pixel = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
    mean_pixel = np.array((122.67891434,116.66876762,104.00698793), dtype=np.float32)
    str_ids = (args.gpu).split(',')
    gpu_ids = []
    for str_id in str_ids:
        gid = int(str_id)
        gpu_ids.append(gid)
    # model = WiderResNet38(num_classes=args.num_classes,config_path="./dataset/test.json")
    # model = nn.DataParallel(model, device_ids=gpu_ids)
    model = torch.nn.DataParallel(Res_Deeplab(num_classes=args.num_classes),[0,1])
    # model = Res_Deeplab(num_classes=args.num_classes)
    saved_state_dict = torch.load(args.restore_from)
    model.load_state_dict(saved_state_dict)
    model.eval()
    model.cuda()
    image_ids = [i.strip() for i in open(args.data_list) if not i.strip() == '']
    data_dir = osp.join(args.data_dir, 'JPEGImages')
    for index, img_id in enumerate(image_ids):
        print(index, img_id)
        image_file = osp.join(data_dir, img_id+'.jpg')

        im = pylab.imread(image_file)
        d1, d2 = float(im.shape[0]), float(im.shape[1])
        scores_all = 0
        for size in [0.75, 1, 1.25]: #[385, 513, 641]
            im_process = preprocess(im,size,mean_pixel)
            _,_,pred= model(Variable(torch.from_numpy(im_process)).cuda())
            scores = np.transpose(pred.cpu().data[0].numpy(), [1, 2, 0])
            del pred 
            scores = nd.zoom(scores, (d1 / scores.shape[0], d2 / scores.shape[1], 1.0), order=1)
            scores_all += scores
        scores_exp = np.exp(scores_all - np.max(scores_all, axis=2, keepdims=True))
        probs = scores_exp / np.sum(scores_exp, axis=2, keepdims=True)
        # import pdb; pdb.set_trace()
        eps = 0.00001
        probs[probs < eps] = eps
        im = np.ascontiguousarray(im, dtype=np.uint8)
        probs = np.transpose(probs, [2, 0, 1])
        # result = np.argmax(krahenbuhl2013.CRF(im, np.log(probs), scale_factor=1.0), axis=2)
        result = np.argmax(crf_inference(im, probs, scale_factor=1.0), axis=0)
        # print(np.unique(result))
        if not os.path.exists(args.result_dir):
                os.makedirs(args.result_dir)
        save_path = osp.join(args.result_dir, img_id+'.png')
        write_to_png_file(result, save_path)

if __name__ == '__main__':
    main()

