# Citation:
#     Gated Fusion Network for Joint Image Deblurring and Super-Resolution
#     The British Machine Vision Conference(BMVC2018 oral)
#     Xinyi Zhang, Hang Dong, Zhe Hu, Wei-Sheng Lai, Fei Wang and Ming-Hsuan Yang
# Contact:
#     cvxinyizhang@gmail.com
# Project Website:
#     http://xinyizhang.tech/bmvc2018
#     https://github.com/jacquelinelala/GFN

from __future__ import print_function
import argparse
from importlib import import_module
import os
import time
from math import log10
from os.path import join
from torchvision import transforms
from torchvision import utils as utils
import torch
from torch.utils.data import DataLoader
from datasets.RS import RSDota_test_loader, RSHAZE_test_loader, RSHAZE_train_loader
import statistics
import re
import torch.nn as nn
import torch.nn.functional as F
import pytorch_ssim
from torch.backends import cudnn


loaders_={
	'rshaze_train':RSHAZE_train_loader,
	'rshaze_test':RSHAZE_test_loader,
    'dota_test':RSDota_test_loader
}

parser = argparse.ArgumentParser(description="PyTorch LapSRN Test")
parser.add_argument("--scale", default=4, type=int, help="scale factor, Default: 4")
parser.add_argument("--isTest", type=bool, default=True, help="Test or not")
parser.add_argument('--dataset', type=str, default='../data/rshaze/', help='Path of the validation dataset')
parser.add_argument('--model', default='MSBDN-DFF-v1-1', type=str, help='Import which network')
parser.add_argument("--checkpoint", default="models/MSBDN-DFF-v1-1/1/MSBDN-DFF-v1-1_latest.pkl", type=str, help="Test on intermediate pkl (default: none)")
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--name', type=str, default='MSBDN-DFF-v1-1', help='filename of the training models')
parser.add_argument("--start", type=int, default=2, help="Activated gate module")
parser.add_argument('--testset',type=str,default='dota_test')
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def is_pkl(filename):
    return any(filename.endswith(extension) for extension in [".pkl"])

def test(loader_test, model, criterion, SR_dir):
    avg_psnr = 0
    avg_ssim = 0
    med_time = []

    with torch.no_grad():
        for iteration, batch in enumerate(loader_test, 1):
            # print(iteration)
            Blur = batch[0]
            HR = batch[1]
            Blur = Blur.to(device)
            HR = HR.to(device)

            # name = batch[2][0][:-4]

            start_time = time.perf_counter()#-------------------------begin to deal with an image's time

            sr = model(Blur)

            #modify
            try:
                sr = torch.clamp(sr, min=0, max=1)
            except:
                sr = sr[0]
                sr = torch.clamp(sr, min=0, max=1)
            torch.cuda.synchronize()#wait for CPU & GPU time syn
            evalation_time = time.perf_counter() - start_time#---------finish an image
            med_time.append(evalation_time)


            ssim = pytorch_ssim.ssim(sr, HR)
            #print(ssim)
            avg_ssim += ssim

            mse = criterion(sr, HR)
            psnr = 10 * log10(1 / mse)
            #
            # resultSRDeblur = transforms.ToPILImage()(sr.cpu()[0])
            # resultSRDeblur.save(join(SR_dir, '{0}_{1}.png'.format(name, opt.name)))

            print("Processing {}:  PSNR:{} TIME:{}".format(iteration, psnr, evalation_time))
            avg_psnr += psnr

        print("===> Avg. SR SSIM: {:.4f} ".format(avg_ssim / iteration))
        print("Avg. SR PSNR:{:4f} dB".format(avg_psnr / iteration))
        median_time = statistics.median(med_time)
        print(median_time)
        return avg_psnr / iteration

def model_test(model):
    criterion = torch.nn.MSELoss(size_average=True)
    criterion = criterion.to(device)
    print(opt)
    psnr = test(loader_test, model, criterion, SR_dir)
    return psnr

opt = parser.parse_args()
# device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if torch.cuda.is_available() else torch.device('cpu')
device='cuda' if torch.cuda.is_available() else 'cpu'

str_ids = opt.gpu_ids.split(',')
torch.cuda.set_device(int(str_ids[0]))
root_val_dir = opt.dataset# #----------Validation path
SR_dir = join(root_val_dir, 'Results')  #--------------------------SR results save path
# isexists = os.path.exists(SR_dir)
# if not isexists:
#     os.makedirs(SR_dir)
# print("The results of testing images sotre in {}.".format(SR_dir))

Net = import_module('networks.' + opt.model)
net = Net.make_model(opt)
net = net.to(device)
if device=='cuda':
    net=torch.nn.DataParallel(net)
    cudnn.benchmark=True
# testloader = DataLoader(DataValSet(root_val_dir), batch_size=1, shuffle=False, pin_memory=False)
loader_test=loaders_[opt.testset]
print("===> Loading model and criterion")

if is_pkl(opt.checkpoint):
    test_pkl = opt.checkpoint
    if is_pkl(test_pkl):
        print("Testing model {}----------------------------------".format(opt.checkpoint))
        ckp = torch.load(test_pkl, map_location=lambda storage, loc: storage)
        net.load_state_dict(ckp['model'])
        max_ssim=ckp['max_ssim']
        max_psnr=ckp['max_psnr']
        print("max_ssim is {}, and max_psnr is {}.".format(max_ssim, max_psnr))
        # model = torch.load(test_pkl, map_location=lambda storage, loc: storage)
        print(get_n_params(net))
        #model = model.eval()
        model_test(net)
    else:
        print("It's not a pkl file. Please give a correct pkl folder on command line for example --opt.checkpoint /models/1/GFN_epoch_25.pkl)")
else:
    test_list = [x for x in sorted(os.listdir(opt.checkpoint)) if is_pkl(x)]
    print("Testing on the given 3-step trained model which stores in /models, and ends with pkl.")
    Results = []
    Max = {'max_psnr':0, 'max_epoch':0}
    for i in range(len(test_list)):
        print("Testing model is {}----------------------------------".format(test_list[i]))
        print(join(opt.checkpoint, test_list[i]))
        model = torch.load(join(opt.checkpoint, test_list[i]), map_location=lambda storage, loc: storage)
        print(get_n_params(model))
        model = model.eval()
        psnr = model_test(model)
        Results.append({'epoch':"".join(re.findall(r"\d", test_list[i])[:]), 'psnr': psnr})
        if psnr > Max['max_psnr']:
            Max['max_psnr'] = psnr
            Max['max_epoch'] = "".join(re.findall(r"\d", test_list[i])[:])
    for Result in Results:
        print(Result)
    print('Best Results is : ===========================> ')
    print(Max)



