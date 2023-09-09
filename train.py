from __future__ import print_function
import torch.optim as optim
import os
import torch
from torch.backends import cudnn
from datasets.RS import RSDota_test_loader, RSDota_train_loader, RSHAZE_test_loader, RSHAZE_train_loader
from importlib import import_module
import random
import time
import numpy as np
import statistics
import torch.nn.functional as F
from option import opt
from pytorch_ssim import ssim,psnr

loaders_={
	'rshaze_train':RSHAZE_train_loader,
	'rshaze_test':RSHAZE_test_loader,
    'dota_train':RSDota_train_loader,
	'dota_test':RSDota_test_loader
}

training_settings=[
    {'nEpochs': 100, 'lr': 1e-4, 'step': 50, 'lr_decay': 0.1},
    {'nEpochs': 100, 'lr': 1e-4, 'step': 50, 'lr_decay': 0.1}
]

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def adjust_learning_rate(epoch):
        lr = opt.lr * (opt.lr_decay ** (epoch // opt.step))
        print(lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def train(model, loader_train, criterion, optimizer, epoch):

    epoch_loss = 0
    start_time_data=0
    med_time_data = []
    med_time_gpu = []

    for iteration, batch in enumerate(loader_train, 1):
        evalation_time_data = time.perf_counter() - start_time_data
        med_time_data.append(evalation_time_data)
        start_time_gpu = time.perf_counter()

        Hazy = batch[0]
        GT = batch[1]
        Hazy = Hazy.to(opt.device)
        GT = GT.to(opt.device)

        dehaze = model(Hazy)
        mse = criterion(dehaze, GT)

        epoch_loss += mse
        optimizer.zero_grad()
        mse.backward()
        optimizer.step()
        evalation_time_gpu = time.perf_counter() - start_time_gpu
        med_time_gpu.append(evalation_time_gpu)

        if iteration % 100 == 0:
            median_time_data = statistics.median(med_time_data)
            median_time_gpu = statistics.median(med_time_gpu)
            print("===> Loading Time: {:.6f}; Runing Time:{:.6f}".format(median_time_data, median_time_gpu))
            print("===> Epoch[{}]({}/{}): Loss{:.4f};".format(epoch, iteration, len(loader_train), mse.cpu()))
            med_time_data = []
            med_time_gpu = []
        start_time_data = time.perf_counter()
    print("===>Epoch{} Part: Avg loss is :{:4f}".format(epoch, epoch_loss / len(loader_train)))
    return epoch_loss / len(loader_train)


def test(net,loader_test,max_psnr,max_ssim,step):
	net.eval()
	torch.cuda.empty_cache()
	ssims=[]
	psnrs=[]
	#s=True
	for i ,(inputs,targets) in enumerate(loader_test):
		inputs=inputs.to(opt.device);targets=targets.to(opt.device)
		pred=net(inputs)
		ssim1=ssim(pred,targets).item()
		psnr1=psnr(pred,targets)
		ssims.append(ssim1)
		psnrs.append(psnr1)
	return np.mean(ssims) ,np.mean(psnrs)


if __name__ == "__main__":
    opt.seed = random.randint(1, 10000)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    losses=[]
    max_ssim=0
    max_psnr=0
    ssims=[]
    psnrs=[]

    loader_train=loaders_[opt.trainset]
    loader_test=loaders_[opt.testset]

    Net = import_module('networks.' + opt.model)
    net = Net.make_model(opt)
    net = net.to(opt.device)
    print(get_n_params(net))
    if opt.device=='cuda':
        net=torch.nn.DataParallel(net)
        cudnn.benchmark=True
    
    if opt.resume and os.path.isfile(opt.resume):
        print("Loading from checkpoint {}".format(opt.resume))
        ckp = torch.load(opt.resume, map_location=lambda storage, loc: storage)
        losses=ckp['losses']
        net.load_state_dict(ckp['model'])
        opt.start_training_step=ckp['step']
        opt.start_epoch=ckp['epoch']+1
        max_ssim=ckp['max_ssim']
        max_psnr=ckp['max_psnr']
        psnrs=ckp['psnrs']
        ssims=ckp['ssims']
        print(f'start_epoch:{opt.start_epoch} start training ---')
    else:
        print('train from scratch *** ')

    criterion = torch.nn.MSELoss(size_average=True)
    criterion = criterion.to(opt.device)
    optimizer = optim.Adam(net.parameters(), lr=opt.lr)

    for i in range(opt.start_training_step, 3):
        opt.nEpochs   = training_settings[i - 1]['nEpochs']
        opt.lr        = training_settings[i - 1]['lr']
        opt.step      = training_settings[i - 1]['step']
        opt.lr_decay  = training_settings[i - 1]['lr_decay']
        
        for epoch in range(opt.start_epoch, opt.nEpochs):
            loss = 0
            adjust_learning_rate(epoch)
            loss = train(net, loader_train, criterion, optimizer, epoch)
            losses.append(loss.item())
            if epoch % opt.eval_epoch == 0:
                with torch.no_grad():
                    ssim_eval,psnr_eval=test(net,loader_test, max_psnr,max_ssim,epoch)
                ssims.append(ssim_eval)
                psnrs.append(psnr_eval)
                if ssim_eval > max_ssim and psnr_eval > max_psnr :
                    max_ssim=max(max_ssim,ssim_eval)
                    max_psnr=max(max_psnr,psnr_eval)
                    model_dir=os.path.join(opt.model_dir, opt.model, str(opt.start_training_step), "{}.pkl".format(opt.model))
                    torch.save({
                                'step':i,
                                'epoch':epoch,
                                'max_psnr':max_psnr,
                                'max_ssim':max_ssim,
                                'ssims':ssims,
                                'psnrs':psnrs,
                                'losses':losses,
                                'model':net.state_dict()
                    }, model_dir)
                    print(f'\n model saved at epoch:{epoch} of step:{i}| max_psnr:{max_psnr:.4f}|max_ssim:{max_ssim:.4f}')

                model_dir=os.path.join(opt.model_dir, opt.model, str(opt.start_training_step), "{}_latest.pkl".format(opt.model))
                torch.save({
                            'step':i,
                            'epoch':epoch,
                            'max_psnr':max_psnr,
                            'max_ssim':max_ssim,
                            'ssims':ssims,
                            'psnrs':psnrs,
                            'losses':losses,
                            'model':net.state_dict()
                }, model_dir)

        opt.start_epoch = 0
        if not os.path.exists(f'./numpy_files/{opt.model}_step_{i}_losses.npy'):
            np.save(f'./numpy_files/{opt.model}_step_{i}_losses.npy',losses)
            np.save(f'./numpy_files/{opt.model}_step_{i}_ssims.npy',ssims)
            np.save(f'./numpy_files/{opt.model}_step_{i}_psnrs.npy',psnrs)
