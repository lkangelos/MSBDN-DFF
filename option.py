# Training settings
import argparse
import torch
import os

parser = argparse.ArgumentParser(description="PyTorch Train")
parser.add_argument('--device',type=str,default='Automatic detection')
parser.add_argument("--batchSize", type=int, default=16, help="Training batch size")
parser.add_argument('--crop',action='store_true')
parser.add_argument('--crop_size',type=int,default=240,help='Takes effect when using --crop ')
parser.add_argument("--start_training_step", type=int, default=1, help="Training step")
parser.add_argument("--step", type=int, default=7, help="Change the learning rate for every 30 epochs")
parser.add_argument("--start-epoch", type=int, default=0, help="Start epoch from 0")
parser.add_argument("--lr_decay", type=float, default=0.5, help="Decay scale of learning rate, default=0.5")
parser.add_argument("--resume", default="models/MSBDN-DFF-v1-1/1/MSBDN-DFF-v1-1_latest.pkl", type=str, help="Path to checkpoint (default: none)")
parser.add_argument('--eval_epoch',type=int,default=1)
parser.add_argument("--isTest", type=bool, default=False, help="Test or not")
parser.add_argument('--dataset', default="../data/rshaze/", type=str, help='Path of the training dataset(.h5)')
parser.add_argument('--model', default='MSBDN-DFF-v1-1', type=str, help='Import which network')
parser.add_argument('--name', default='MSBDN-DFF-v1-1', type=str, help='Filename of the training models')
parser.add_argument('--gpu_ids', type=str, default='3,5,6,9', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument("--train_step", type=int, default=1, help="Activated gate module")
parser.add_argument("--clip", type=float, default=0.25, help="Clipping Gradients. Default=0.1")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate, default=1e-4")
parser.add_argument('--model_dir',type=str,default='./models/')
parser.add_argument('--trainset',type=str,default='rshaze_train')
parser.add_argument('--testset',type=str,default='rshaze_test')

opt = parser.parse_args()
opt.device='cuda' if torch.cuda.is_available() else 'cpu'
# opt.model_dir=os.path.join(opt.model_dir, opt.model, str(opt.start_training_step), "{}.pkl".format(opt.model))

print(opt)
print('model_dir:',opt.model_dir)

root_folder = os.path.abspath('.')
models_folder = os.path.join(root_folder, 'models')
models_folder = os.path.join(models_folder, opt.name)
step1_folder, step2_folder, step3_folder = os.path.join(models_folder,'1'), os.path.join(models_folder,'2'), os.path.join(models_folder, '3')
isexists = os.path.exists(step1_folder) and os.path.exists(step2_folder)
if not isexists:
    os.makedirs(step1_folder)
    os.makedirs(step2_folder)
    os.makedirs(step3_folder)
    print("===> Step training models store in models/1 & /2 & /3.")

if not os.path.exists('./numpy_files'):
    os.makedirs('./numpy_files')