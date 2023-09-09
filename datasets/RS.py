import torch.utils.data as data
import torchvision.transforms as tfs
from torchvision.transforms import functional as FF
import random
from torch.utils.data import DataLoader
import sys
import os
from PIL import Image
from option import opt
sys.path.append('.')
sys.path.append('..')

BS=opt.batchSize
print(BS)
crop_size='whole_img'
if opt.crop:
    crop_size=opt.crop_size


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", "JPG"])

class RSHAZE_Dataset(data.Dataset):
    def __init__(self,path,train,size=crop_size,format='.png'):
        self.size = size
        self.train = train
        self.format = format

        self.haze_imgs_dir=os.listdir(os.path.join(path,'input'))
        self.haze_imgs=[os.path.join(path,'input',img) for img in self.haze_imgs_dir]
        self.clear_dir=os.path.join(path,'gt')

    def __len__(self):
        return len(self.haze_imgs)

    def __getitem__(self, index):
        haze=Image.open(self.haze_imgs[index])
        if isinstance(self.size,int):
            while haze.size[0]<self.size or haze.size[1]<self.size :
                index=random.randint(0,9000)
                haze=Image.open(self.haze_imgs[index])        
        img=self.haze_imgs[index]
        clear_name=img.split('/')[-1]
        clear=Image.open(os.path.join(self.clear_dir,clear_name))
        clear=tfs.CenterCrop(haze.size[::-1])(clear)
        if not isinstance(self.size,str):
            i,j,h,w=tfs.RandomCrop.get_params(haze,output_size=(self.size,self.size))
            haze=FF.crop(haze,i,j,h,w)
            clear=FF.crop(clear,i,j,h,w)
        haze,clear=self.augData(haze.convert("RGB") ,clear.convert("RGB") )
        return haze,clear

    def augData(self,data,target):
        if self.train:
            rand_hor=random.randint(0,1)
            rand_rot=random.randint(0,3)
            data=tfs.RandomHorizontalFlip(rand_hor)(data)
            target=tfs.RandomHorizontalFlip(rand_hor)(target)
            if rand_rot:
                data=FF.rotate(data,90*rand_rot)
                target=FF.rotate(target,90*rand_rot)
        data=tfs.ToTensor()(data)
        data=tfs.Normalize(mean=[0.64, 0.6, 0.58],std=[0.14,0.15, 0.152])(data)
        target=tfs.ToTensor()(target)
        return  data ,target


pwd=os.getcwd()
print(pwd)
path='/home/louanqi/pycharmp/data'#path to your 'data' folder

RSHAZE_train_loader=DataLoader(dataset=RSHAZE_Dataset(path+'/rshaze/train',train=True,size=crop_size),batch_size=BS,shuffle=True)
RSHAZE_test_loader=DataLoader(dataset=RSHAZE_Dataset(path+'/rshaze/test',train=False,size='whole img'),batch_size=1,shuffle=False)

RSDota_train_loader=DataLoader(dataset=RSHAZE_Dataset(path+'/dota2/rsdota3000',train=True,size=crop_size),batch_size=BS,shuffle=True)
RSDota_test_loader=DataLoader(dataset=RSHAZE_Dataset(path+'/dota2/rsdota3000',train=False,size='whole img'),batch_size=1,shuffle=False)


if __name__ == "__main__":
    pass