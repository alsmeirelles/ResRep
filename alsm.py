#!/usr/bin/env python3
#-*- coding: utf-8

import os
import argparse
import sys
import importlib
import numpy as np
import pickle
import torch
import re
import pretrainedmodels
import multiprocessing
import imgaug as ia
from imgaug import augmenters as iaa
import skimage
from skimage import io
from types import SimpleNamespace

#Locals
from utils import alsm_utils as alu
from utils import engine
from utils import training_utils as tu

#Filter warnings
import warnings
warnings.filterwarnings('ignore')

class PImage(object):
    def __init__(self,path,arr=None,keepImg=False,origin=None,coord=None,label=None,verbose=0):
        """
        @param path <str>: path to image
        @param arr <ndarray>: numpy array with image data
        @param keepImg <bool>: keep image data in memory
        @param origin <str>: current image is originated from origin
        @param coord <tuple>: coordinates in original image
        @param label <int or string>: class label
        """
        if not arr is None and isinstance(arr,np.ndarray):
            self._data = arr
        else:
            self._data = None

        if not label is None and isinstance(label,str):
            self._label = int(label)
        else:
            self._label = label
            
        self._coord = coord
        self._origin = origin
        self._path = path
        self._verbose = verbose
        self._keep = keepImg
        self._dim = None        
        
    def __str__(self):
        """
        String representation is (coord)-origin if exists, else, file name
        """
        if not (self._coord is None and self._origin is None):
            return "{0}-{1}".format(self._coord,self._origin)
        else:
            return os.path.basename(self._path)

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        # Hashes current dir and file name
        return hash((self._path.split(os.path.sep)[-2],os.path.basename(self._path)))
    
    def readImage(self,keepImg=None,size=None,verbose=None,toFloat=True):
        
        data = None

        if keepImg is None:
            keepImg = self._keep
        elif keepImg:
            #Change seting if we are going to keep the image in memory now
            self.setKeepImg(keepImg)
        if not verbose is None:
            self._verbose = verbose
            
        if self._data is None or size != self._dim:
            if self._verbose > 1:
                print("Reading image: {0}".format(self._path))
                
            data = io.imread(self._path)
            
            if(data.shape[2] > 3): # remove the alpha
                data = data[:,:,0:3]
                
            if not size is None and data.shape != size:
                if self._verbose > 1:
                    print("Resizing image {0} from {1} to {2}".format(os.path.basename(self._path),data.shape,size))
                data = skimage.transform.resize(data,size)

            #Convert data to float and also normalizes between [0,1]
            if toFloat:
                data = skimage.img_as_float32(data)
            else:
                data = skimage.img_as_ubyte(data)
                
            h,w,c = data.shape
            self._dim = (w,h,c)
            
            if self._keep:
                self._data = data
                
        else:
            if self._verbose > 1:
                print("Data already loaded:\n - {0}".format(self._path))
            if not toFloat and self._data.dtype != np.uint8:
                self._data = skimage.img_as_ubyte(self._data)
            data = self._data

        return data

    def getLabel(self):
        return self._label
    
class TILDataset(torch.utils.data.Dataset):
    pf_form = '(UN-(?P<unc>[0-9])+-){,1}(?P<tcga>TCGA-.*-.*-.*-.*-.*)-(?P<x>[0-9]+)-(?P<y>[0-9]+)-(?P<s1>[0-9]+)-(?P<s2>[0-9]+)(_(?P<lb>[01])){,1}\\.png'
    num_classes = 2
    name = 'TILDataset'
    def __init__(self,path,augment=False,keep=False,imsize=None,proc=2,cache='cache'):
        """
        TIL Dataset data generator

        @param path <str> Path to patches
        @param proc <int> Multiprocessing run (number of processes)
        """
        if not os.path.isdir(path):
            raise RuntimeError("No such directory: {}".format(path))

        self._path = path
        self._data = self.generate_set(self._path,cache,proc)
        self._imsize = imsize
        if augment:
            self._augmenter = iaa.SomeOf(3,[iaa.AddToBrightness((-30,30)),
                                          iaa.AddToSaturation((-50,50)),
                                          iaa.LinearContrast((0.4,1.6)),
                                          #iaa.Rotate((0,22.5)),
                                          iaa.Fliplr(),
                                          iaa.Flipud(),
                                          iaa.KeepSizeByResize(iaa.CenterCropToFixedSize(width=self._imsize[0]-20,height=self._imsize[1]-20))
                                            ])
        else:
            self._augmenter = None

        self._keep = keep

    def __len__(self):
        return len(self._data)

    def __getitem__(self,idx):
        toFloat = self._augmenter is None
        img = self._data[idx].readImage(keepImg=self._keep,size=self._imsize,verbose=0,toFloat=toFloat)

        if not self._augmenter is None:
            img = self.applyDataAugmentation(img)
            img = img.astype(np.float32)
            img /= 255

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        img = img.transpose((2, 0, 1))

        return torch.from_numpy(np.ascontiguousarray(img)),self._data[idx].getLabel()

    def _run_dir(self,path):

        rex = re.compile(self.pf_form)
        patches = list(filter(lambda f:f.endswith('.png'),os.listdir(path)))

        rt = []
        for p in patches:
            pm = rex.match(p)
            if pm is None:
                continue
            seg = PImage(os.path.join(path,p),keepImg=False,origin=pm.group('tcga'),coord=(pm.group('x'),pm.group('y')),label=pm.group('lb'),
                            verbose=1)
            rt.append(seg)

        return rt

    def applyDataAugmentation(self,batch_x):
        #Additional data augmentation
        dimensions = len(batch_x.shape)
        if dimensions == 4:
            return self._augmenter(images=batch_x)
        elif dimensions == 3:
            return self._augmenter(images=[batch_x])[0]
        else:
            return None
    
    def generate_set(self,path,cache,processes=2):
        """
        Returns and ndarray of randomly selected items from data pool
        """
        cache_file = os.path.join(cache,'TILDS-{}.pik'.format(os.path.basename(self._path)))
        if os.path.isfile(cache_file):
            with open(cache_file,'rb') as fd:
                rt,name = pickle.load(fd)
            if name == self.name:
                print("[TILDataset] Dataset cache loaded.")
                return rt
            
        dlist = []
        files = os.listdir(path)

        dirs = list(filter(lambda i:os.path.isdir(os.path.join(path,i)),files))
        multi_dir = True if len(dirs) > 0 else False

        rt = []
        if multi_dir:
            results = None
            with multiprocessing.Pool(processes=processes) as pool:
                results = [pool.apply_async(self._run_dir,(os.path.join(path,d),)) for d in dirs]
                for r in results:
                    rt.extend(r.get())
        else:
            rt = self._run_dir(path)

        with open(cache_file,'wb') as fd:
            pickle.dump((rt,self.name),fd)

        return rt

def main_exec(config):
    """
    Main execution line. Dispatch processes according to parameter groups.
    Multiple processes here prevent main process from consuming too much memory.
    """

    if not os.path.isdir(config.weights_path):
        os.mkdir(config.weights_path)

    if config.train:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # use our dataset and defined transformations
        dataset = TILDataset(config.data, augment=config.augment,keep=config.keepimg,imsize=config.tdim,proc=config.cpu_count,cache=config.cache)
        if not config.test is None:
            dataset_test = TILDataset(config.test, augment=False,keep=False,imsize=config.tdim,proc=config.cpu_count,cache=config.cache)
        elif not config.split is None:
            # split the dataset in train and test set
            indices = torch.randperm(len(dataset)).tolist()
            dataset = torch.utils.data.Subset(dataset, indices[:-config.tsize])
            dataset_test = torch.utils.data.Subset(dataset_test, indices[-config.tsize:])

        # define training and validation data loaders
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.cpu_count,pin_memory=config.keepimg)

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=1, shuffle=False, num_workers=config.cpu_count)

        # get the model using our helper function
        model = getattr(pretrainedmodels,config.network)(num_classes=dataset.num_classes,pretrained=None)

        # move model to the right device
        model.to(device)        

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=config.learn_r,
                                    momentum=0.9, weight_decay=0.0005)
        # and a learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=3,
                                                    gamma=0.1)
        criterion = torch.nn.CrossEntropyLoss()

        other_args = {'print_freq':10, 'clip_grad_norm':None, 'lr_warmup_epochs':0, 'model_ema_steps':32}
        other_args = SimpleNamespace(**other_args)
        
        for epoch in range(config.epochs):
            # train for one epoch, printing every 10 iterations
            alu.train_one_epoch(model, criterion, optimizer, data_loader, device, epoch,args=other_args)
            # update the learning rate
            lr_scheduler.step()
            # evaluate on the test dataset
            alu.evaluate(model, criterion, data_loader_test, device=device)

    elif config.predict:
        pass
        
if __name__ == "__main__":

    #Parse input parameters
    arg_groups = []
    parser = argparse.ArgumentParser(description='Convolunional Neural \
        Network for Active Learning.')

    ##Training options
    train_args = parser.add_argument_group('Training','Common network training options')
    arg_groups.append(train_args)

    train_args.add_argument('--train', action='store_true', dest='train', default=False, 
        help='Train model')
    train_args.add_argument('-net',dest='network',type=str,default='inceptionv4',help='Network name which should be trained.\n \
    Check documentation for available models.')
    train_args.add_argument('-data',dest='data',type=str,help='Dataset path',default=None)
    train_args.add_argument('-b', dest='batch_size', type=int, 
        help='Batch size (Default: 8).', default=8)
    train_args.add_argument('-lr', dest='learn_r', type=float, 
        help='Learning rate (Default: 0.00005).', default=0.00005)
    train_args.add_argument('-e', dest='epochs', type=int, 
        help='Number of epochs (Default: 1).', default=1)
    train_args.add_argument('-aug', action='store_true', dest='augment',
        help='Applies data augmentation during training.',default=False)
    train_args.add_argument('-wpath', dest='weights_path',
        help='Use weights file contained in path - usefull for sequential training (Default: None).',
        default='ModelWeights')
    train_args.add_argument('-tdim', dest='tdim', nargs='+', type=int, 
        help='Tile width and heigth, optionally inform the number of channels (Use: 200 200 for SVS 50 um).', 
        default=None, metavar=('Width', 'Height'))
    train_args.add_argument('-k', action='store_true', dest='keepimg', default=False, 
        help='Keep loaded images in memory.')

    ##Predictions
    pred_args = parser.add_argument_group('Predictions')
    arg_groups.append(pred_args)
    pred_args.add_argument('--pred', action='store_true', dest='pred', default=False, 
        help='Runs prediction with a given model (use -net parameter).')
    pred_args.add_argument('-test',dest='test',type=str,help='Test set path',default=None)
    pred_args.add_argument('-tsize', dest='tsize', type=int, 
        help='Size of test set', default=None)
    
    ##Hardware configurations
    hd_args = parser.add_argument_group('Hardware')
    arg_groups.append(hd_args)

    hd_args.add_argument('-gpu', dest='gpu_count', type=int, 
        help='Number of GPUs available (Default: 0).', default=0)
    hd_args.add_argument('-cpu', dest='cpu_count', type=int, 
        help='Number of CPU cores available (Default: 1).', default=1)

    ##Runtime options
    parser.add_argument('-out', dest='temp', type=str,default='temp', 
        help='Base dir to store all temporary data and general output')
    parser.add_argument('-cache', dest='cache', type=str,default='cache', 
        help='Keeps caches in this directory',required=False)
    parser.add_argument('-v', action='count', default=0, dest='verbose',
        help='Amount of verbosity (more \'v\'s means more verbose).')
    parser.add_argument('-i', action='store_true', dest='info', default=False, 
        help='Return general info about data input, the CNN, etc.')
    parser.add_argument('-d', action='store_true', dest='delay_load', default=False, 
        help='Delay the loading of images to the latest moment possible (memory efficiency).')


    config, unparsed = parser.parse_known_args()

    if not os.path.isdir(config.cache):
        os.mkdir(config.cache)

    if not os.path.isdir(config.temp):
        os.mkdir(config.temp)

    if not config.tdim is None:
        config.tdim = tuple(config.tdim)

    if config.gpu_count > 0:
        tu.init_distributed_mode(config)
        
    main_exec(config)
    
