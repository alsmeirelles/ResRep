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
import base_model
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
    """
    Dataset specific parameters
    """
    pf_form = '(UN-(?P<unc>[0-9])+-){,1}(?P<tcga>TCGA-.*-.*-.*-.*-.*)-(?P<x>[0-9]+)-(?P<y>[0-9]+)-(?P<s1>[0-9]+)-(?P<s2>[0-9]+)(_(?P<lb>[01])){,1}\\.png'
    num_classes = 2
    name = 'TILDataset'
    input_space = 'RGB'
    input_range = [0,1]
    
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

    def get_subset(self,indices,replace=False):
        ss = object.__new__(TILDataset)
        arr = np.asarray(self._data)
        setattr(ss,'_path',self._path)
        setattr(ss,'_data',list(arr[indices]))
        setattr(ss,'_imsize',self._imsize)
        setattr(ss,'_augmenter',self._augmenter)
        setattr(ss,'_keep',self._keep)
        
        if replace:
            self._data = list(np.delete(arr,indices))

        return ss

def main_exec(config):
    """
    Main execution line. Dispatch processes according to parameter groups.
    Multiple processes here prevent main process from consuming too much memory.
    """

    if not os.path.isdir(config.weights_path):
        os.mkdir(config.weights_path)

    # use our dataset and defined transformations
    dataset = TILDataset(config.data, augment=config.augment,keep=config.keepimg,imsize=config.tdim,proc=config.cpu_count,cache=config.cache)
    if not config.test is None:
        dataset_test = TILDataset(config.test, augment=False,keep=False,imsize=config.tdim,proc=config.cpu_count,cache=config.cache)
    elif not config.tsize is None:
        # split the dataset in train and test set
        indices = torch.randperm(len(dataset)).tolist()
        dataset_test = dataset.get_subset(indices[-config.tsize:],replace=True)
        dataset_test._augmenter = None

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.cpu_count,pin_memory=config.keepimg)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=config.cpu_count)    

    #Needed both for training and testing
    model = None
    criterion = None
    lr_scheduler = None
    optimizer = None
    
    ## Main execution sequence
    if config.rrtrain:
        from ndp_train import train_main
        from base_config import BaseConfigByEpoch
        from constants import LRSchedule
        from builder import ConvBuilder

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        lrs = LRSchedule(base_lr=0.1, max_epochs=config.epochs, lr_epoch_boundaries=[120, 180], lr_decay_factor=0.1,
                         linear_final_lr=None, cosine_minimum=None)
        
        bcfg = BaseConfigByEpoch(network_type=config.network, dataset_name='TILDataset',
                                     dataset_subset='train',
                                     global_batch_size=config.batch_size,
                                     device=device,
                                     num_node=1,
                                     weight_decay=1e-4,
                                     optimizer_type='AdamW',
                                     momentum=0.9,
                                     max_epochs=lrs.max_epochs,
                                     base_lr=lrs.base_lr,
                                     lr_epoch_boundaries=lrs.lr_epoch_boundaries,
                                     lr_decay_factor=lrs.lr_decay_factor,
                                     bias_lr_factor=2,
                                     cosine_minimum=lrs.cosine_minimum,
                                     warmup_epochs=0,
                                     warmup_method='linear',
                                     warmup_factor=0,
                                     ckpt_iter_period=40000,
                                     tb_iter_period=100,
                                     output_dir=config.temp,
                                     tb_dir=config.temp,
                                     save_weights=None,
                                     init_weights=None, #Initial weights as checkpoint continuation
                                     val_epoch_period=0,
                                     linear_final_lr=lrs.linear_final_lr,
                                     weight_decay_bias=0,
                                     grad_accum_iters=1,
                                     se_reduce_scale=0,
                                     se_layers=None,
                                     deps=None)

        convbuilder = ConvBuilder(base_config=bcfg)
        model = getattr(base_model,config.network)(bcfg, convbuilder, num_classes = dataset.num_classes)
        model.input_size = (3,) + config.tdim
        model.input_space = dataset.input_space
        model.input_range = dataset.input_range
        
        params = [p for p in model.parameters() if p.requires_grad]
        #optimizer = torch.optim.SGD(params, lr=config.learn_r,
        #                            momentum=0.9, weight_decay=0.0005)
        optimizer = torch.optim.Adam(params,lr=config.learn_r,weight_decay=0.0)
    
        # and a learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                        step_size=5,
                                                        gamma=0.5)
        criterion = torch.nn.CrossEntropyLoss()
        
        # move model to the right device
        device = torch.device(device)
        model.to(device)
        train_conf = (optimizer,lr_scheduler,criterion)
        train_main(local_rank=config.lrank,cfg=bcfg,net=model,train_dataloader=data_loader,
                       val_dataloader=None,init_hdf5=None,train_conf=train_conf,show_variables=True,num_gpus=config.gpu_count)

    elif config.train:
        ## Common definitions
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        # get the model using our helper function
        model = getattr(base_model,config.network)(num_classes=dataset.num_classes,pretrained=None)
        model.input_size = (3,) + config.tdim
        model.input_space = dataset.input_space
        model.input_range = dataset.input_range

        params = [p for p in model.parameters() if p.requires_grad]
        #optimizer = torch.optim.SGD(params, lr=config.learn_r,
        #                            momentum=0.9, weight_decay=0.0005)
        optimizer = torch.optim.Adam(params,lr=config.learn_r,weight_decay=0.0005)
    
        # and a learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                        step_size=5,
                                                        gamma=0.1)
        criterion = torch.nn.CrossEntropyLoss()        
        
        # move model to the right device
        model.to(device)        

        other_args = {'print_freq':10, 'clip_grad_norm':None, 'lr_warmup_epochs':0, 'model_ema_steps':32}
        other_args = SimpleNamespace(**other_args)

        run_engine = engine.Engine(local_rank=config.lrank)
        run_engine.setup_log(name='train', log_dir=config.temp, file_name='log.txt')
        run_engine.register_state(scheduler=lr_scheduler, model=model, optimizer=optimizer)
        run_engine.show_variables()
        
        for epoch in range(config.epochs):
            # train for one epoch, printing every 10 iterations
            alu.train_one_epoch(model, criterion, optimizer, data_loader, device, epoch,args=other_args)
            # update the learning rate
            lr_scheduler.step()
            # evaluate on the test dataset
            if epoch > 0 and epoch % config.eval_freq == 0:
                alu.evaluate(model, criterion, data_loader_test, device=device)

        run_engine.save_hdf5(os.path.join(config.weights_path, '{}-finish.hdf5'.format(config.network)))

    elif config.resrep:
        from rr.exp_resrep import run_rr

        model = getattr(base_model,config.network)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        other_args = {'arch':config.network, 'conti_or_fs':'fs', 'dataset_name':'TILDataset','batch_size':config.batch_size,'output_dir':config.temp,
                        'local_rank':config.lrank,'model':model,'num_classes':dataset.num_classes,'device':device, 'learn_r':config.learn_r,'epochs':config.epochs}
        other_args = SimpleNamespace(**other_args)

        if not optimizer is None and lr_scheduler is None:
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                        step_size=5,
                                                        gamma=0.5)

        if criterion is None:
            criterion = torch.nn.CrossEntropyLoss()

        train_conf = (optimizer,lr_scheduler,criterion)
        run_rr(other_args,train_dataloader=data_loader,test_dataloader=data_loader_test,train_cfg=train_conf)
        config.predict = True
        model = None

    if config.predict:
        from base_config import get_baseconfig_for_test
        from builder import ConvBuilder
        from utils.misc import extract_deps_from_weights_file
        from rr.resrep_scripts import calculate_resnet_bottleneck_flops
        from rr.resrep_builder import ResRepBuilder
        from rr.resrep_config import ResRepConfig
        import constants
        
        weights_file = os.path.join(config.weights_path,'finish_converted.hdf5')
        rdeps = None
        #Load model if one was not just trained
        if model is None:
            print("Building new model")
            
            if not os.path.isfile(weights_file):
                weights_file = os.path.join(config.weights_path,'finish.hdf5')
                
            if not os.path.isfile(weights_file):
                print("No model available and no weights found: {}".format(weights_file))
                sys.exit(1)

            rdeps = extract_deps_from_weights_file(weights_file)
            cfg = get_baseconfig_for_test(network_type=config.network, dataset_subset='test', global_batch_size=config.batch_size,
                                              init_weights=weights_file, deps=rdeps, dataset_name='TILDataset')
            
            convbuilder = ConvBuilder(base_config=cfg)                
            
            model = getattr(base_model,config.network)(cfg, convbuilder, num_classes = dataset.num_classes)
            model.input_size = (3,) + config.tdim
            model.input_space = dataset.input_space
            model.input_range = dataset.input_range

            test_engine = engine.Engine(local_rank=0, for_val_only=True)
            test_engine.setup_log(name='test', log_dir=config.temp, file_name='test_detail.log')
            test_engine.register_state(scheduler=None, model=model, optimizer=None)
            test_engine.load_hdf5(weights_file)
            criterion = torch.nn.CrossEntropyLoss()
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            if device == 'cuda':
                model.cuda()
            
        alu.evaluate(model, criterion, data_loader_test, device=device,calc_auc=True)
        #TODO: calculate FLOPS
        if rdeps is None and not weights_file is None and os.path.isfile(weights_file):
            rdeps = extract_deps_from_weights_file(weights_file)

        if rdeps is None and hasattr(model,'deps'):
            rdeps = model.deps
        else:
            print("No deps available")
            sys.exit(1)
        flops = calculate_resnet_bottleneck_flops(rdeps,model.num_blocks)
        print("Model FLOPS: {}".format(flops))
        
if __name__ == "__main__":

    #Parse input parameters
    arg_groups = []
    parser = argparse.ArgumentParser(description='Convolunional Neural \
        Network for Active Learning.')

    ##Training options
    train_args = parser.add_argument_group('Training','Common network training options')
    arg_groups.append(train_args)

    train_args.add_argument('--rrtrain', action='store_true', dest='rrtrain', default=False, 
        help='Train model with ResRep system')
    
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
        default=None)
    train_args.add_argument('-tdim', dest='tdim', nargs='+', type=int, 
        help='Tile width and heigth, optionally inform the number of channels (Use: 200 200 for SVS 50 um).', 
        default=None, metavar=('Width', 'Height'))
    train_args.add_argument('-k', action='store_true', dest='keepimg', default=False, 
        help='Keep loaded images in memory.')

    ##Predictions
    pred_args = parser.add_argument_group('Predictions')
    arg_groups.append(pred_args)
    pred_args.add_argument('--pred', action='store_true', dest='predict', default=False, 
        help='Runs prediction with a given model (use -net parameter).')
    pred_args.add_argument('-test',dest='test',type=str,help='Test set path',default=None)
    pred_args.add_argument('-tsize', dest='tsize', type=int, 
        help='Size of test set', default=None)
    pred_args.add_argument('-eval', dest='eval_freq', type=int, 
        help='Run evaluation on test set every -eval epochs (Default: 5).', default=5)
    
    ##Hardware configurations
    hd_args = parser.add_argument_group('Hardware')
    arg_groups.append(hd_args)

    hd_args.add_argument('-gpu', dest='gpu_count', type=int, 
        help='Number of GPUs available (Default: 0).', default=0)
    hd_args.add_argument('-cpu', dest='cpu_count', type=int, 
        help='Number of CPU cores available (Default: 1).', default=1)
    hd_args.add_argument('-dist-url', default="env://", type=str, help="URL used to set up distributed training")
    hd_args.add_argument('-world-size', default=1, type=int, help="Number of distributed processes")
    hd_args.add_argument('-lrank', dest='lrank', default=0, type=int, help="Local rank on node.")

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

    parser.add_argument('--rr', action='store_true', dest='resrep', default=False, 
        help='Runs reduction on a given model (use -net parameter).')


    config, unparsed = parser.parse_known_args()

    if config.weights_path is None:
        config.weights_path = config.temp
        
    if not os.path.isdir(config.cache):
        os.mkdir(config.cache)

    if not os.path.isdir(config.temp):
        os.mkdir(config.temp)

    if not config.tdim is None:
        config.tdim = tuple(config.tdim)

    #if config.gpu_count > 0:
    #    tu.init_distributed_mode(config)
        
    main_exec(config)
    
