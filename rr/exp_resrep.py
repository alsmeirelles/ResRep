from constants import *
from rr.resrep_builder import ResRepBuilder
from rr.resrep_config import ResRepConfig
from rr.resrep_train import resrep_train_main
from base_config import get_baseconfig_by_epoch
from model_map import get_dataset_name_by_model_name
from rr.resrep_scripts import *
from utils.misc import extract_deps_from_weights_file

import argparse
from ndp_test import general_test
import os

def run_rr(cfg,train_dataloader=None,val_dataloader=None,test_dataloader=None,train_cfg=None):

    network_type = cfg.arch
    conti_or_fs = cfg.conti_or_fs
    assert conti_or_fs in ['continue', 'fs']
    auto_continue = conti_or_fs == 'continue'
    print('auto continue: ', auto_continue)
        

    if network_type == 'sres50' or network_type == "swresnet50v2":
        weight_decay_strength = 1e-4
        batch_size = cfg.batch_size
        deps = RESNET50_ORIGIN_DEPS_FLATTENED
        succeeding_strategy = resnet_bottleneck_succeeding_strategy(50)
        print(succeeding_strategy)
        pacesetter_dict = resnet_bottleneck_follow_dict(50)
        init_hdf5 = 'finish.hdf5'
        flops_func = calculate_resnet_50_flops
        target_layers = RESNET50_INTERNAL_KERNEL_IDXES
        lrs = LRSchedule(base_lr=cfg.learn_r, max_epochs=cfg.epochs, lr_epoch_boundaries=None, lr_decay_factor=None,
                         linear_final_lr=None, cosine_minimum=0)
        resrep_config = ResRepConfig(target_layers=target_layers, succeeding_strategy=succeeding_strategy,
                                     pacesetter_dict=pacesetter_dict, lasso_strength=1e-4,
                                     flops_func=flops_func, flops_target=0.455, mask_interval=200,
                                     compactor_momentum=0.99, before_mask_iters=5*1281167//batch_size,
                                     begin_granularity=4, weight_decay_on_compactor=False, num_at_least=1)

    elif network_type == 'src56':
        weight_decay_strength = 1e-4
        batch_size = cfg.batch_size
        deps = rc_origin_deps_flattened(9)
        succeeding_strategy = rc_succeeding_strategy(9)
        pacesetter_dict = rc_pacesetter_dict(9)
        flops_func = calculate_rc56_flops
        init_hdf5 = 'src56_train/finish.hdf5'
        target_layers = rc_internal_layers(9)
        lrs = LRSchedule(base_lr=0.01, max_epochs=480, lr_epoch_boundaries=None, lr_decay_factor=None,
                         linear_final_lr=None, cosine_minimum=0)
        resrep_config = ResRepConfig(target_layers=target_layers, succeeding_strategy=succeeding_strategy,
                                     pacesetter_dict=pacesetter_dict, lasso_strength=1e-4,
                                     flops_func=flops_func, flops_target=0.471, mask_interval=200,
                                     compactor_momentum=0.99, before_mask_iters=5 * 50000 // batch_size,
                                     begin_granularity=4, weight_decay_on_compactor=False, num_at_least=1)

    else:
        raise ValueError('...')

    log_dir = os.path.join(cfg.output_dir,'{}_train'.format(network_type))

    weight_decay_bias = 0
    warmup_factor = 0

    config = get_baseconfig_by_epoch(network_type=network_type,
                                     dataset_name=cfg.dataset_name, dataset_subset='train',
                                     global_batch_size=cfg.batch_size, num_node=1,
                                     weight_decay=weight_decay_strength, optimizer_type='sgd', momentum=0.9,
                                     max_epochs=lrs.max_epochs, base_lr=lrs.base_lr, lr_epoch_boundaries=lrs.lr_epoch_boundaries, cosine_minimum=lrs.cosine_minimum,
                                     lr_decay_factor=lrs.lr_decay_factor,
                                     warmup_epochs=0, warmup_method='linear', warmup_factor=warmup_factor,
                                     ckpt_iter_period=40000, tb_iter_period=100, output_dir=cfg.output_dir,
                                     tb_dir=log_dir, save_weights=None, val_epoch_period=0 if val_dataloader is None else 2,
                                     linear_final_lr=lrs.linear_final_lr,
                                     weight_decay_bias=weight_decay_bias, deps=deps, device=cfg.device)

    resrep_builder = ResRepBuilder(base_config=config, resrep_config=resrep_config)

    if resrep_config.weight_decay_on_compactor:
        no_l2_keywords = ['depth']
    else:
        no_l2_keywords = ['depth', 'compactor']
        
    print('######################################################')
    print('start here, the original flops is ', flops_func(deps))
    print('######################################################')

    conv_weights = os.path.join(cfg.output_dir,  'finish_converted.hdf5')
    rmodel = None
    if not os.path.exists(conv_weights):
        rmodel = resrep_train_main(local_rank=cfg.local_rank, 
                          cfg=config, resrep_config=resrep_config, resrep_builder=resrep_builder, show_variables=True,
                          init_hdf5=init_hdf5, net=cfg.model, train_dataloader=train_dataloader, val_dataloader=val_dataloader,
                          auto_continue=auto_continue,train_cfg = train_cfg,
                          no_l2_keywords=no_l2_keywords,num_classes=cfg.num_classes)


    rdeps = extract_deps_from_weights_file(conv_weights)
    print('######################################################')
    print('After convertion, the reduced flops is ', flops_func(rdeps))
    print('######################################################')

    #Runs test from main script
    #general_test(network_type=network_type,test_dataloader=test_dataloader,
    #             weights=conv_weights, net=cfg.model, num_classes=cfg.num_classes,
    #             builder=ResRepBuilder(base_config=config, resrep_config=resrep_config,
    #                                   mode='deploy'))


    return rmodel

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--arch', default='src56')
    parser.add_argument('-c', '--conti_or_fs', default='fs')
    parser.add_argument(
        '--local_rank', default=0, type=int,
        help='process rank on node')
    parser.add_argument(
        '--num_classes', default=1000, type=int,
        help='Number of classes in the dataset')

    config = parser.parse_args()
    run_rr(config)

