# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn
import json
from detectron2.utils.events import get_event_storage
from detectron2.config import configurable
from detectron2.structures import ImageList, Instances, Boxes, ROIMasks
import detectron2.utils.comm as comm
import random
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
# from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.utils.visualizer import Visualizer, _create_text_labels
from detectron2.data.detection_utils import convert_image_to_rgb
import torch.distributed as dist
from torch.cuda.amp import autocast
from ..text.text_encoder import build_text_encoder
from ..utils import load_class_freq, get_fed_loss_inds
import os,torchshow
import inspect
import json
# allgather
from detectron2.utils.comm import get_world_size, all_gather
from collections import deque
class DynamicThreshold:
    def __init__(self, buffer_size=100, percentile=0.85):
        # 初始化一个固定长度的队列
        self.queue = deque(maxlen=buffer_size)
        # 保存设置的百分位数值
        self.percentile = percentile*100

    def add_score(self, score):
        # 添加新的预测分数到队列
        self.queue.append(score)
    def set_percentile(self, percentile):
        # 设置百分位数值
        self.percentile = percentile*100

    def get_threshold(self):
        # 计算并返回当前的动态阈值
        if len(self.queue) == 0:
            return 0
        else:
            return np.percentile(np.array(self.queue), self.percentile)
@META_ARCH_REGISTRY.register()
class CustomRCNN(GeneralizedRCNN):
    '''
    Add image labels
    '''
    @configurable
    def __init__(
        self, 
        with_image_labels = False,
        dataset_loss_weight = [],
        fp16 = False,
        sync_caption_batch = False,
        roi_head_name = '',
        cap_batch_ratio = 4,
        with_caption = False,
        dynamic_classifier = False,
        freeze_backbone = False,
        active_select = False,
        active_mode = '',
        active_loss = '',
        active_loss_update = '',
        active_compare = '',
        active_lr = 0.0001,
        active_optimizer = '',
        active_optim_mode = '', # 'sgd' or 'adam'
        output_dir = '',
        active_pred = False,
        active_pred_choose = '',
        active_pred_sup = '',
        active_only_gt_train = False,
        active_only_gt_test = False,
        #ONLY_PASTE_SUP
        only_paste_sup = False,
        active_seed = 0,
        active_grad_compare = False,
        active_grad_norm = False,
        active_grad_save = False,
        active_grad_update = '',
        active_forward_once = False,
        active_once_mode = '',
        active_eval = False,
        active_test_batchsize = 4,
        cfg = None,
        **kwargs):
        """
        """
        self.with_image_labels = with_image_labels
        self.dataset_loss_weight = dataset_loss_weight
        self.fp16 = fp16
        self.with_caption = with_caption
        self.sync_caption_batch = sync_caption_batch
        self.roi_head_name = roi_head_name
        self.cap_batch_ratio = cap_batch_ratio
        self.dynamic_classifier = dynamic_classifier
        self.return_proposal = False
        if self.dynamic_classifier:
            self.freq_weight = kwargs.pop('freq_weight')
            self.num_classes = kwargs.pop('num_classes')
            self.num_sample_cats = kwargs.pop('num_sample_cats')
        super().__init__(**kwargs)
        assert self.proposal_generator is not None
        if freeze_backbone:
            for p in self.backbone.parameters():
                print(p)
                p.requires_grad = False
            print("Freeze backbone!")
            self.freeze_backbone = True
        self.active_select = active_select
        self.active_mode = active_mode
        self.active_loss = active_loss
        self.active_loss_update = active_loss_update
        self.active_optimizer = active_optimizer
        self.active_optim_mode = active_optim_mode
        #self.active_optimizer = False
        self.forward_once = active_forward_once
        self.active_once_mode = active_once_mode
        self.active_test_batchsize = active_test_batchsize
        self.cfg = cfg
        if 'dynamic' in self.active_once_mode:
            if 'linear' not in self.active_once_mode:
                filter_rate = float(self.active_once_mode.split('_')[-1])
                self.dynamic_queue = DynamicThreshold(buffer_size=1000, percentile=1-filter_rate)
            else:
                # "only_paste_dynamic_linear_0.3_0.5"
                self.start_rate = float(self.active_once_mode.split('_')[-2])
                self.end_rate = float(self.active_once_mode.split('_')[-1])
                self.dynamic_queue = DynamicThreshold(buffer_size=1000, percentile=1-self.start_rate)
                self.max_iter =  self.cfg.SOLVER.MAX_ITER

        self.active_eval = active_eval

    

        if self.with_caption:
            assert not self.dynamic_classifier
            self.text_encoder = build_text_encoder(pretrain=True)
            for v in self.text_encoder.parameters():
                v.requires_grad = False
        if self.active_select:
            self.lr = active_lr #0.0001
            print('active_lr',self.lr)
            if self.active_optim_mode == 'sgd':
                self.optimizer = torch.optim.SGD(self.parameters(), lr=self.lr) # 
            elif self.active_optim_mode == 'adam':
                self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
                # if no moment
                self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr,betas=(0.0, 0.0))
            #ADAMW
            elif self.active_optim_mode == 'adamw':
                self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
            
            self.rank = str(comm.get_rank())
        self.output_dir = output_dir
        self.count = 0
        self.paste_count = 0
        self.not_paste_count = 0   
        self.iter = 0
        self.active_compare = active_compare
        self.active_only_gt_train = active_only_gt_train
        self.active_only_gt_test = active_only_gt_test
        self.only_paste_sup = only_paste_sup
        self.active_seed = active_seed
        self.active_grad_compare = active_grad_compare
        self.active_grad_norm = active_grad_norm
        self.active_grad_save = active_grad_save
        self.active_grad_update = active_grad_update
        print('active_grad_update', self.active_grad_update)
        if self.active_grad_save:
            self.init_grad_bank()
        self.true_pred_num = 0
        # check whether resume
        if os.path.exists(self.output_dir + '/last_checkpoint'):
            print('resume from', self.output_dir + '/last_checkpoint')
            with open(self.output_dir + '/last_checkpoint', 'r') as f:
                self.iter = int(f.read().split('.')[0].split('_')[1])+1
            print('resume from', self.iter)



        



    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        ret.update({
            'with_image_labels': cfg.WITH_IMAGE_LABELS,
            'dataset_loss_weight': cfg.MODEL.DATASET_LOSS_WEIGHT,
            'fp16': cfg.FP16,
            'with_caption': cfg.MODEL.WITH_CAPTION,
            'sync_caption_batch': cfg.MODEL.SYNC_CAPTION_BATCH,
            'dynamic_classifier': cfg.MODEL.DYNAMIC_CLASSIFIER,
            'roi_head_name': cfg.MODEL.ROI_HEADS.NAME,
            'cap_batch_ratio': cfg.MODEL.CAP_BATCH_RATIO,
            'freeze_backbone': cfg.MODEL.CENTERNET.FREEZE_BACKBONE,
            'active_select': cfg.INPUT.ACTIVE_SELECT,
            'active_mode': cfg.MODEL.ACTIVE_MODE,
            'active_loss': cfg.MODEL.ACTIVE_LOSS,
            'active_loss_update': cfg.MODEL.ACTIVE_LOSS_UPDATE,
            'active_compare': cfg.MODEL.ACTIVE_COMPARE,
            'active_lr': cfg.MODEL.ACTIVE_LR,
            'active_optimizer': cfg.MODEL.ACTIVE_OPTIMIZER,
            'active_optim_mode': cfg.MODEL.ACTIVE_OPTIMIZER_MODE,
            'output_dir': cfg.OUTPUT_DIR,
            'active_pred': cfg.MODEL.ACTIVE_PRED,
            'active_pred_choose': cfg.MODEL.ACTIVE_PRED_CHOOSE,
            'active_pred_sup': cfg.MODEL.ACTIVE_PRED_SUP,
            'active_only_gt_train': cfg.MODEL.ACTIVE_ONLY_GT_TRAIN,
            'active_only_gt_test': cfg.MODEL.ACTIVE_ONLY_GT_TEST,
            'only_paste_sup': cfg.MODEL.ONLY_PASTE_SUP,
            'active_seed': cfg.MODEL.ACTIVE_SEED,
            'active_grad_compare': cfg.MODEL.ACTIVE_GRAD_COMPARE,
            'active_grad_norm': cfg.MODEL.ACTIVE_GRAD_NORM,
            'active_grad_save': cfg.MODEL.ACTIVE_GRAD_SAVE,
            'active_grad_update': cfg.MODEL.ACTIVE_GRAD_UPDATE,
            'active_forward_once': cfg.MODEL.ACTIVE_FORWARD_ONCE,
            'active_once_mode': cfg.MODEL.ACTIVE_ONCE_MODE,
            'active_eval': cfg.MODEL.ACTIVE_EVAL,
            'active_test_batchsize': cfg.MODEL.ACTIVE_TEST_BATCHSIZE,
            'cfg': cfg,
        })
        if ret['dynamic_classifier']:
            ret['freq_weight'] = load_class_freq(
                cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH,
                cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_FREQ_WEIGHT)
            ret['num_classes'] = cfg.MODEL.ROI_HEADS.NUM_CLASSES
            ret['num_sample_cats'] = cfg.MODEL.NUM_SAMPLE_CATS
        return ret


    def inference(
        self,
        batched_inputs: Tuple[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        assert not self.training
        assert detected_instances is None

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        if 'proposals' in  batched_inputs[0]:
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
        else :
            proposals, _ = self.proposal_generator(images, features, None)
        results, _ = self.roi_heads(images, features, proposals)
        if do_postprocess:
            assert not torch.jit.is_scripting(), \
                "Scripting is not supported for postprocess."
            results, inds =  self._postprocess(
                results, batched_inputs, images.image_sizes)
            if hasattr(self.roi_heads, 'save_bbox_features') :
                self.roi_heads.save_bbox_features = self.roi_heads.save_bbox_features[inds]
            return results
        else:
            return results
    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        try:
            images = [x["image"].to(self.device) for x in batched_inputs]
        except:
            images = [x.to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Add ann_type
        Ignore proposal loss when training with image labels
        """
        if not self.training: 
                    
                return self.inference(batched_inputs)
        

        images = self.preprocess_image(batched_inputs)
        ann_type = 'box'
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        paste = True
        self.print_gpu_memory() # 1.65G  / 1.65G
        if self.active_select:
            
            # print('rank:', comm.get_rank(), 'old_weights', old_weights['backbone.fpn_lateral3.weight'][0][0][0][0])
            ori_images = self.preprocess_image([x['origin_image'] for x in batched_inputs])
            test_images = self.preprocess_image([x['test_image'] for x in batched_inputs])
            ori_gt_instances = [x["origin_instances"].to(self.device) for x in batched_inputs]
            test_gt_instances = [x["test_instances"].to(self.device) for x in batched_inputs]
            test_gt_select_classes = [x["test_image_class"] for x in batched_inputs]
            if self.active_test_batchsize > len(ori_images):
                img_list = [x['test_image'] for x in batched_inputs]
                img_list2 = [x['test_image2'] for x in batched_inputs]
                img_list = img_list + img_list2
                test_images = self.preprocess_image(img_list)
                test_gt_instances = test_gt_instances + [x["test_instances2"].to(self.device) for x in batched_inputs]
                test_gt_select_classes = test_gt_select_classes + [x["test_image_class2"] for x in batched_inputs]
            test_gt_all_classes = [set(x["test_instances"].gt_classes.tolist()) for x in batched_inputs]
            test_gt_all_classes = list(set.union(*test_gt_all_classes))
            # compare  gt_instances and ori gt_instances to decide whether paste
            paste_classes = [set(x.gt_classes.tolist())- set(y.gt_classes.tolist()) for x,y in zip(gt_instances, ori_gt_instances)]
            # merge paste_classes
            paste_classes = list(set.union(*paste_classes))
            paste_filename_lists = [x['paste_filename_list'] for x in batched_inputs]
            paste_num_list = [len(x) for x in paste_filename_lists]
            paste_num = sum(paste_num_list)
            def reset_instance_source(gt_instances):
                # reset instance_source [0,0,0,1,1,1] -> [0,0,0,1,2,3]
                total_num = 1
                gt_instances = copy.deepcopy(gt_instances)
                for i, gt_instance in enumerate(gt_instances):
                    paste_num = gt_instance.instance_source.sum()
                    if paste_num>0:
                        assert gt_instance.instance_source[-paste_num:].sum() == paste_num
                        gt_instance.instance_source[-paste_num:] = torch.arange(total_num, total_num+paste_num)
                    total_num += paste_num
                return gt_instances
            gt_instances_ = gt_instances
            gt_instances = reset_instance_source(gt_instances)
            if test_gt_select_classes != paste_classes:
                print('inconsistent select_classes and paste_classes')
            if test_gt_select_classes != test_gt_all_classes:
                print('inconsistent select_classes and test_all_classes')
            # for test_gt_instance in test_gt_instances:
            #     print('test_gt_instance', test_gt_instance.gt_classes)
            if  self.active_test_batchsize < len(test_images):
                test_images = ImageList.from_tensors([x for x in test_images.tensor[:self.active_test_batchsize]], self.backbone.size_divisibility)
                test_gt_instances = test_gt_instances[:self.active_test_batchsize]
            if  self.active_compare != 'all':
                # record old_weights
                if not self.active_grad_compare:
                    old_weights = copy.deepcopy(self.state_dict())
                    # pop grad_bank.weight from old_weights
                    # old_weights.pop('grad_bank.weight')
                if self.active_grad_compare:
                    #init_test_loss = self.no_grad_loss(test_images, test_gt_instances, ann_type, batched_inputs, no_grad = True)
                    init_test_loss_with_grad = self.no_grad_loss(test_images, test_gt_instances, ann_type, batched_inputs, no_grad=False, for_test=True)
                    #init_test_loss_grad = self.get_loss_grad(init_test_loss_with_grad)
                    # init_test_loss_cls0 = self.fetchloss(init_test_loss_with_grad, ['loss_cls_stage0'])
                    if self.active_loss != 'all':
                        init_test_loss_cls0 = self.fetchloss(init_test_loss_with_grad, [self.active_loss])
                    else:
                        init_test_loss_cls0 = init_test_loss_with_grad
                    init_test_loss_cls0_grad = self.get_loss_grad(init_test_loss_cls0,retain = False)
                    #del init_test_loss_with_grad
                else:
                    init_test_loss = self.no_grad_loss(test_images, test_gt_instances, ann_type, batched_inputs, no_grad=True)
                self.print_gpu_memory() # 2.7～2.9G  / 2.3～2.5G
                # set seed
                if self.active_seed != 0:
                    torch.manual_seed(self.active_seed+self.iter)
                # torch.manual_seed(0)
                if not self.forward_once:
                    paste_train_loss = self.no_grad_loss(images, gt_instances, ann_type, batched_inputs, no_grad=False)
                    if self.active_loss_update != 'all':
                        paste_train_loss = self.fetchloss(paste_train_loss, [self.active_loss_update])
                    # if self.active_grad_compare:
                    #     paste_train_loss_grad = self.get_loss_grad(paste_train_loss)
                    # pop 'loss_cls_per_paste_stage0/1/2' from paste_train_loss
                    self.print_gpu_memory() # 3.5G～3.8G  / 2.5G～3.0G
                    if not self.only_paste_sup:
                        paste_train_loss, paste_train_loss_paste = self.pop_loss_paste(paste_train_loss)
                    else:
                        paste_train_loss, paste_train_loss_paste = self.pop_loss_paste(paste_train_loss)
                    if self.active_grad_compare:
                        paste_train_loss_grad = self.get_loss_grad(paste_train_loss,retain = False)
                    self.print_gpu_memory() # 3.5G～3.8G  / 2.8 ~ 3.2G
                    if self.active_only_gt_train:
                        try:
                            paste_train_loss, paste_train_loss_per_ins = self.pop_loss_cls_per_paste(paste_train_loss)
                            print('paste_num', paste_num)
                            print(paste_train_loss_per_ins['loss_cls_per_paste_stage0'].shape)
                            print(paste_train_loss_per_ins['loss_cls_per_paste_stage1'].shape)
                            print(paste_train_loss_per_ins['loss_cls_per_paste_stage2'].shape)
                            paste_ins_loss = paste_train_loss_per_ins['loss_cls_per_paste_stage0'].sum(dim=1)
                            max_error_class  = paste_train_loss_per_ins['loss_cls_per_paste_stage0'].argmax(dim=1)
                            max_error_loss = paste_train_loss_per_ins['loss_cls_per_paste_stage0'].max(dim=1)[0]
                        except:
                            print('fail to pop loss_cls_per_paste')
                            paste_ins_loss = None
                    if not self.active_grad_compare:
                        self.update_with_loss(paste_train_loss, use_optimizer=True)
                        if self.active_seed != 0:
                            torch.manual_seed(self.active_seed+self.iter)
                        paste_test_loss = self.no_grad_loss(test_images, test_gt_instances, ann_type, batched_inputs, no_grad=True)
                        # print('load,rank:', comm.get_rank(), 'old_weights', old_weights['backbone.fpn_lateral3.weight'][0][0][0][0])
                        self.load_state_dict(old_weights)
                        # zero grad
                        self.zero_grad()
                else:
                    # only forward once
                    pass

                    
            else:
                init_test_loss = None
                paste_train_loss = None
                paste_test_loss = None
            
            paste = False
            self.count += 1
            self.print_gpu_memory()   # 3.0G～3.2G / 2.8G～3.2G
            if self.active_mode == 'paste_or_zero':
                if self.compare_loss(init_test_loss, paste_test_loss) == '<':
                    # not paste is better
                    # use ori_images to train
                    images = ori_images
                    gt_instances = ori_gt_instances
                    # fake_loss = paste_train_loss
                    # for k in fake_loss:
                    #     fake_loss[k] = 0*sum(p.mean() for p in self.parameters())
                    # print('not paste is better')
                    # return fake_loss 
                    self.not_paste_count += 1
                else:
                    # paste is better
                    # print('paste is better')
                    paste = True
                    self.paste_count += 1
                    pass
            elif self.active_mode == 'paste_or_ori' or self.active_mode == 'paste_only':
                # use ori image to train
                #print(self.active_mode)
                if  self.active_compare != 'all' and self.active_mode == 'paste_or_ori':
                    ori_train_loss = self.no_grad_loss(ori_images, ori_gt_instances, ann_type, batched_inputs, no_grad=False)
                    if self.active_loss_update != 'all':
                        ori_train_loss = self.fetchloss(ori_train_loss, [self.active_loss_update])
                    if self.active_grad_compare:
                        ori_train_loss_grad = self.get_loss_grad(ori_train_loss,retain = False)
                        if self.active_grad_save:
                            # all gather
                            # init_test_loss_cls0_grad_all = all_gather(init_test_loss_cls0_grad)
                            # # mean
                            # init_test_loss_cls0_grad_all = torch.stack([x.detach().cpu() for x in init_test_loss_cls0_grad_all]).mean(dim=0).to(self.device)
                            init_test_loss_cls0_grad_all = init_test_loss_cls0_grad
                            #print('active_grad_update',self.active_grad_update)
                            init_test_loss_cls0_grad  =  self.update_grad_bank(init_test_loss_cls0_grad_all)
                            del init_test_loss_cls0_grad_all
                        # computer grad sim
                        sim_paste_init = self.compute_grad_sim(paste_train_loss_grad, init_test_loss_cls0_grad)
                        sim_ori_init = self.compute_grad_sim(ori_train_loss_grad, init_test_loss_cls0_grad)
                        self.print_gpu_memory()   # 3.8G～4.0G / 3.2G ~3.7G
                        del paste_train_loss_grad, ori_train_loss_grad, init_test_loss_cls0_grad
                        del ori_train_loss, paste_train_loss
                        self.print_gpu_memory()  #  3.8G～4.0G / 2.0G

                    else:
                        try:
                            ori_train_loss, ori_train_loss_per_ins = self.pop_loss_cls_per_paste(ori_train_loss)
                        except:
                            print('fail to pop loss_cls_per_paste in ori')
                        self.update_with_loss(ori_train_loss, use_optimizer=True)
                        if self.active_seed != 0:
                            torch.manual_seed(self.active_seed+self.iter)
                        ori_test_loss = self.no_grad_loss(test_images, test_gt_instances, ann_type, batched_inputs, no_grad=True)
                        self.load_state_dict(old_weights)
                        # whether need to zero grad? and cuda synchronize?
                        self.zero_grad()
                    self.print_gpu_memory()  # 3.5G ～ 3.6G / 2.0G

                elif self.active_mode == 'paste_only':
                    if self.active_grad_save:
                        
                            init_test_loss_cls0_grad_all = init_test_loss_cls0_grad
                            #print('active_grad_update',self.active_grad_update)
                            init_test_loss_cls0_grad  =  self.update_grad_bank(init_test_loss_cls0_grad_all)
                            del init_test_loss_cls0_grad_all
                    if self.active_seed != 0:
                        torch.manual_seed(self.active_seed+self.iter)
                    paste_train_loss_ = self.no_grad_loss(images, gt_instances, ann_type, batched_inputs, no_grad=False)
                    
                    def pop_certain_loss(paste_train_loss):
                            paste_train_loss_paste = {}
                            # drop _cls_  _paste_
                            for k in paste_train_loss:
                                if '_cls_' in k or '_paste_' in k:
                                    paste_train_loss_paste[k] = paste_train_loss[k]
                            for k in paste_train_loss_paste:
                                paste_train_loss.pop(k)
                            return paste_train_loss, paste_train_loss_paste
                    if not self.active_grad_compare:
                        ori_train_loss, _ = pop_certain_loss(paste_train_loss_)
                        self.update_with_loss(ori_train_loss, use_optimizer=True)
                        ori_test_loss = self.no_grad_loss(test_images, test_gt_instances, ann_type, batched_inputs, no_grad=True)
                        self.load_state_dict(old_weights)
                        # whether need to zero grad? and cuda synchronize?
                        self.zero_grad() 
                    else:
                        if not self.forward_once:
                            ori_train_loss, _ = pop_certain_loss(paste_train_loss_)
                            ori_train_loss_grad = self.get_loss_grad(ori_train_loss,retain = False)
                            
                            sim_ori_init = self.compute_grad_sim(init_test_loss_cls0_grad, ori_train_loss_grad)
                            sim_paste_init = self.compute_grad_sim(init_test_loss_cls0_grad, paste_train_loss_grad)
                            del paste_train_loss_grad, ori_train_loss_grad, init_test_loss_cls0_grad
                            # sim_ori_init_nonorm = self.compute_grad_sim(init_test_loss_cls0_grad, ori_train_loss_grad, norm=False)
                        else:
                            # only forward here 
                            def split_loss(paste_train_loss):
                                paste_train_loss_ori = {}
                                no_paste_loss = {}
                                for k in paste_train_loss:
                                    if '_paste_' in k:
                                        continue
                                    elif '_cls_' in k:
                                        paste_train_loss_ori[k] = paste_train_loss[k]
                                    elif '_nopaste_' in k:
                                        no_paste_loss[k] = paste_train_loss[k]
                                    else:
                                        paste_train_loss_ori[k] = paste_train_loss[k]
                                        no_paste_loss[k] = paste_train_loss[k]
                                return paste_train_loss_ori, no_paste_loss
                            if self.active_once_mode == 'only_gt':
                                paste_train_loss, ori_train_loss = split_loss(paste_train_loss_)
                                ori_train_loss_grad = self.get_loss_grad(ori_train_loss)
                                paste_train_loss_grad = self.get_loss_grad(paste_train_loss,retain = False)
                                sim_ori_init = self.compute_grad_sim(init_test_loss_cls0_grad, ori_train_loss_grad)
                                sim_paste_init = self.compute_grad_sim(init_test_loss_cls0_grad, paste_train_loss_grad)
                                del paste_train_loss_grad, ori_train_loss_grad, init_test_loss_cls0_grad
                            elif  'only_paste' in self.active_once_mode:
                                only_paste_loss = self.fetchloss(paste_train_loss_, ['_paste_'])
                                only_paste_loss_grad = self.get_loss_grad(only_paste_loss,retain = False)
                                sim_paste_init = self.compute_grad_sim(init_test_loss_cls0_grad,only_paste_loss_grad)
                                del init_test_loss_cls0_grad, only_paste_loss_grad
                                # paste_only_0.0
                                if 'dynamic'  not in self.active_once_mode:
                                    sim_ori_init = float(self.active_once_mode.split('_')[-1])
                                elif 'dynamic' in self.active_once_mode:
                                    # get thres from queue
                                    if 'linear' not in self.active_once_mode:
                                        sim_ori_init = self.dynamic_queue.get_threshold()
                                        # update queue
                                        self.dynamic_queue.add_score(sim_paste_init.item())
                                    else:
                                        percentage = self.start_rate + (self.end_rate - self.start_rate) * self.iter / self.max_iter
                                        self.dynamic_queue.set_percentile(1-percentage)
                                        sim_ori_init = self.dynamic_queue.get_threshold()
                                        # update queue
                                        self.dynamic_queue.add_score(sim_paste_init.item())
                                    
                                

                else:
                    # ori_train_loss = None
                    # ori_test_loss = None
                    if self.active_compare == 'all':
                        ori_train_loss = self.no_grad_loss(ori_images, ori_gt_instances, ann_type, batched_inputs, no_grad=False)
                    else:
                        ori_train_loss = self.no_grad_loss(ori_images, ori_gt_instances, ann_type, batched_inputs, no_grad=False)
                        self.update_with_loss(ori_train_loss, use_optimizer=True)
                        ori_test_loss = self.no_grad_loss(test_images, test_gt_instances, ann_type, batched_inputs, no_grad=True)
                        self.load_state_dict(old_weights)
                
                
                if not self.active_pred:
                    if not self.active_grad_compare:
                        if self.compare_loss(ori_test_loss, paste_test_loss) == '<':
                            # ori is better
                            # use ori_images to train
                            images = ori_images
                            gt_instances = ori_gt_instances
                            self.not_paste_count += 1
                        else:
                            # paste is better
                            self.paste_count += 1
                            paste = True
                            pass
                    else:
                        if False:
                            pass
                        else:
                            if sim_ori_init > sim_paste_init:
                                # ori is better
                                # use ori_images to train
                                images = ori_images
                                gt_instances = ori_gt_instances
                                self.not_paste_count += 1
                            else:
                                # paste is better
                                self.paste_count += 1
                                paste = True
                                pass
                else:
                    if self.compare_loss(ori_test_loss, paste_test_loss) == '<':
                        # ori is better
                        # use ori_images to train
                        # 稍后再进行替换
                        self.not_paste_count += 1
                    else:
                        # paste is better
                        self.paste_count += 1
                        paste = True

            else:
                raise NotImplementedError


           
            # 记录贴图来源
            if self.active_compare != 'all':
                if not self.active_grad_compare:
                    loss_dif = self.compute_diff_loss(ori_test_loss, paste_test_loss)
                    loss_dif = loss_dif.item()
                    # 保留四位小数
                    loss_dif = round(loss_dif, 4)
                else:
                    loss_dif =   sim_paste_init - sim_ori_init
                    loss_dif = loss_dif.item()
                 
                output_txt_path = self.output_dir + '/paste_source/' + 'rank_' + self.rank + '/'+ str(self.iter//10000+1) + '0000.txt'
                if not os.path.exists(os.path.dirname(output_txt_path)):
                    os.makedirs(os.path.dirname(output_txt_path))
                if self.active_mode != 'paste_only':
                    with open(output_txt_path, 'a') as f:
                        
                        for i,filename_list in enumerate(paste_filename_lists):    
                                if not self.active_grad_compare:
                                    f.write(filename + ' select_class: ' + str(test_gt_select_classes[i]) + ' paste: ' + str(int(paste)) + ' iter: ' + str(self.iter) + ' loss_dif: '
                                            + str(loss_dif) + ' paste_num: '+ str(paste_num)+ ' paste_test_loss: '+ str(round(float(paste_test_loss['loss_cls_stage0']),4))+
                                            ' ori_test_loss: '+ str(round(float(ori_test_loss['loss_cls_stage0']),4)) + 'paste_train_loss: '+ str(round(float(paste_train_loss['loss_cls_stage0']),4))+
                                                ' ori_train_loss: '+ str(round(float(ori_train_loss['loss_cls_stage0']),4))+'\n')
                                else: 
                                    f.write(filename + ' select_class: ' + str(test_gt_select_classes[i]) + ' paste: ' + str(int(paste)) + ' iter: ' + str(self.iter) + ' loss_dif: '
                                            + str(loss_dif) + ' paste_num: '+ str(paste_num)+
                                                "sim_paste_init: "+ str(round(float(sim_paste_init),4))+
                                                "sim_ori_init: "+ str(round(float(sim_ori_init),4))+'\n')
                        try:
                            del init_test_loss_with_grad
                        except:
                            print('fail to del init_test_loss_with_grad')
                            pass
                else:
                    try:
                        with open(output_txt_path, 'a') as f:
                            for i,filename_list in enumerate(paste_filename_lists):
                                for filename in filename_list:
                                    if not self.active_grad_compare:
                                    #if True:
                                        f.write(filename + ' select_class: ' + str(test_gt_select_classes[i]) + ' paste: ' + str(int(paste)) + ' iter: ' + str(self.iter) + ' loss_dif: '
                                                + str(loss_dif) + ' paste_num: '+ str(paste_num)+ ' paste_test_loss: '+ str(round(float(paste_test_loss['loss_cls_stage0']),4))+
                                                ' ori_test_loss: '+ str(round(float(ori_test_loss['loss_cls_stage0']),4)) + 'init_test_loss: '+ str(round(float(init_test_loss['loss_cls_stage0']),4))+
                                                    "sim_paste_init: "+ str(round(float(sim_paste_init),4))+
                                                    "sim_ori_init: "+ str(round(float(sim_ori_init),4))+'\n')
                                    else:
                                        if self.forward_once:
                                            f.write(filename + ' select_class: ' + str(test_gt_select_classes[i]) + ' paste: ' + str(int(paste)) + ' iter: ' + str(self.iter) + ' loss_dif: '
                                                    + str(loss_dif) + ' paste_num: '+ str(paste_num)+ 'paste_train_loss: '+ str(round(float(paste_train_loss_['loss_cls_stage0']),4))+
                                                        "sim_init_cls0_and_paste: "+ str(round(float(sim_paste_init),4))+
                                                        "sim_init_cls0_and_ori: "+ str(round(float(sim_ori_init),4))+'\n')

                                        else:
                                            f.write(filename + ' select_class: ' + str(test_gt_select_classes[i]) + ' paste: ' + str(int(paste)) + ' iter: ' + str(self.iter) + ' loss_dif: '
                                                    + str(loss_dif) + ' paste_num: '+ str(paste_num)+ 'paste_train_loss: '+ str(round(float(paste_train_loss['loss_cls_stage0']),4))+
                                                        ' ori_train_loss: '+ str(round(float(ori_train_loss['loss_nopaste_ins_stage0']),4))+
                                                        "sim_init_cls0_and_paste: "+ str(round(float(sim_paste_init),4))+
                                                        "sim_init_cls0_and_ori: "+ str(round(float(sim_ori_init),4))+'\n')
                    except:
                        print('fail to save paste_source')
                        
                            
                # record paste_ins_loss
                if self.active_only_gt_train:
                    output_txt_path = self.output_dir + '/paste_ins_loss/' + 'rank_' + self.rank + '/'+ str(self.iter//10000+1) + '0000.txt'
                    if not os.path.exists(os.path.dirname(output_txt_path)):
                        os.makedirs(os.path.dirname(output_txt_path))
                    try:
                        with open(output_txt_path, 'a') as f:
                            j = 0
                            assert len(paste_ins_loss) == paste_num
                            for i,filename_list in enumerate(paste_filename_lists):
                                for filename in filename_list:
                                    f.write(filename + ' select_class: ' + str(test_gt_select_classes[i]) + ' paste: ' + str(int(paste)) + ' iter: ' + str(self.iter) + ' paste_num: '+ str(paste_num)
                                            + ' paste_ins_loss: '+ str(round(float(paste_ins_loss[j]),4))+ ' max_error_class: '+ str(max_error_class[j].item())+ ' max_error_loss: '+ str(round(float(max_error_loss[j]),4))+'\n')
                                    j += 1
                    except:
                        print('fail to save paste_ins_loss')
                self.iter += 1     
        if self.count % 100 == 0:
            print('paste_count:', self.paste_count, 'not_paste_count:', self.not_paste_count)

        if self.with_image_labels:
            for inst, x in zip(gt_instances, batched_inputs):
                inst._ann_type = x['ann_type']
                inst._pos_category_ids = x['pos_category_ids']
            ann_types = [x['ann_type'] for x in batched_inputs]
            assert len(set(ann_types)) == 1
            ann_type = ann_types[0]
            if ann_type in ['prop', 'proptag']:
                for t in gt_instances:
                    t.gt_classes *= 0
        self.print_gpu_memory()  # 3.5G ～ 3.6G
        if self.fp16: # TODO (zhouxy): improve
            with autocast():
                features = self.backbone(images.tensor.half())
            features = {k: v.float() for k, v in features.items()}
        else:
            features = self.backbone(images.tensor)
        
       
        cls_features, cls_inds, caption_features = None, None, None

        if self.with_caption and 'caption' in ann_type:
            inds = [torch.randint(len(x['captions']), (1,))[0].item() \
                for x in batched_inputs]
            caps = [x['captions'][ind] for ind, x in zip(inds, batched_inputs)]
            caption_features = self.text_encoder(caps).float()
        if self.sync_caption_batch:
            caption_features = self._sync_caption_features(
                caption_features, ann_type, len(batched_inputs))
        
        if self.dynamic_classifier and ann_type != 'caption':
            cls_inds = self._sample_cls_inds(gt_instances, ann_type) # inds, inv_inds
            ind_with_bg = cls_inds[0].tolist() + [-1]
            cls_features = self.roi_heads.box_predictor[
                0].cls_score.zs_weight[:, ind_with_bg].permute(1, 0).contiguous()

        classifier_info = cls_features, cls_inds, caption_features
        proposals, proposal_losses = self.proposal_generator(
            images, features, gt_instances)

        if self.roi_head_name in ['StandardROIHeads', 'CascadeROIHeads']:
            proposals, detector_losses = self.roi_heads(
                images, features, proposals, gt_instances)
        else:
            sem_seg_gt = None
            if "sem_seg" in batched_inputs[0]:
                sem_seg_gt = [x["sem_seg"].to(self.device) for x in batched_inputs]
                sem_seg_gt = ImageList.from_tensors(
                    sem_seg_gt, self.backbone.size_divisibility, -1
                ).tensor
            proposals, detector_losses = self.roi_heads(
                images, features, proposals, gt_instances,
                ann_type=ann_type, classifier_info=classifier_info, sem_seg_gt=sem_seg_gt)
        
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        
        if self.with_image_labels:
            if ann_type in ['box', 'prop', 'proptag']:
                losses.update(proposal_losses)
            else: # ignore proposal loss for non-bbox data
                losses.update({k: v * 0 for k, v in proposal_losses.items()})
        else:
            losses.update(proposal_losses)
        if len(self.dataset_loss_weight) > 0:
            dataset_sources = [x['dataset_source'] for x in batched_inputs]
            assert len(set(dataset_sources)) == 1
            dataset_source = dataset_sources[0]
            for k in losses:
                losses[k] *= self.dataset_loss_weight[dataset_source]
        # pop paste_loss
        losses,_ = self.pop_loss_paste(losses)
        
        if self.active_mode == 'paste_or_zero' and not paste:
            for k in losses:
                losses[k] *= 0.0
        if self.active_compare == 'all':
            for k in losses:
                losses[k] += ori_train_loss[k]
        if self.return_proposal:
            return proposals, losses
        else:
            return losses
            
    def no_grad_loss(self, images,gt_instances, ann_type, batched_inputs,no_grad=True,for_test=False):
        # get loss without grad
        self.backbone.eval()
        if no_grad:
            # also need to set model to eval mode
            #self.backbone.eval()
            with torch.no_grad():
                if self.fp16: # TODO (zhouxy): improve
                    with autocast():
                        features = self.backbone(images.tensor.half())
                    features = {k: v.float() for k, v in features.items()}
                else:
                    features = self.backbone(images.tensor)

                cls_features, cls_inds, caption_features = None, None, None

                if self.with_caption and 'caption' in ann_type:
                    raise 
                    inds = [torch.randint(len(x['captions']), (1,))[0].item() \
                        for x in batched_inputs]
                    caps = [x['captions'][ind] for ind, x in zip(inds, batched_inputs)]
                    caption_features = self.text_encoder(caps).float()
                if self.sync_caption_batch:
                    raise
                    caption_features = self._sync_caption_features(
                        caption_features, ann_type, len(batched_inputs))
                
                if self.dynamic_classifier and ann_type != 'caption':
                    cls_inds = self._sample_cls_inds(gt_instances, ann_type) # inds, inv_inds
                    ind_with_bg = cls_inds[0].tolist() + [-1]
                    cls_features = self.roi_heads.box_predictor[
                        0].cls_score.zs_weight[:, ind_with_bg].permute(1, 0).contiguous()

                classifier_info = cls_features, cls_inds, caption_features
                proposals, proposal_losses = self.proposal_generator(
                    images, features, gt_instances)
                if self.roi_head_name in ['StandardROIHeads', 'CascadeROIHeads']:
                    proposals, detector_losses = self.roi_heads(
                        images, features, proposals, gt_instances)
                else:
                    sem_seg_gt = None
                    if "sem_seg" in batched_inputs[0]:
                        raise NotImplementedError
                        sem_seg_gt = [x["sem_seg"].to(self.device) for x in batched_inputs]
                        sem_seg_gt = ImageList.from_tensors(
                            sem_seg_gt, self.backbone.size_divisibility, -1
                        ).tensor
                    proposals, detector_losses = self.roi_heads(
                        images, features, proposals, gt_instances,
                        ann_type=ann_type, classifier_info=classifier_info, sem_seg_gt=sem_seg_gt,only_gt_proposals=self.active_only_gt_test)
                
                if self.vis_period > 0:
                    raise NotImplementedError
                    storage = get_event_storage()
                    if storage.iter % self.vis_period == 0:
                        self.visualize_training(batched_inputs, proposals)

                losses = {}
                losses.update(detector_losses)
                if self.with_image_labels:
                    if ann_type in ['box', 'prop', 'proptag']:
                        losses.update(proposal_losses)
                    else: # ignore proposal loss for non-bbox data
                        losses.update({k: v * 0 for k, v in proposal_losses.items()})
                else:
                    losses.update(proposal_losses)
                if len(self.dataset_loss_weight) > 0:
                    dataset_sources = [x['dataset_source'] for x in batched_inputs]
                    assert len(set(dataset_sources)) == 1
                    dataset_source = dataset_sources[0]
                    for k in losses:
                        losses[k] *= self.dataset_loss_weight[dataset_source]
                # set model to training
                self.train() # add 11.20
                return losses
        else:
                if self.fp16: 
                    with autocast():
                        features = self.backbone(images.tensor.half())
                    features = {k: v.float() for k, v in features.items()}
                else:
                    features = self.backbone(images.tensor)

                cls_features, cls_inds, caption_features = None, None, None

                if self.with_caption and 'caption' in ann_type:
                    raise 
                    inds = [torch.randint(len(x['captions']), (1,))[0].item() \
                        for x in batched_inputs]
                    caps = [x['captions'][ind] for ind, x in zip(inds, batched_inputs)]
                    caption_features = self.text_encoder(caps).float()
                if self.sync_caption_batch:
                    raise
                    caption_features = self._sync_caption_features(
                        caption_features, ann_type, len(batched_inputs))
                
                if self.dynamic_classifier and ann_type != 'caption':
                    cls_inds = self._sample_cls_inds(gt_instances, ann_type) # inds, inv_inds
                    ind_with_bg = cls_inds[0].tolist() + [-1]
                    cls_features = self.roi_heads.box_predictor[
                        0].cls_score.zs_weight[:, ind_with_bg].permute(1, 0).contiguous()

                classifier_info = cls_features, cls_inds, caption_features
                proposals, proposal_losses = self.proposal_generator(
                    images, features, gt_instances)
                if self.roi_head_name in ['StandardROIHeads', 'CascadeROIHeads']:
                    proposals, detector_losses = self.roi_heads(
                        images, features, proposals, gt_instances)
                else:
                    sem_seg_gt = None
                    if "sem_seg" in batched_inputs[0]:
                        raise NotImplementedError
                        sem_seg_gt = [x["sem_seg"].to(self.device) for x in batched_inputs]
                        sem_seg_gt = ImageList.from_tensors(
                            sem_seg_gt, self.backbone.size_divisibility, -1
                        ).tensor
                    gt_ins_num = [len(x) for x in gt_instances]
                    if 0 not in gt_ins_num:
                        if not for_test:
    
                            proposals, detector_losses = self.roi_heads(
                                images, features, proposals, gt_instances,
                                ann_type=ann_type, classifier_info=classifier_info, sem_seg_gt=sem_seg_gt,only_gt_proposals=self.active_only_gt_train)
                        else:
                            proposals, detector_losses = self.roi_heads(
                                images, features, proposals, gt_instances,
                                ann_type=ann_type, classifier_info=classifier_info, sem_seg_gt=sem_seg_gt,only_gt_proposals=self.active_only_gt_test)
                    else:
                        print('fail to use only_gt_proposals in train')
                        # storage = get_event_storage()
                        # storage._current_prefix = ''
                        proposals, detector_losses = self.roi_heads(
                            images, features, proposals, gt_instances,
                            ann_type=ann_type, classifier_info=classifier_info, sem_seg_gt=sem_seg_gt,only_gt_proposals=False)

                
                if self.vis_period > 0:
                    raise NotImplementedError
                    storage = get_event_storage()
                    if storage.iter % self.vis_period == 0:
                        self.visualize_training(batched_inputs, proposals)

                losses = {}
                losses.update(detector_losses)
                
                if self.with_image_labels:
                    if ann_type in ['box', 'prop', 'proptag']:
                        losses.update(proposal_losses)
                    else: # ignore proposal loss for non-bbox data
                        losses.update({k: v * 0 for k, v in proposal_losses.items()})
                else:
                    losses.update(proposal_losses)
                if len(self.dataset_loss_weight) > 0:
                    dataset_sources = [x['dataset_source'] for x in batched_inputs]
                    assert len(set(dataset_sources)) == 1
                    dataset_source = dataset_sources[0]
                    for k in losses:
                        losses[k] *= self.dataset_loss_weight[dataset_source]
                self.train()
                return losses

    def update_with_loss(self, losses, use_optimizer=True):
        # update model with loss
        if self.active_loss_update == 'all':
            losses_sum = sum(losses.values())
        elif self.active_loss_update == 'cls':
            losses_sum = sum([losses[k] for k in losses if 'cls' in k])
        else:
            raise NotImplementedError
        use_optimizer = self.active_optimizer 
        # print('lr', self.lr)
        # if self.lr == 0:
        #     print('lr is 0')
        #     losses_sum*=0
        #cls_loss = sum([losses[k] for k in losses if 'cls' in k])
        if use_optimizer:
            #old_optimizer_state_dict = copy.deepcopy(self.optimizer.state_dict())
            self.optimizer.zero_grad()
            losses_sum.backward() # 如果要backward两次，需要retain_graph=True 即 losses_sum.backward(retain_graph=True)
            #losses_sum.backward(retain_graph=True)
            self.optimizer.step()
            #self.optimizer.load_state_dict(old_optimizer_state_dict)
        else:
            # Update parameters manually
            parameters_to_grad = [(param, name) for name, param in self.named_parameters() if param.requires_grad]
            grad = torch.autograd.grad(losses_sum, [param for param, _ in parameters_to_grad], allow_unused=True,retain_graph=True) # TODO check create_graph
            for (p, name), g in zip(parameters_to_grad, grad):
                if g is not None:
                    p.data -= self.lr * g
                else:
                    # print(f"Encountered None gradient for parameter: {name}")
                    pass
    
    def get_loss_grad(self, losses,retain=True):
        # get loss grad
        losses_sum = sum(losses.values())
        self.zero_grad()
        #with torch.no_grad():
        losses_sum.backward(retain_graph=retain)
        def save_grad():
            # save grad to a tensor
            # with torch.no_grad():
            grad = []
            str_list = "backbone.bottom_up.base.fc"
            for name, param in self.named_parameters():
                if str_list in name:
                    continue
                if param.requires_grad:
                    if param.grad is not None:
                        grad.append(param.grad.clone())
                    else:
                        #print(name+'is none')
                        # padding 0
                        grad.append(torch.zeros_like(param))    
                else:
                    #print(name + 'not require grad')
                    pass
            # flatten all grad
            grad = torch.cat([x.flatten() for x in grad], dim=0)
            return grad
        grad = save_grad()
        #self.optimizer.zero_grad()
        return grad
    def locate_grad(self,index):
        # given index, locate grad, return its name 
        str_list = "backbone.bottom_up.base.fc"
        begin = 0
        for name, param in self.named_parameters():
            if str_list in name:
                continue
            if param.requires_grad:
                end = begin + param.numel()
                if index >= begin and index < end:
                    return name, begin, end
                begin = end
    def mean_grad(self,grad):
        # given grad, return each layer's mean
        str_list = "backbone.bottom_up.base.fc"
        result  = {}
        begin = 0
        for name, param in self.named_parameters():
            if str_list in name:
                continue
            if param.requires_grad:
                end = begin + param.numel()
                result[name] = grad[begin:end].abs().mean()
                begin = end
                result[name] = result[name].item()
        return result
        
    
    def init_grad_bank(self):
        assert self.active_grad_save, 'active_grad_save should be True'
        # init grad bank
        grad_size = 0
        str_list = "backbone.bottom_up.base.fc"
        for name, param in self.named_parameters():
            if str_list in name:
                continue
            if param.requires_grad:
                grad_size += param.numel()
        self.grad_bank = torch.nn.Embedding(grad_size, 1)
        self.grad_bank.weight.requires_grad = False
        self.grad_bank.weight.data.fill_(0)
        self.grad_bank = self.grad_bank.to(self.device)
    
    def update_grad_bank(self, grad):
        # update grad bank
        assert self.active_grad_save, 'active_grad_save should be True'
        #print(self.active_grad_update)
        with torch.no_grad():
            if self.active_grad_update == "AVERAGE":
                self.grad_bank.weight.data.mul_(self.iter / (self.iter+1))
                #self.grad_bank.weight.data += grad / (self.iter+1)
                #output with shape [77722099, 1] doesn't match the broadcast shape [77722099, 77722099]
                self.grad_bank.weight.data += grad.unsqueeze(-1) / (self.iter+1)
            # elif self.active_grad_update == "MOMENTUM0.9":
            #     self.grad_bank.weight.data.mul_(0.9)
            #     self.grad_bank.weight.data += grad.unsqueeze(-1) * 0.1
            elif 'MOMENTUM' in self.active_grad_update:
                momentum = float(self.active_grad_update.split('TUM')[1])
                self.grad_bank.weight.data.mul_(momentum)
                self.grad_bank.weight.data += grad.unsqueeze(-1) * (1-momentum)
            else:
                raise NotImplementedError
        if self.iter % 10000 == 0:
            # save grad bank
            output_dir = self.output_dir + '/grad_bank/' + 'rank_' + self.rank + '/'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_path = output_dir + str(self.iter//10000+1) + '0000.pth'
            torch.save(self.grad_bank.state_dict(), output_path)
        return self.grad_bank.weight.data.squeeze(-1)
    
    def compute_grad_sim(self, grad1, grad2,norm=None):
        # grad1: (N)
        # grad2: (N)
        # return (grad1 * grad2).sum()
        # Normalize
        if norm is not None:
            pass
        else:
            norm = self.active_grad_norm
        if norm:
            return (grad1 * grad2).sum() / (grad1.norm() * grad2.norm() + 1e-8)
        else:
            return (grad1 * grad2).sum()
    
    def fetchloss(self, losses, str_list):
        # fetch loss with str_list
        res = {}
        for k in losses:
            for s in str_list:
                if s in k:
                    res[k] = losses[k]
        return res

    def compare_loss(self, old_loss, new_loss,mode='cls'):
        # sum loss 
        if self.active_compare == 'all':
            return '>'
        if 'random' in self.active_compare:
            if self.active_compare == 'random':
                if random.random() > 0.5: # prob to paste
                    return '<'
                else:
                    return '>'
            else:
                thres = float(self.active_compare.split('_')[1]) # 'random_0.8'
                if random.random() > thres:
                    return '<'
                else:
                    return '>'
        mode = self.active_loss
        if mode == 'all':
            old_loss_sum = sum(old_loss.values())
            new_loss_sum = sum(new_loss.values())
        elif mode == 'cls':
            old_loss_sum = sum([old_loss[k] for k in old_loss if 'cls' in k])
            new_loss_sum = sum([new_loss[k] for k in new_loss if 'cls' in k])
        elif mode == 'box':
            old_loss_sum = sum([old_loss[k] for k in old_loss if 'box' in k])
            new_loss_sum = sum([new_loss[k] for k in new_loss if 'box' in k])
        elif mode == 'mask':
            old_loss_sum = sum([old_loss[k] for k in old_loss if 'mask' in k])
            new_loss_sum = sum([new_loss[k] for k in new_loss if 'mask' in k])
        elif mode == 'cls_stage0': 
            old_loss_sum = sum([old_loss[k] for k in old_loss if 'cls_stage0' in k])
            new_loss_sum = sum([new_loss[k] for k in new_loss if 'cls_stage0' in k])
        elif mode == 'stage0':
            old_loss_sum = sum([old_loss[k] for k in old_loss if 'stage0' in k])
            new_loss_sum = sum([new_loss[k] for k in new_loss if 'stage0' in k])
        else:
            raise NotImplementedError
        # for k in old_loss:
        #     print (k, old_loss[k], new_loss[k])
        # print ("loss sum", old_loss_sum, new_loss_sum)
        if self.active_compare == 'contra':
            if new_loss_sum < old_loss_sum:
                return '<'
            else:
                return '>'
        elif self.active_compare == 'prob':
            # with prob 0.8
            if random.random() < 0.8:
                if new_loss_sum < old_loss_sum:
                        return '>'
                else:
                        return '<'
            else:
                if new_loss_sum < old_loss_sum:
                    return '<'
                else:
                    return '>'
        elif self.active_compare == 'default':
            if new_loss_sum < old_loss_sum:
                return '>'
            else:
                return '<'
        elif self.active_compare == 'schedule':
            thres = self.iter / 90000 
            if random.random() > thres:
                if new_loss_sum < old_loss_sum:
                        return '>'
                else:
                        return '<'
            else: 
                return '>'
        else:
            raise NotImplementedError
    def compute_diff_loss(self, old_loss, new_loss,mode=None):
        # sum loss 
        if mode == None:
            mode = self.active_loss
        if mode == 'all':
            old_loss_sum = sum(old_loss.values())
            new_loss_sum = sum(new_loss.values())
        elif mode == 'cls':
            old_loss_sum = sum([old_loss[k] for k in old_loss if 'cls' in k])
            new_loss_sum = sum([new_loss[k] for k in new_loss if 'cls' in k])
        elif mode == 'box':
            old_loss_sum = sum([old_loss[k] for k in old_loss if 'box' in k])
            new_loss_sum = sum([new_loss[k] for k in new_loss if 'box' in k])
        elif mode == 'cls_stage0':
            old_loss_sum = sum([old_loss[k] for k in old_loss if 'cls_stage0' in k])
            new_loss_sum = sum([new_loss[k] for k in new_loss if 'cls_stage0' in k])
        elif mode == 'stage0':
            old_loss_sum = sum([old_loss[k] for k in old_loss if 'stage0' in k])
            new_loss_sum = sum([new_loss[k] for k in new_loss if 'stage0' in k])
        else:
            raise NotImplementedError
        if 'debug' in self.output_dir:
            print('old_loss_sum', old_loss_sum, 'new_loss_sum', new_loss_sum)
        return old_loss_sum-new_loss_sum
    def pop_loss_cls_per_paste(self,paste_train_loss):
                        paste_train_loss_per_ins = {}
                        for k in paste_train_loss:
                            if 'loss_cls_per_paste' in k:
                                paste_train_loss_per_ins[k] = paste_train_loss[k]
                        for k in paste_train_loss_per_ins:
                            paste_train_loss.pop(k)
                        return paste_train_loss, paste_train_loss_per_ins
    def pop_loss_paste(self,paste_train_loss):
                        paste_train_loss_paste = {}
                        for k in paste_train_loss:
                            if 'paste' in k:
                                paste_train_loss_paste[k] = paste_train_loss[k]
                        for k in paste_train_loss_paste:
                            paste_train_loss.pop(k)
                        return paste_train_loss, paste_train_loss_paste
    def print_gpu_memory(self):
        if False:
            byte_to_gb_factor = 1.0 / 1024**3   # convert bytes to GB.
            torch.cuda.synchronize()
            # get rank of current process
            rank = comm.get_rank()
            if rank != 0:
                return
            # The current frame is print_gpu_memory function's frame.
            current_frame = inspect.currentframe()

            # The frame of code who calls print_gpu_memory function.
            caller_frame = inspect.getouterframes(current_frame, 2)
            
            print('Called from line:', caller_frame[1][2])
            print('总显存 (GB):', torch.cuda.get_device_properties(0).total_memory * byte_to_gb_factor)
            print('当前已经分配的显存 (GB):', torch.cuda.memory_allocated() * byte_to_gb_factor)
            print('最高消耗的显存 (GB):', torch.cuda.max_memory_allocated() * byte_to_gb_factor)
            print('被缓存 (保留)的显存 (GB):', torch.cuda.memory_reserved() * byte_to_gb_factor)
            print('历史最大缓存 (保留)的显存 (GB):', torch.cuda.max_memory_reserved() * byte_to_gb_factor)
       
            
    def _sync_caption_features(self, caption_features, ann_type, BS):
        has_caption_feature = (caption_features is not None)
        BS = (BS * self.cap_batch_ratio) if (ann_type == 'box') else BS
        rank = torch.full(
            (BS, 1), comm.get_rank(), dtype=torch.float32, 
            device=self.device)
        if not has_caption_feature:
            caption_features = rank.new_zeros((BS, 512))
        caption_features = torch.cat([caption_features, rank], dim=1)
        global_caption_features = comm.all_gather(caption_features)
        caption_features = torch.cat(
            [x.to(self.device) for x in global_caption_features], dim=0) \
                if has_caption_feature else None # (NB) x (D + 1)
        return caption_features


    def _sample_cls_inds(self, gt_instances, ann_type='box'):
        if ann_type == 'box':
            gt_classes = torch.cat(
                [x.gt_classes for x in gt_instances])
            C = len(self.freq_weight)
            freq_weight = self.freq_weight
        else:
            gt_classes = torch.cat(
                [torch.tensor(
                    x._pos_category_ids, 
                    dtype=torch.long, device=x.gt_classes.device) \
                    for x in gt_instances])
            C = self.num_classes
            freq_weight = None
        assert gt_classes.max() < C, '{} {}'.format(gt_classes.max(), C)
        inds = get_fed_loss_inds(
            gt_classes, self.num_sample_cats, C, 
            weight=freq_weight)
        cls_id_map = gt_classes.new_full(
            (self.num_classes + 1,), len(inds))
        cls_id_map[inds] = torch.arange(len(inds), device=cls_id_map.device)
        return inds, cls_id_map

    def _postprocess(self, instances, batched_inputs: List[Dict[str, torch.Tensor]], image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r, inds = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
            # if hasattr(self, 'roi_heads.')
        return processed_results, inds

def detector_postprocess(
    results: Instances, output_height: int, output_width: int, mask_threshold: float = 0.5
):
    """
    Resize the output instances.
    The input images are often resized when entering an object detector.
    As a result, we often need the outputs of the detector in a different
    resolution from its inputs.

    This function will resize the raw outputs of an R-CNN detector
    to produce outputs according to the desired output resolution.

    Args:
        results (Instances): the raw outputs from the detector.
            `results.image_size` contains the input image resolution the detector sees.
            This object might be modified in-place.
        output_height, output_width: the desired output resolution.

    Returns:
        Instances: the resized output from the model, based on the output resolution
    """
    # Change to 'if is_tracing' after PT1.7
    if isinstance(output_height, torch.Tensor):
        # Converts integer tensors to float temporaries to ensure true
        # division is performed when computing scale_x and scale_y.
        output_width_tmp = output_width.float()
        output_height_tmp = output_height.float()
        new_size = torch.stack([output_height, output_width])
    else:
        new_size = (output_height, output_width)
        output_width_tmp = output_width
        output_height_tmp = output_height

    scale_x, scale_y = (
        output_width_tmp / results.image_size[1],
        output_height_tmp / results.image_size[0],
    )
    results = Instances(new_size, **results.get_fields())

    if results.has("pred_boxes"):
        output_boxes = results.pred_boxes
    elif results.has("proposal_boxes"):
        output_boxes = results.proposal_boxes
    else:
        output_boxes = None
    assert output_boxes is not None, "Predictions must contain boxes!"

    output_boxes.scale(scale_x, scale_y)
    output_boxes.clip(results.image_size)

    inds = output_boxes.nonempty()
    results = results[inds]

    if results.has("pred_masks"):
        if isinstance(results.pred_masks, ROIMasks):
            roi_masks = results.pred_masks
        else:
            # pred_masks is a tensor of shape (N, 1, M, M)
            roi_masks = ROIMasks(results.pred_masks[:, 0, :, :])
        results.pred_masks = roi_masks.to_bitmasks(
            results.pred_boxes, output_height, output_width, mask_threshold
        ).tensor  # TODO return ROIMasks/BitMask object in the future

    if results.has("pred_keypoints"):
        results.pred_keypoints[:, :, 0] *= scale_x
        results.pred_keypoints[:, :, 1] *= scale_y

    return results, inds
