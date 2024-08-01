import logging
import os
import cv2
import re
import numpy as np
import torch
from pytorch_msssim import ssim
from detectron2.projects.segmentation.evaluation.evaluator_base import BaseEvaluator
from detectron2.projects.segmentation.data import ImageSample
from typing import List, Any
from detectron2.utils.events import get_event_storage


class GnrtMTTEvaluator(BaseEvaluator):
    def __init__(self,
                 output_dir,
                 checkpoint = '',
                 save_features = False,
                 save_cams = False):

        # super().__init__(prefix)
        self._logger = logging.getLogger(__name__)
        self.output_dir = output_dir
        self.prefix = 'GnrtMTTEvaluator'
        self.keys = ['img_name', 'MSE', 'vMSE', 'SSIM', 'PSNR']
        self.write_eval_results()
        self.checkpoint = checkpoint
        self.save_features = save_features
        self.results = []
        self.save_cams = save_cams


    def process(self, data_samples: List[ImageSample]):
        try:
            storage = get_event_storage()
            iteration = storage.iter
        except:
            iteration = int(re.findall(r'\d+', self.checkpoint.split('\\')[-1])[0])

            # iteration = 10086

        for sample in data_samples:
            img_name = sample.img_name
            label = sample.label.cuda()
            # print(label.shape)
            vmask = label[1:2]
            label = label[0:1]
            pred = sample.pred.cuda()
            MSE = torch.mean((label-pred)**2)
            vMSE = torch.sum(vmask*(label-pred)**2) / torch.sum(vmask)
            PSNR = 10*torch.log10(1/MSE)
            SSIM = ssim(pred[None], label[None], data_range=1, size_average=True)

            res = {
                'img_name': img_name,
                'MSE': MSE,
                'vMSE': vMSE,
                'SSIM': SSIM,
                'PSNR': PSNR
            }

            self.save_data(sample, self.output_dir, img_name, iteration)
            self.write_eval_results(res, iteration)
            self.results.append(res)

    def compute_metrics(self, results:list) -> Any:
        metrics = {}
        for k in self.keys:
            if k == 'img_name':
                continue
            metrics[k] = sum([x[k] for x in results]) / max(1, len(results))
        return metrics

    def save_data(self, sample, output_dir, output_name, iteration):
        pred_dir = os.path.join(output_dir,'evaluation')
        os.makedirs(pred_dir,exist_ok=True)
        # np.savez_compressed(os.path.join(pred_dir,str(iteration)+'_'+output_name+'.npz'),
        #                     sample.pred.detach().cpu().numpy())
        cv2.imwrite(os.path.join(pred_dir,str(iteration)+'_'+output_name+'.jpg'),
                    np.transpose(sample.pred.detach().cpu().numpy()*255, axes=(1,2,0)))

        if self.save_features:
            feature_dir = os.path.join(output_dir, 'features')
            os.makedirs(feature_dir, exist_ok=True)
            np.savez(os.path.join(feature_dir,str(iteration)+'_'+output_name+'.npz'),
                     *([x.detach().cpu().numpy() for x in sample.logits+sample.y+sample.x]))

        if self.save_cams:
            cams_dir = os.path.join(output_dir,'cams')
            os.makedirs(cams_dir, exist_ok=True)
            np.savez(os.path.join(cams_dir, str(iteration) + '_' + output_name + '.npz'),
                     *([x.detach().cpu().numpy() for x in sample.cam]))



    def write_eval_results(self, data=None, iteration=None):
        if data is None:
            self.output_file = os.path.join(self.output_dir, 'eval.txt')
            if not os.path.exists(self.output_file):
                with open(self.output_file,'w') as f:
                    f.write('iteration')
                    for key in self.keys:
                        f.write('\t'+key)
                    f.write('\n')
        else:
            with open(self.output_file, 'a') as f:
                f.write(str(iteration))
                for key in self.keys:
                    if key == 'img_name':
                        f.write('\t'+data[key])
                    else:
                        f.write('\t'+ str(data[key].detach().cpu().numpy()))
                f.write('\n')




