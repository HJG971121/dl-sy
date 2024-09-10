import logging
import os
import numpy as np
import torch
from detectron2.projects.segmentation.evaluation.evaluator_base import BaseEvaluator
from detectron2.projects.segmentation.data import ImageSample
from typing import List, Any
from detectron2.utils.events import get_event_storage
import cv2


class VesSegEvaluator(BaseEvaluator):
    def __init__(self,
                 output_dir):

        # super().__init__(prefix)
        self._logger = logging.getLogger(__name__)
        self.output_dir = output_dir
        self.prefix = 'GnrtMTTEvaluator'
        self.keys = ['img_name', 'Dice']
        self.write_eval_results()


    def process(self, data_samples: List[ImageSample]):
        try:
            storage = get_event_storage()
            iteration = storage.iter
        except:
            iteration = 10086

        for sample in data_samples:
            img_name = sample.img_name
            label = sample.label.cuda()
            pred = sample.pred.cuda()>0.5
            Dice = (torch.sum(2*pred*label))/(torch.sum(pred)+torch.sum(label))

            res = {
                'img_name': img_name,
                'Dice': Dice,
            }

            self.save_data(pred, os.path.join(self.output_dir,'evaluation'), img_name)
            self.write_eval_results(res, iteration)
            self.results.append(res)

    def compute_metrics(self, results:list) -> Any:
        metrics = {}
        for k in self.keys:
            if k == 'img_name':
                continue
            metrics[k] = sum([x[k] for x in results]) / max(1, len(results))
        return metrics

    def save_data(self, data, output_dir, output_name, show_status = False):
        os.makedirs(output_dir,exist_ok=True)
        cv2.imwrite(os.path.join(output_dir,output_name+'.jpg'),np.transpose(data.detach().cpu().numpy()*255, axes=(1,2,0)))
        if show_status:
            print(output_name+'.bmp has been saved!!')

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




