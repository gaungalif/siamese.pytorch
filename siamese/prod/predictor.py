import os
import sys
import time
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np
import torch
import torch.nn.functional as F

from prod.utils import *

import onnx
import onnxruntime as ort

from typing import *




class SiamesePredictorOnnx(object):
    def __init__(self, weight: str = None, num_classes: int = 1, 
                device: str = 'cpu'):

        self.weight = weight
        self.num_classes = num_classes
        self.device = device
        
        self._load_check_model()
        self._init_session()

    @property
    def _providers(self):
        return {"cpu":'CPUExecutionProvider', "cuda": 'CUDAExecutionProvider'}
    
    def _init_session(self):
        if self.weight:
            self.session = ort.InferenceSession(self.weight)
        providers = [self._providers.get(self.device, "cpu")]
        self.session.set_providers(providers)
        
    def _load_check_model(self):
        self.onnx_model = onnx.load(self.weight)
        onnx.checker.check_model(self.onnx_model)
        # print('model loaded')
        
    def _to_numpy(self, tensor: torch.Tensor):
        if tensor.requires_grad:
            return tensor.detach().cpu().numpy() 
        else:
            return tensor.cpu().numpy()        

    def _onnx_predict(self, images: np.ndarray):
        start_time = time.time()

        sess_input = {self.session.get_inputs()[0].name: images}
        ort_outs = self.session.run(None, sess_input)
        total_time =time.time() - start_time
        onnx_predict = ort_outs[0]

        show_time(total_time)
        return onnx_predict
    
    def _preprocess(self, input: torch.Tensor):
        inputs = self._to_numpy(input)
        return inputs

    def _postprocess(self, main_feats: np.ndarray, comp_feats: np.ndarray):
        # threshold: float = 0.5
        # prediction[prediction >= threshold] = 1.
        # prediction[prediction <= threshold] = 0.
        distances = F.pairwise_distance(torch.sigmoid(torch.from_numpy(main_feats)), torch.sigmoid(torch.from_numpy(comp_feats)))
        return distances

    def _predict(self, main_imgs: torch.Tensor, comp_imgs: torch.Tensor, label: torch.Tensor, pic_idx: int):
        main_imgs, comp_imgs = self._preprocess(main_imgs), self._preprocess(comp_imgs)
        main_feats, comp_feats = self._onnx_predict(main_imgs), self._onnx_predict(comp_imgs)
        distances = self._postprocess(main_feats, comp_feats)
        result = show_results(distances, label, main_imgs, comp_imgs, pic_idx)
        return result
    
    def predict(self, main_imgs: torch.Tensor, comp_imgs: torch.Tensor, label: torch.Tensor, pic_idx: int):
        result = self._predict(main_imgs, comp_imgs, label, pic_idx)
        return result

if __name__ == "__main__":
    # from data.datamodule import OmniglotDataModule

    weight_onnx = './weights/siamese-10.onnx'

    data_dir = '/home/gaungalif/workspace/datasets/omniglot/Alphabet_of_the_Magi'
    idx = 2
    pic_idx = 7
    main_imgs, comp_imgs, labels = image_loader(idx=idx,data_dir=data_dir)
    net_onnx = SiamesePredictorOnnx(weight=weight_onnx)
    res = net_onnx.predict(main_imgs, comp_imgs, labels, pic_idx)