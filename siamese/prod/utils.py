from matplotlib.image import NonUniformImage
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from data.datamodule import OmniglotDataModule


def image_loader(idx: int = None, data_dir: str = None):
    dataset = OmniglotDataModule(data_dir,batch_size=32,num_workers=2,simmilar_data_multiplier=1)
    valid_loader = dataset.val_dataloader()
    dataiter = iter(valid_loader)
    for _ in range(idx):
        main_imgs, comp_imgs, labels = dataiter.next()
    return main_imgs, comp_imgs, labels

class show_results():
    def __init__(self, distance: np.ndarray, label:torch.Tensor, main_imgs: torch.Tensor, comp_imgs: torch.Tensor, pic_idx: int):
        self.distance = distance
        self.label = label
        self.main_imgs = main_imgs
        self.comp_imgs = comp_imgs
        self. pic_idx = pic_idx
        self.threshold: float = 0.0149
        self.clean_result(self.distance, self.label, self.main_imgs, self.comp_imgs, self.pic_idx)


    def _cfirst_to_clast(self, inputs):
        """
        Args:
            images: numpy array (N, C, W, H) or (C, W, H)
        return:
            images: numpy array(N, W, H, C) or (W, H, C)
        """
        inputs = np.swapaxes(inputs, -3, -2)
        inputs = np.swapaxes(inputs, -2, -1)
        return inputs

    def clean_result(self, distance: np.ndarray, label:torch.Tensor, main_imgs: torch.Tensor, comp_imgs: torch.Tensor, pic_idx: int,  figsize=(30,60)):
        i1 = self._cfirst_to_clast(main_imgs[pic_idx])
        i2 = self._cfirst_to_clast(comp_imgs[pic_idx])
        lbl = label[pic_idx]

        pl = distance[pic_idx].item()
        plt.figure(figsize=figsize)
        
        for i in range(1):
            plt.subplot(2, 1, 1+2 * i)
            plt.imshow(i1, cmap='gray')
            plt.xlabel(f'Predicted: {self.step(pl)} || {pl}', )
            
            plt.subplot(2, 1, 2+2 * i)
            plt.imshow(i2, cmap='gray')
            plt.xlabel(f'Ground Truth: {self.step(lbl.item())} || threshold: {self.threshold}')
        plt.show()

    def step(self, value):
        if value>self.threshold:
            return ("different")
        else:
            return ("same")
 

def show_time(total_time):
    unit = "s"
    if total_time<1:
        total_time = float(total_time * 1000)
        unit="ms"
    print(f'Execution Time: {total_time:.0f} {unit}')