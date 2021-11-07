from pathlib import Path
import logging
import argparse

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from siamese.data.datamodule import OmniglotDataModule
from siamese.modules.model import siamese_net

from pytorch_lightning.callbacks import QuantizationAwareTraining



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./dataset/omniglot/Malay_(Jawi_-_Arabic)')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--backbone_name', type=str, default='siamese')
    parser.add_argument('--simmilar_data_multiplier', type=int, default=20)

    
    
    parser = pl.Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()
    
    dict_args = vars(hparams)
    
    datamodule = OmniglotDataModule(**dict_args)
    # datamodule.setup()

    # print('validset lenght : ',len(datamodule.validset))
    module = siamese_net(pretrained=True, encoder_digit=32, **dict_args)
    print(module)
    model_checkpoint = ModelCheckpoint(
        dirpath='checkpoints/',
        save_top_k=1,
        filename="siamese-{val_step_loss:.4f}",
        verbose=True,
        monitor='val_step_loss',
        mode='min',
    )

    trainer = pl.Trainer.from_argparse_args(hparams, callbacks=[QuantizationAwareTraining(observer_type='histogram', input_compatible=True), model_checkpoint])
    # with mlflow.start_run() as run:
    trainer.fit(module, datamodule)
    trainer.save_checkpoint("checkpoints/latest.ckpt")
    
    metrics =  trainer.logged_metrics
    vloss = metrics['val_step_loss']
    
    filename = f'siamese-loss{vloss:.4f}.pth'
    saved_filename = str(Path('weights').joinpath(filename))
    
    logging.info(f"Prepare to save training results to path {saved_filename}")
    torch.save(module.feature_extractor.state_dict(), saved_filename)