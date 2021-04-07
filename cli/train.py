import hydra
import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from label_augmented import utils, lightning
import logging
from pytorch_lightning.loggers import WandbLogger
import sys
import os


@hydra.main(config_path='../conf', config_name='config')
def main(config: DictConfig):

    # Set environment variables
    os.environ['HYDRA_FULL_ERROR'] = '1'

    print(OmegaConf.to_yaml(config))

    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        stream=sys.stdout,
    )

    logger = logging.getLogger(os.path.basename(__file__))

    pl.seed_everything(seed=config.general.seed)

    datamodule = utils.load_object(config=config.datamodule)
    logger.info('Data Module loaded')

    backbone = utils.load_object(config.backbone)
    logger.info('Backbone Loaded')

    aggregation = utils.load_object(config.aggregation)
    logger.info('Aggregation Loaded')

    head = utils.load_object(config.head)
    logger.info('Head Loaded')

    model = utils.model_assembly(backbone, aggregation, head)
    logger.info('Model Assembled')

    criterion = utils.load_object(config=config.loss)
    logger.info('Criterion Loaded')

    classifier = lightning.LightningClassifier(model=model,
                                               criterion=criterion,
                                               optimizer_config=config.optimizer)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(os.getcwd(), config.general.checkpoint_path),
        save_last=True,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )

    trainer = pl.Trainer(callbacks=[checkpoint_callback],
                         logger=WandbLogger(project=config.general.project_name),
                         **config.train)

    trainer.fit(model=classifier, datamodule=datamodule)


if __name__ == '__main__':
    main()
