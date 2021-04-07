import logging
import os
import sys

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import WandbLogger

from label_augmented import utils, lightning


@hydra.main(config_path='../conf', config_name='config')
def main(config: DictConfig):

    os.environ['HYDRA_FULL_ERROR'] = '1'

    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        stream=sys.stdout,
    )

    logger = logging.getLogger(os.path.basename(__file__))

    pl.seed_everything(seed=config.general.seed)
    logger.info(OmegaConf.to_yaml(config))

    datamodule = utils.load_object(config=config.datamodule)
    logger.info('Data Module Loaded')

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
        monitor='valid_epoch_loss',
        mode='min'
    )

    trainer = pl.Trainer(callbacks=[checkpoint_callback],
                         logger=WandbLogger(project=config.general.project_name),
                         **config.train)

    trainer.fit(model=classifier, datamodule=datamodule)


if __name__ == '__main__':
    main()
