import wandb
import hydra
from omegaconf import DictConfig
from datetime import datetime
import pathlib
import os

from fab.utils.logging import PandasLogger, WandbLogger, Logger
from fab_vae.models.vae import VAE

def setup_logger(cfg: DictConfig, save_path: str) -> Logger:
    if hasattr(cfg.logger, "pandas_logger"):
        logger = PandasLogger(save=True,
                              save_path=save_path + "logging_hist.csv",
                              save_period=cfg.logger.pandas_logger.save_period)
    elif hasattr(cfg.logger, "wandb"):
        logger = WandbLogger(**cfg.logger.wandb, config=dict(cfg))
    else:
        raise Exception("No logger specified, try adding the wandb or "
                        "pandas logger to the config file.")
    return logger


def train(cfg: DictConfig):
    current_time = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    save_path = cfg.logger.save_path + current_time + "/"
    if not hasattr(cfg.logger, "wandb"):
        pathlib.Path(save_path).mkdir(parents=True, exist_ok=False)
    logger = setup_logger(cfg, save_path)
    if hasattr(cfg.logger, "wandb"):
        # if using wandb then save to wandb path
        save_path = os.path.join(wandb.run.dir, save_path)
        pathlib.Path(save_path).mkdir(parents=True, exist_ok=False)
    with open(save_path + "config.txt", "w") as file:
        file.write(str(cfg))


    vae = VAE(loss_type=cfg.vae.loss_type,
              fab_loss_type=cfg.vae.fab_loss_type,
              latent_size=cfg.vae.latent_size,
              use_flow=cfg.vae.use_flow,
              use_conv=cfg.vae.use_conv,
              lr=cfg.vae.lr,
              batch_size=cfg.vae.batch_size,
              seed=cfg.vae.seed,
              n_samples_z_train=cfg.vae.n_samples_z_train,
              n_samples_test=cfg.vae.n_samples_z_test,
              ais_eval=cfg.vae.ais_eval,
              n_ais_dist=cfg.vae.n_ais_dist,
              logger=logger
              )
    vae.train(n_step=cfg.train.n_step, eval_freq=cfg.train.eval_freq)


@hydra.main(config_path="", config_name="config.yaml")
def run(config: DictConfig):
    train(config)

if __name__ == '__main__':
    run()
