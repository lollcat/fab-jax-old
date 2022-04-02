from typing import Tuple

from functools import partial
import optax
import chex
import jax.numpy as jnp
import jax
import tensorflow_datasets as tfds
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import pickle

from fab.utils.logging import ListLogger, to_numpy, Logger
from fab.utils.plotting import plot_history

from fab_vae.models.fab_types import VAENetworks, Params, Info, State, AISOutput
from fab_vae.utils.data import load_dataset, Batch, MNIST_IMAGE_SHAPE
from fab_vae.models.networks import make_vae_networks
from fab_vae.utils.numerical import remove_inf_and_nan
from fab_vae.sampling_methods.annealed_importance_sampling import AnnealedImportanceSampler
from fab_vae.models.encoder_only import VAE_encoder



class VAE_encoder_only(VAE_encoder):
    train_ds = load_dataset(tfds.Split.TRAIN, 1, shuffle=False)
    train_batch = next(train_ds)  # train
    x = train_batch["image"][0]

    def train(self,
              n_step: int = int(1e3),
              eval_freq: int = 50):

        valid_ds = load_dataset(tfds.Split.TEST, self.batch_size)

        train_batch = self.train_batch
        pbar = tqdm(range(n_step))
        for step in pbar:
            if step % eval_freq == 0:  # evaluate
                valid_batch = next(valid_ds)
                val_info = self.eval(self.state, valid_batch)
                pbar.set_description(f"STEP: {step}; Validation ELBO: {val_info['elbo']}")
                val_info = {"eval_" + key: val for key, val in val_info.items()}
                self.logger.write(to_numpy(val_info))

            self.state, info = self.update(self.state, train_batch)
            self.logger.write(to_numpy(info))


    def single_ais_run(self,
                       n_itermediate_dist = 50,
                       n_samples_z = 120,
                       use_prior_as_base: bool = True,
                       seed=0):
        x = self.x
        state = self.state
        key = jax.random.PRNGKey(seed)

        def base_log_prob(z):
            if use_prior_as_base:
                return self.vae_networks.prior_log_prob(z)
            else:
                return self.vae_networks.encoder_network.log_prob.apply(state.params.encoder, x, z)


        def target_log_prob(z):
            log_p_z = self.vae_networks.prior_log_prob(z)
            log_p_x_given_z = self.vae_networks.decoder_log_prob.apply(
                state.params.decoder, x, z)
            chex.assert_equal_shape((log_p_z, log_p_x_given_z))
            return log_p_z + log_p_x_given_z


        ais = AnnealedImportanceSampler(dim=self.latent_size,
                                        n_intermediate_distributions=n_itermediate_dist,
                                        )
        if use_prior_as_base:
            z_base, log_q_z_base = self.vae_networks.prior_sample_and_log_prob(
                seed=key, sample_shape=(n_samples_z,))
        else:
            z_base, log_q_z_base = self.vae_networks.encoder_network.sample_and_log_prob.apply(
                state.params.encoder, key, x, sample_shape=(n_samples_z,))
        z_ais, log_w_ais, new_transition_operator_state, info = \
            ais.run(z_base, log_q_z_base,
                         key,
                         state.transition_operator_state,
                         base_log_prob=base_log_prob,
                         target_log_prob=target_log_prob)
        pass
        print(info["ess_base"])
        print(info["ess_ais"])












if __name__ == '__main__':
    import numpy as np
    loss_type = "fab"
    n_z = 64
    batch_size = 8
    use_flow = True
    use_trained_encoder = False
    vae_enc = VAE_encoder_only(loss_type=loss_type, n_samples_z_train=n_z, batch_size=batch_size,
                          lr=1e-4, use_flow=use_flow, use_trained_encoder=use_trained_encoder,
                               n_ais_dist=4)
    """
    # Check single AIS run
    vae_enc.single_ais_run(n_samples_z=512,
                           n_itermediate_dist=128,
                           use_prior_as_base=True)
    """
    # ********************* Visualise in 2D  ****************************
    vae_enc.visualise_model(vae_enc.state, vae_enc.x)

    # ********************* Train ****************************
    vae_enc.visualise_model(vae_enc.state, vae_enc.x)
    vae_enc.train(n_step=500, eval_freq=1000)
    vae_enc.visualise_model(vae_enc.state, vae_enc.x)
    plot_history(vae_enc.logger.history)
    plt.show()

    print(np.asarray(vae_enc.logger.history["_ais_ess_base"]))
    print(np.asarray(vae_enc.logger.history["ess_ais"]))
    # print(np.max(np.asarray(vae_enc.logger.history["eval_elbo"])))
    # print(np.max(np.asarray(vae_enc.logger.history["eval_marginal_log_lik_vanilla"])))
    # print(np.max(np.asarray(vae_enc.logger.history["eval_marginal_log_lik_ais"])))

