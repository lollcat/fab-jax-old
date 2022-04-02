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
from fab_vae.models.vae import VAE

class VAE_encoder(VAE):
    def __init__(
            self,
            loss_type,
            use_trained_encoder: bool = False,
            use_flow: bool = True,
            batch_size: int = 128,
            n_samples_z_train: int = 20,
            n_samples_test: int = 20,
            latent_size: int = 2,
            seed: int = 0,
            ais_eval: bool = True,
            *args,
            **kwargs,
    ):
        super(VAE_encoder, self).__init__(
            latent_size=latent_size,
            loss_type=loss_type,
            use_flow=use_flow,
            batch_size=batch_size,
            n_samples_test=n_samples_test,
            n_samples_z_train=n_samples_z_train,
            ais_eval=ais_eval,
            seed=seed,
            *args,
            **kwargs,
        )
        init_state = self.state
        # overwrites state with saved encoder
        if use_flow:
            self.load("saved_model_2d")
        else:
            self.load("saved_model_2d_gaussian")
        params = Params(
            encoder=self.state.params.encoder if use_trained_encoder else init_state.params.encoder,
            decoder=self.state.params.decoder
                        )
        self.state = State(
            params=params,
            opt_state=init_state.opt_state,
            rng_key=init_state.rng_key,
            transition_operator_state=init_state.transition_operator_state
                           )
        self.use_ais = True

    def init_state(self, seed) -> State:
        rng_key = jax.random.PRNGKey(seed)
        key1, key2 = jax.random.split(rng_key)
        params = self.vae_networks.init(key1)
        opt_state = self.optimizer.init(params.encoder) # only optimizing encoder now
        transition_operator_state = self.ais.transition_operator_manager.get_init_state()
        state = State(params=params,
                      rng_key=rng_key,
                      opt_state=opt_state,
                      transition_operator_state=transition_operator_state)
        return state

    @partial(jax.jit, static_argnums=0)
    def update(
            self,
            state: State,
            batch: Batch,
    ) -> Tuple[State, Info]:
        """Single SGD update step."""
        key, subkey = jax.random.split(state.rng_key)
        info = {}
        if self.use_ais:
            grads_fab, transition_operator_state, info_fab = self.get_fab_grad(state, batch)
            chex.assert_trees_all_equal_shapes(state.transition_operator_state,
                                              transition_operator_state)
            info.update(info_fab)
        if self.use_vanilla:
            grads_vanilla, info_vanilla = jax.grad(self.vanilla_loss_fn, argnums=0,
                                                   has_aux=True)(state.params,
                                                                 batch["image"],
                                                                 subkey)
            info.update(info_vanilla)
        if self.loss_type == "fab":
            grads = grads_fab.encoder
        elif self.loss_type == "fab_combo":
            grads = jax.tree_map(lambda x, y: x + y, grads_fab.encoder, grads_vanilla.encoder)
        else: #  self.loss_type == "vanilla"
            grads = grads_vanilla.encoder
        if not "transition_operator_state" in locals():
            transition_operator_state = state.transition_operator_state
        updates, new_opt_state = self.optimizer.update(grads, state.opt_state)
        new_params_encoder = optax.apply_updates(state.params.encoder, updates)
        state = State(params=Params(encoder=new_params_encoder,
                                    decoder=state.params.decoder),
                      opt_state=new_opt_state,
                      transition_operator_state=transition_operator_state,
                      rng_key=key)
        return state, info


if __name__ == '__main__':
    # need x for plotting
    train_ds = load_dataset(tfds.Split.TRAIN, 10, shuffle=False)
    train_batch = next(train_ds)  # train
    x1 = train_batch["image"][0]
    x2 = train_batch["image"][1]
    x3 = train_batch["image"][2]


    import numpy as np
    loss_type = "vanilla"
    n_z = 64
    batch_size = 24
    use_trained_encoder = False
    use_flow = True
    vae_enc = VAE_encoder(
        loss_type=loss_type,
        n_samples_z_train=n_z,
        batch_size=batch_size,
        lr=1e-4,
        n_samples_test=100,
        n_ais_dist=8,
        use_trained_encoder=use_trained_encoder,
        use_flow=use_flow
    )

    print(n_z)
    initial_state = vae_enc.state
    vae_enc.visualise_model(vae_enc.state, x1)
    vae_enc.visualise_model(vae_enc.state, x2)
    vae_enc.visualise_model(vae_enc.state, x3)
    vae_enc.train(n_step=500, eval_freq=50)
    trained_state = vae_enc.state
    vae_enc.visualise_model(vae_enc.state, x1)
    vae_enc.visualise_model(vae_enc.state, x2)
    plot_history(vae_enc.logger.history)
    plt.show()
    vae_enc.save(loss_type)
    print(np.asarray(vae_enc.logger.history["_ais_ess_base"])[0:5])
    print(np.asarray(vae_enc.logger.history["_ais_ess_base"])[-5:])
    print(np.asarray(vae_enc.logger.history["_ais_ess_ais"])[0:5])
    print(np.asarray(vae_enc.logger.history["_ais_ess_ais"])[-5:])

    print(np.max(np.asarray(vae_enc.logger.history["eval_elbo"])))
    print(np.max(np.asarray(vae_enc.logger.history["eval_marginal_log_lik_vanilla"])))
    print(np.max(np.asarray(vae_enc.logger.history["eval_marginal_log_lik_ais"])))

