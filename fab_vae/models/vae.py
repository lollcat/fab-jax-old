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
from fab.utils.plotting import plot_contours_2D


class VAE:
    def __init__(
            self,
            loss_type: str = "fab",
            latent_size: int = 10,
            use_flow: bool = True,
            use_conv: bool = False,
            lr: float = 1e-3,
            max_grad_norm: float = 1.0,
            batch_size: int = 64,
            seed: int = 0,
            n_samples_z_train: int = 10,
            n_samples_test: int = 20,
            ais_eval: bool = True,
            n_ais_dist: int = 4,
            logger: Logger = ListLogger()
    ):
        self.loss_type = loss_type
        self.vae_networks = make_vae_networks(
            latent_size=latent_size,
            output_shape=MNIST_IMAGE_SHAPE,
            use_flow=use_flow,
            use_conv=use_conv
        )
        self.lr = lr
        self.latent_size = latent_size
        self.use_flow = use_flow
        self.optimizer = optax.chain(optax.zero_nans(),
                            optax.clip_by_global_norm(max_grad_norm),
                            optax.adam(lr))
        self.batch_size = batch_size
        self.seed = seed
        self.logger = logger
        assert loss_type in ["fab", "fab_combo", "fab_decoder", "vanilla"]
        self.use_vanilla = False if loss_type == "fab" else True
        self.ais = AnnealedImportanceSampler(dim=latent_size,
                                                 n_intermediate_distributions=n_ais_dist)
        self.n_samples_z_train = n_samples_z_train
        self.n_samples_z_test = n_samples_test
        if "fab" in loss_type:
            self.use_ais: bool = True
        else:
            self.use_ais: bool = False
        self.ais_eval = ais_eval
        self.state = self.init_state(seed)


    def init_state(self, seed) -> State:
        rng_key = jax.random.PRNGKey(seed)
        key1, key2 = jax.random.split(rng_key)
        params = self.vae_networks.init(key1)
        opt_state = self.optimizer.init(params)
        transition_operator_state = self.ais.transition_operator_manager.get_init_state()
        state = State(params=params,
                      rng_key=rng_key,
                      opt_state=opt_state,
                      transition_operator_state=transition_operator_state)
        return state


    def ais_forward(
            self,
            state: State,
            x_batch: chex.Array,
            n_samples_z) -> AISOutput:


        def ais_forward_inner(rng_key: chex.PRNGKey, x: chex.Array):
            key1, key2 = jax.random.split(rng_key)

            def base_log_prob(z):
                return self.vae_networks.encoder_network.log_prob.apply(state.params.encoder, x, z)

            def target_log_prob(z):
                log_p_z = self.vae_networks.prior_log_prob(z)
                log_p_x_given_z = self.vae_networks.decoder_log_prob.apply(
                    state.params.decoder, x, z)
                chex.assert_equal_shape((log_p_z, log_p_x_given_z))
                return log_p_z + log_p_x_given_z

            z_base, log_q_z_base = self.vae_networks.encoder_network.sample_and_log_prob.apply(
                state.params.encoder, key1, x, sample_shape=(n_samples_z,))
            z_ais, log_w_ais, new_transition_operator_state, info = \
                self.ais.run(z_base, log_q_z_base,
                             key2,
                             state.transition_operator_state,
                             base_log_prob=base_log_prob,
                             target_log_prob=target_log_prob)
            return log_w_ais, z_ais, new_transition_operator_state, info

        key_batch = jax.random.split(state.rng_key, x_batch.shape[0])
        log_w_ais, z_ais, transition_operator_state, info = \
            jax.vmap(ais_forward_inner, in_axes=(0, 0))(key_batch, x_batch)
        info, transition_operator_state = jax.tree_map(lambda x: jnp.mean(x, axis=0),
                                                       (info, transition_operator_state))
        z_ais, log_w_ais, invalid_sample_frac = remove_inf_and_nan(z_ais, log_w_ais)
        info.update(invalid_sample_frac=invalid_sample_frac)
        ais_output = AISOutput(z_ais=z_ais, log_w_ais=log_w_ais,
                               transition_operator_state=transition_operator_state, info=info)
        return ais_output

    def estimate_marginal_log_lik_vanilla(self, state, x, n_samples_z):
        z, log_q_z_given_x = self.vae_networks.encoder_network.sample_and_log_prob.apply(
            state.params.encoder, state.rng_key, x, sample_shape=(n_samples_z, ))
        log_p_x_given_z = jax.vmap(self.vae_networks.decoder_log_prob.apply,
                                   in_axes=(None, None, 0))(state.params.decoder, x, z)
        log_p_z = self.vae_networks.prior_log_prob(z)
        return jnp.mean(jax.nn.logsumexp(log_p_x_given_z + log_p_z - log_q_z_given_x, axis=0)) - \
               jnp.log(n_samples_z)

    def eval_with_ais(self, state, x, n_samples_z) -> Info:
        ais_output = self.ais_forward(state, x, n_samples_z=n_samples_z)
        marg_log_lik = jnp.mean(jax.nn.logsumexp(ais_output.log_w_ais, axis=1)) - jnp.log(n_samples_z)
        info = {"marginal_log_lik_ais": marg_log_lik}
        info.update(ess_base=ais_output.info["ess_base"],
                    ess_ais=ais_output.info["ess_ais"])
        return info

    def fab_loss_fn(self, params: Params, z_ais_batch: chex.Array, log_w_ais_batch: chex.Array,
                    x_batch: chex.Array):
        def inner_loss_fn(z_ais: chex.Array, log_w_ais: chex.Array, x: chex.Array) -> \
                Tuple[jnp.ndarray, Info]:
            chex.assert_shape(z_ais, (self.n_samples_z_train, self.latent_size))
            chex.assert_shape(log_w_ais, (self.n_samples_z_train,))
            log_q_z_given_x = self.vae_networks.encoder_network.log_prob.apply(params.encoder, x,
                                                                               z_ais)
            log_p_x_z = self.vae_networks.decoder_log_prob.apply(params.decoder, x, z_ais) + \
                        self.vae_networks.prior_log_prob(z_ais)
            chex.assert_equal_shape((log_p_x_z, log_q_z_given_x, log_w_ais))
            # decoder_loss = - jnp.sum(jax.nn.softmax(log_w_ais, axis=0) * log_p_x_z)
            decoder_loss = jax.nn.logsumexp(log_w_ais +
                            jax.lax.stop_gradient(log_p_x_z) - log_q_z_given_x)
            encoder_loss = - jnp.sum(jax.nn.softmax(log_w_ais, axis=0) * log_q_z_given_x)
            info = {}
            info.update(decoder_loss=decoder_loss, encoder_loss=encoder_loss)
            return encoder_loss + decoder_loss, info

        loss, info = jax.vmap(inner_loss_fn, in_axes=(0, 0, 0))(z_ais_batch,
                                                                log_w_ais_batch,
                                                                x_batch)
        info = jax.tree_map(jnp.mean, info)
        return jnp.mean(loss), info

    def get_fab_grad(self, state: State, batch: Batch) -> Tuple[Params, chex.ArrayTree, Info]:
        ais_output: AISOutput = self.ais_forward(state, batch["image"], self.n_samples_z_train)
        (loss_fab, info), grads_fab = jax.value_and_grad(self.fab_loss_fn, has_aux=True, argnums=0)(
          state.params, ais_output.z_ais, ais_output.log_w_ais, batch["image"])
        ais_info = {"_ais_" + key: val for key, val in ais_output.info.items()}
        info.update(ais_info)
        return grads_fab, ais_output.transition_operator_state, info


    def vanilla_loss_fn(self, params: Params, x: chex.Array, rng_key: chex.PRNGKey) -> \
            Tuple[chex.Array, Info]:
        z, log_q_z_given_x = self.vae_networks.encoder_network.sample_and_log_prob.apply(
            params.encoder, rng_key, x, sample_shape=())
        log_p_x_given_z = self.vae_networks.decoder_log_prob.apply(params.decoder, x, z)
        log_p_z = self.vae_networks.prior_log_prob(z)
        kl = log_q_z_given_x - log_p_z
        elbo = jnp.mean(log_p_x_given_z - kl)
        info = {"elbo": elbo,
                "kl": jnp.mean(kl),
                "log_p_x_given_z": jnp.mean(log_p_x_given_z),
                "log_q_z_given_x": jnp.mean(log_q_z_given_x),
                "log_p_z ": jnp.mean(log_p_z )
                }
        return - elbo, info

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
            grads = grads_fab
        elif self.loss_type == "fab_decoder":
            grads = Params(encoder=grads_vanilla.encoder, decoder=grads_fab.decoder)
        elif self.loss_type == "fab_combo":
            grads = jax.tree_map(lambda x, y: x + y, grads_fab, grads_vanilla)
        else: #  self.loss_type == "vanilla"
            grads = grads_vanilla

        if not "transition_operator_state" in locals():
            transition_operator_state = state.transition_operator_state
        updates, new_opt_state = self.optimizer.update(grads, state.opt_state)
        new_params = optax.apply_updates(state.params, updates)
        # TODO: could be cleaner
        chex.assert_trees_all_equal_shapes(state.params, new_params)
        state = State(params=new_params, opt_state=new_opt_state,
                      transition_operator_state=transition_operator_state,
                      rng_key=key)
        return state, info


    @partial(jax.jit, static_argnums=0)
    def eval(
        self,
        state: State,
        batch: Batch,
        ) -> Info:
        """Evaluate the model."""
        info = {}
        vanilla_loss, vanilla_info = self.vanilla_loss_fn(state.params, batch["image"],
                                                          state.rng_key)
        info.update(vanilla_info)
        marginal_log_lik_vanilla = self.estimate_marginal_log_lik_vanilla(state, batch["image"],
                                                                          self.n_samples_z_test)
        info.update(marginal_log_lik_vanilla=marginal_log_lik_vanilla)
        if self.ais_eval:
            ais_info = self.eval_with_ais(state, batch["image"], self.n_samples_z_test)
            info.update(ais_info)
        return info


    def train(self,
              n_step: int = int(1e3),
              eval_freq: int = 50):
        train_ds = load_dataset(tfds.Split.TRAIN, self.batch_size)
        valid_ds = load_dataset(tfds.Split.TEST, self.batch_size)

        pbar = tqdm(range(n_step))
        for step in pbar:
            if step % eval_freq == 0:  # evaluate
                valid_batch = next(valid_ds)
                val_info = self.eval(self.state, valid_batch)
                pbar.set_description(f"STEP: {step}; Validation ELBO: {val_info['elbo']}")
                val_info = {"eval_" + key: val for key, val in val_info.items()}
                self.logger.write(to_numpy(val_info))

            train_batch = next(train_ds) # train
            if train_batch["image"].shape[0] != self.batch_size:
                print(f"batch size changed to {train_batch['image'].shape[0]}")

            self.state, info = self.update(self.state, train_batch)
            self.logger.write(to_numpy(info))


    def save(self, path: str):
        with open(os.path.join(path, "state.pkl"), "wb") as f:
            pickle.dump(self.state, f)

    def load(self, path: str):
        self.state = pickle.load(open(os.path.join(path, "state.pkl"), "rb"))

    def visualise_model(self, state, x, n_samples_z = 100, ax1 = None, ax2=None):
        key = jax.random.PRNGKey(0)


        def target_log_prob(z):
            log_p_z = self.vae_networks.prior_log_prob(z)
            log_p_x_given_z = self.vae_networks.decoder_log_prob.apply(
                state.params.decoder, x, z)
            chex.assert_equal_shape((log_p_z, log_p_x_given_z))
            return log_p_z + log_p_x_given_z

        def base_log_prob(z):
            return self.vae_networks.encoder_network.log_prob.apply(state.params.encoder, x, z)

        z_base, log_q_z_base = self.vae_networks.encoder_network.sample_and_log_prob.apply(
            state.params.encoder, key, x, sample_shape=(n_samples_z,))


        z_ais, log_w_ais, new_transition_operator_state, info = \
            self.ais.run(z_base, log_q_z_base,
                    key,
                    state.transition_operator_state,
                    base_log_prob=base_log_prob,
                    target_log_prob=target_log_prob)

        fig, axs = plt.subplots(1, 2)
        bound = float(jnp.max(jnp.abs(z_ais)) + 1)
        plot_contours_2D(target_log_prob, ax=axs[0], bound=bound, levels=50)
        axs[0].plot(z_base[:, 0], z_base[:, 1], "o", alpha=0.5)

        plot_contours_2D(target_log_prob, ax=axs[1], bound=bound, levels=50)
        axs[1].plot(z_ais[:, 0], z_ais[:, 1], "o", alpha=0.5)
        plt.show()

        z_sample_indx = 1
        fig, axs = plt.subplots(1, 3)
        axs[0].imshow(x, cmap="gray")
        dist_x_given_z_base = self.vae_networks.decoder_forward.apply(state.params.decoder,
                                                                      z_base[z_sample_indx])
        x_base = dist_x_given_z_base.sample(seed=key)
        axs[1].imshow(x_base, cmap="gray")
        dist_x_given_z_ais = self.vae_networks.decoder_forward.apply(state.params.decoder,
                                                                      z_ais[z_sample_indx])
        x_ais = dist_x_given_z_ais.sample(seed=key)
        axs[2].imshow(x_ais, cmap="gray")
        plt.show()



if __name__ == '__main__':
    import numpy as np
    loss_type = "vanilla"  # "fab_decoder", "fab_combo", "vanilla", "fab"
    print(loss_type)
    vae = VAE(use_flow=False,
              batch_size=128,
              latent_size=2,
              loss_type=loss_type,
              n_samples_z_train=20,
              n_samples_test=20,
              seed=0,
              ais_eval=False,
              lr=1e-4)

    vae.train(n_step=5000, eval_freq=1000)

    plot_history(vae.logger.history)
    plt.show()

    print(np.max(np.asarray(vae.logger.history["eval_elbo"])))
    print(np.max(np.asarray(vae.logger.history["eval_marginal_log_lik_vanilla"])))
    print(np.max(np.asarray(vae.logger.history["eval_marginal_log_lik_ais"])))
    # np.asarray(vae.logger.history["eval_elbo"])[-5:]
    # np.asarray(vae.logger.history["eval_marginal_log_lik_vanilla"])[-5:]
    # np.asarray(vae.logger.history["eval_marginal_log_lik_ais"])[-5:]









