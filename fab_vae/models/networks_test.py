import jax.numpy as jnp
import jax
import chex

from fab_vae.models.networks import make_vae_networks, MNIST_IMAGE_SHAPE

def test_vae_networks(
    use_flow = True,
    use_conv = False,
    latent_size = 2,
    x_shape = (5, 3, 2)  # MNIST_IMAGE_SHAPE  #
):
    """
    check consistency across:
     - single z vs multiple z for single x
     - vmapping over multiple x
     - batching over multiple x for single z
    """
    key = jax.random.PRNGKey(0)
    vae_networks = make_vae_networks(latent_size=latent_size,
                                     output_shape=x_shape,
                                     use_flow=use_flow,
                                     use_conv=use_conv)
    params = vae_networks.init(key)

    x_sample = jax.random.normal(key, shape=x_shape)

    def target_log_prob(z):
        return vae_networks.prior_log_prob(z) + vae_networks.decoder_log_prob.apply(
            params.decoder, x_sample, z)

    # single x, single z
    z_sample, z_log_q = vae_networks.encoder_network.sample_and_log_prob.apply(
                params.encoder, key, x_sample, sample_shape=())
    z_log_q_ = vae_networks.encoder_network.log_prob.apply(params.encoder, x_sample, z_sample)
    chex.assert_trees_all_close(z_log_q_, z_log_q)
    log_p_x_z = target_log_prob(z_sample)
    chex.assert_equal_shape((log_p_x_z, z_log_q))

    # for single x, batch of z
    z_batch, log_q_z_batch = vae_networks.encoder_network.sample_and_log_prob.apply(
                params.encoder, key, x_sample, sample_shape=(5,))
    log_q_z_batch_ = vae_networks.encoder_network.log_prob.apply(params.encoder, x_sample, z_batch)
    chex.assert_trees_all_close(log_q_z_batch_, log_q_z_batch)
    log_p_x_z_batch = target_log_prob(z_batch)

    log_q_z_first = vae_networks.encoder_network.log_prob.apply(params.encoder,
                                                                x_sample, z_batch[0])
    log_p_x_z_first = target_log_prob(z_batch[0])
    chex.assert_trees_all_close(log_p_x_z_first, log_p_x_z_batch[0])
    chex.assert_trees_all_close(log_q_z_first, log_q_z_batch[0])

    # batch x, single z.
    x_batch = jax.random.normal(key, shape=(4, *x_shape))

    def target_log_prob_batch(z):
        return vae_networks.prior_log_prob(z) + vae_networks.decoder_log_prob.apply(
            params.decoder, x_batch, z)

    z_batch, log_q_z_batch = jax.vmap(vae_networks.encoder_network.sample_and_log_prob.apply,
                                      in_axes=(None, None, 0, None))(
        params.encoder, key, x_batch, ())

    log_p_x_z_batch = target_log_prob_batch(z_batch)

    log_q_z_single_x = vae_networks.encoder_network.log_prob.apply(params.encoder,
                                                                   x_batch[0],
                                                                   z_batch[0])
    log_p_x_z_single_x = vae_networks.prior_log_prob(z_batch[0]) + \
                       vae_networks.decoder_log_prob.apply(params.decoder, x_batch[0], z_batch[0])
    chex.assert_trees_all_close(log_q_z_batch[0], log_q_z_single_x)
    chex.assert_trees_all_close(log_p_x_z_single_x, log_p_x_z_batch[0])


if __name__ == '__main__':
    test_vae_networks(use_flow=False)
