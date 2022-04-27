import jax
from jax import vmap
import jax.numpy as jnp


def quadratic_data(rng, batch_size=128, dim=10):
    keys = jax.random.split(rng, 3)
    w = jax.random.normal(keys[0], (batch_size, dim, dim))
    y = jax.random.normal(keys[1], (batch_size, dim))
    theta = jax.random.normal(keys[2], (batch_size, dim))
    return w, y, theta


def quadratic_task(w, y, theta, opt_fn, opt_state, steps=100):
    @jax.jit
    def f(theta):
        product = vmap(jnp.matmul)(w, theta)
        return jnp.mean(jnp.sum((product - y) ** 2, axis=1))

    losses = []
    for _ in range(steps):
        loss, grads = jax.value_and_grad(f)(theta)
        updates, opt_state = opt_fn(grads, opt_state)
        theta += updates
        losses.append(loss)

    return jnp.stack(losses), theta, opt_state
