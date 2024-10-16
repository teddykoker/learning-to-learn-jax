from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp


class LSTMOptimizer(eqx.Module):
    lstm1: eqx.nn.LSTMCell
    lstm2: eqx.nn.LSTMCell
    fc: eqx.nn.Linear
    hidden_units: int = 10

    def __init__(self, key):
        keys = jax.random.split(key, 3)
        self.lstm1 = eqx.nn.LSTMCell(1, self.hidden_units, key=keys[0])
        self.lstm2 = eqx.nn.LSTMCell(self.hidden_units, self.hidden_units, key=keys[1])
        self.fc = eqx.nn.Linear(self.hidden_units, 1, key=keys[2])

    @partial(jax.vmap, in_axes=(None, 0, 0))
    @partial(jax.vmap, in_axes=(None, 0, 0))
    def __call__(self, gradient, state):
        # gradients of optimizee do not depend on optimizer
        gradient = jax.lax.stop_gradient(gradient)

        # expand parameter dimension to extra batch dimension so that network
        # is "coodinatewise"
        gradient = gradient[..., None]

        carry1, carry2 = state
        carry1 = self.lstm1(gradient, carry1)
        carry2 = self.lstm2(carry1[0], carry2)
        update = self.fc(carry2[0])
        update = update[..., 0]  # remove last dimension
        return update, (carry1, carry2)

    def initialize_carry(self, input_shape):
        shape = input_shape + (self.hidden_units,)
        return (
            (jnp.zeros(shape), jnp.zeros(shape)),
            (jnp.zeros(shape), jnp.zeros(shape)),
        )
