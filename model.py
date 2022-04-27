import jax
from flax import linen as nn


class LSTMOptimizer(nn.Module):
    hidden_units: int = 20

    def setup(self):
        self.lstm1 = nn.recurrent.LSTMCell()
        self.lstm2 = nn.recurrent.LSTMCell()
        self.fc = nn.Dense(1)

    def __call__(self, gradient, state):
        # gradients of optimizee do not depend on optimizer
        gradient = jax.lax.stop_gradient(gradient)

        # expand parameter dimension to extra batch dimension so that network
        # is "coodinatewise"
        gradient = gradient[..., None]

        carry1, carry2 = state
        carry1, x = self.lstm1(carry1, gradient)
        carry2, x = self.lstm2(carry2, x)
        update = self.fc(x)
        update = update[..., 0]  # remove last dimension
        return update, (carry1, carry2)

    def initialize_carry(self, rng, params):
        return (
            nn.LSTMCell.initialize_carry(rng, params.shape, self.hidden_units),
            nn.LSTMCell.initialize_carry(rng, params.shape, self.hidden_units),
        )
