from argparse import ArgumentParser
from functools import partial

import jax
import jax.numpy as jnp
import optax
import wandb
from flax import serialization
from jax import value_and_grad

from model import LSTMOptimizer
from tasks import quadratic_data, quadratic_task


def train(args):

    rng = jax.random.PRNGKey(seed=args.seed)

    example_input = jnp.zeros((args.batch_size, args.dim))
    lstm_opt = LSTMOptimizer()
    lstm_state = lstm_opt.initialize_carry(rng, example_input)
    params = lstm_opt.init(rng, example_input, lstm_state)

    meta_opt = optax.adam(learning_rate=args.learning_rate)
    meta_opt_state = meta_opt.init(params)

    @jax.jit
    def train_step(params, w, y, theta, state):
        def loss_fn(params):
            update = partial(lstm_opt.apply, params)
            losses, _theta, _state = quadratic_task(
                w, y, theta, update, state, args.unroll_steps
            )
            return losses.sum(), (_theta, _state)

        (loss, (_theta, _state)), grads = value_and_grad(loss_fn, has_aux=True)(params)
        return loss, grads, _theta, _state

    best_loss = jnp.inf

    for _ in range(args.steps):
        rng, subkey = jax.random.split(rng)
        w, y, theta = quadratic_data(subkey, args.batch_size, args.dim)
        lstm_state = lstm_opt.initialize_carry(rng, theta)
        meta_loss = 0.0

        for _ in range(args.unrolls):
            loss, grads, theta, lstm_state = train_step(params, w, y, theta, lstm_state)
            updates, meta_opt_state = meta_opt.update(grads, meta_opt_state)
            params = optax.apply_updates(params, updates)
            meta_loss += loss

        if meta_loss < best_loss:
            best_loss = meta_loss
            with open(args.model_path, "wb") as f:
                f.write(serialization.to_bytes(params))

        wandb.log({"loss": meta_loss})

    wandb.log({"best_loss": best_loss})


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--dim", type=int, default=10)
    parser.add_argument("--unroll_steps", type=int, default=20)
    parser.add_argument("--unrolls", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--model_path", default="models/params.mp")
    args = parser.parse_args()
    wandb.init(config=args, project="learning-to-learn-jax")
    train(args)
