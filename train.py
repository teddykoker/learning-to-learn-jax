from argparse import ArgumentParser

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import wandb

from model import LSTMOptimizer
from tasks import quadratic_data, quadratic_task


def train(args):

    rng = jax.random.PRNGKey(seed=args.seed)

    lstm_opt = LSTMOptimizer(rng)
    lstm_state = lstm_opt.initialize_carry((args.batch_size, args.dim))

    meta_opt = optax.adam(learning_rate=args.learning_rate)
    meta_opt_state = meta_opt.init(lstm_opt)

    @eqx.filter_jit
    def train_step(opt, w, y, theta, state):
        def loss_fn(opt):
            losses, _theta, _state = quadratic_task(
                w, y, theta, opt, state, args.unroll_steps
            )
            return losses.sum(), (_theta, _state)

        (loss, (_theta, _state)), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(opt)
        return loss, grads, _theta, _state

    best_loss = jnp.inf

    for _ in range(args.steps):
        rng, subkey = jax.random.split(rng)
        w, y, theta = quadratic_data(subkey, args.batch_size, args.dim)
        lstm_state = lstm_opt.initialize_carry((args.batch_size, args.dim))
        meta_loss = 0.0

        for _ in range(args.unrolls):
            loss, grads, theta, lstm_state = train_step(lstm_opt, w, y, theta, lstm_state)
            updates, meta_opt_state = meta_opt.update(grads, meta_opt_state)
            lstm_opt = optax.apply_updates(lstm_opt, updates)
            meta_loss += loss

        if meta_loss < best_loss:
            best_loss = meta_loss
            eqx.tree_serialise_leaves(args.model_path, lstm_opt)

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
    parser.add_argument("--model_path", default="models/params.eqx")
    args = parser.parse_args()
    wandb.init(config=args, project="learning-to-learn-jax")
    train(args)
