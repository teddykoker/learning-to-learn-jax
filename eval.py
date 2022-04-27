from argparse import ArgumentParser
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from flax import serialization

from model import LSTMOptimizer
from tasks import quadratic_data, quadratic_task

HAND_OPTIMIZERS = {
    "sgd": optax.sgd,
    "adam": optax.adam,
    "rmsprop": partial(optax.rmsprop),
    "nag": partial(optax.sgd, momentum=0.9, nesterov=True),
}


def eval(args):
    rng = jax.random.PRNGKey(args.seed)
    w, y, theta = quadratic_data(rng, args.batch_size, args.dim)

    plt.figure(dpi=300, figsize=(5, 3))

    for name, optimizer in HAND_OPTIMIZERS.items():
        if name not in args.optimizers:
            continue

        best_losses = None
        for learning_rate in jnp.logspace(-2, 1, num=7):
            opt = optimizer(learning_rate=learning_rate)
            opt_state = opt.init(theta)
            losses, *_ = quadratic_task(w, y, theta, opt_state=opt_state, opt_fn=opt.update)

            if best_losses is None or losses[-1] < best_losses[-1]:
                best_losses = losses
                print(f"{name}, best lr: {learning_rate:.0e}")

        plt.plot(best_losses, label=name, linestyle="--")

    if "lstm" in args.optimizers:
        lstm_opt = LSTMOptimizer()
        lstm_state = lstm_opt.initialize_carry(rng, theta)
        params = lstm_opt.init(rng, theta, lstm_state)

        with open(args.model_path, "rb") as f:
            params = serialization.from_bytes(params, f.read())

        update = partial(lstm_opt.apply, params)
        losses, *_ = quadratic_task(w, y, theta, opt_state=lstm_state, opt_fn=update)
        plt.plot(losses, label="lstm")

    plt.legend()
    plt.title("losses")
    plt.xlabel("steps ($t$)")
    plt.ylabel("$f(\\theta)$")
    plt.yscale("log")
    plt.savefig(args.save_path, bbox_inches="tight")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--optimizers", nargs="+", default=["sgd"])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--dim", type=int, default=10)
    parser.add_argument("--model_path", default="models/params.mp")
    parser.add_argument("--save_path", default="figures/test.png")
    args = parser.parse_args()
    eval(args)
