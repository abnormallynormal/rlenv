import argparse
from pathlib import Path

import torch

from arcade.agents import train_agent
from arcade.environments import ENVIRONMENTS


CONFIGS = {
    "flappy": {
        "name": "Flappy Bird",
        "episodes": 5000,
        "epsilon_decay": 0.999,
        "filename": "flappy_bird.pt",
    },
    "snake": {
        "name": "Snake",
        "episodes": 5000,
        "epsilon_decay": 0.995,
        "filename": "snake.pt",
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("environment", choices=CONFIGS)
    args = parser.parse_args()
    config = CONFIGS[args.environment]

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    models = Path(__file__).parent / "models"
    models.mkdir(exist_ok=True)
    resume = models / f".{args.environment}-training.pt"

    print(
        f"Training {config['name']} for {config['episodes']:,} episodes "
        f"(epsilon 0.9999 -> 0.01, decay {config['epsilon_decay']})",
        flush=True,
    )
    if resume.exists():
        saved = torch.load(resume, map_location="cpu", weights_only=False)
        print(f"Resuming from episode {saved['episode']:,}", flush=True)

    def progress(done, total, average, epsilon):
        print(
            f"{config['name']}: {done:>5,}/{total:,} | "
            f"recent score {average:>7.2f} | epsilon {epsilon:.4f}",
            flush=True,
        )

    policy, scores, final_epsilon = train_agent(
        ENVIRONMENTS[config["name"]],
        config["episodes"],
        epsilon_decay=config["epsilon_decay"],
        checkpoint_path=resume,
        checkpoint_every=100,
        progress=progress,
    )
    destination = models / config["filename"]
    policy.save(
        destination,
        metadata={
            "environment": config["name"],
            "episodes": config["episodes"],
            "epsilon_start": 0.9999,
            "epsilon_min": 0.01,
            "epsilon_decay": config["epsilon_decay"],
            "final_epsilon": final_epsilon,
            "algorithm": "Dueling Double DQN with prioritized replay",
            "per_alpha": 0.6,
            "per_beta_start": 0.4,
            "per_beta_increment": 0.001,
            "optimizer": "Adam",
            "learning_rate": 0.0001,
            "gamma": 0.99,
            "batch_size": 32,
            "target_sync_steps": 5000,
            "recent_score": sum(scores[-100:]) / min(100, len(scores)),
        },
    )
    print(f"Saved final model to {destination}", flush=True)


if __name__ == "__main__":
    main()
