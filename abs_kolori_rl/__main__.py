import datetime
import os
import sys
import warnings

import torch
import tqdm
from ray.rllib.algorithms import Algorithm, AlgorithmConfig
from ray.rllib.algorithms.ppo import PPOConfig

from .environment import KoloriEnv


def build_algo(
    game_path: str, num_rollout_workers: int = 1, evaluation_num_workers: int = 0
) -> Algorithm:
    config: AlgorithmConfig = (
        PPOConfig()
        .environment(
            KoloriEnv,
            env_config={"game_path": game_path},
        )
        .rollouts(num_rollout_workers=num_rollout_workers)
        .framework("torch")
        .training(model={"fcnet_hiddens": [512, 512, 256, 128, 64, 32]})
        .evaluation(
            evaluation_interval=1, evaluation_num_workers=evaluation_num_workers
        )
    )
    algo: Algorithm = config.build()
    return algo


def main() -> None:
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="numpy")

    game_path = os.environ["KOLORI_GAME_PATH"]

    print(
        f"Using PyTorch version {torch.__version__}"
        f" and CUDA {torch.version.cuda}"
        f" with {torch.cuda.device_count()} devices."
        + f" Device 0 is {torch.cuda.get_device_name(0)}."
        if torch.cuda.is_available()
        else ""
    )

    algo = build_algo(game_path, num_rollout_workers=8, evaluation_num_workers=8)

    print("Training...")

    try:
        for episode in range(1000):
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{now}] Episode {episode + 1}")
            print(algo.train())
            algo.save_checkpoint("checkpoints")
    except KeyboardInterrupt:
        print("Interrupted by user. Evaluating and saving checkpoint...")

    print("Evaluating...")
    algo.evaluate()

    algo.save_checkpoint("checkpoints")


def showcase() -> None:
    game_path = os.environ["KOLORI_GAME_PATH"]

    algo = build_algo(game_path)
    algo.load_checkpoint("checkpoints")

    print("Showcasing...")
    env = KoloriEnv({"game_path": game_path})
    episodes = 10
    steps = 6000
    for episode in tqdm.tqdm(range(episodes)):
        print(f"Episode {episode + 1}")
        observation = env.reset(return_info=False)
        if isinstance(observation, tuple):
            raise ValueError("The environment must return just an observation.")
        for _ in range(steps):
            action = algo.compute_single_action(observation)
            observation, reward, done, _ = env.step(action)
            env.render()
            if done:
                break


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "showcase":
        showcase()
    else:
        main()
