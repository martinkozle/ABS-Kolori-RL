import socket
import subprocess
from typing import Any, TypedDict, cast

import numpy as np
import numpy.typing as npt
import zmq
from gym import Env, spaces
from ray.rllib.env import EnvContext

BoxSampleType = npt.NDArray[np.float32]


class SpeedBoostType(TypedDict):
    speed_boost_multiplier: BoxSampleType
    speed_boost_current_duration: BoxSampleType
    speed_boost_duration: BoxSampleType
    speed_boost_active: BoxSampleType


class PlayerType(TypedDict):
    position: BoxSampleType
    speed: BoxSampleType
    speed_boost: SpeedBoostType
    trail_color: BoxSampleType
    score: BoxSampleType
    health: BoxSampleType


class EnemiesType(TypedDict):
    count: BoxSampleType
    positions: BoxSampleType


class PaintBucketsType(TypedDict):
    count: BoxSampleType
    positions: BoxSampleType
    types: BoxSampleType


class ProjectilesType(TypedDict):
    count: BoxSampleType
    positions: BoxSampleType
    angles: BoxSampleType
    speeds: BoxSampleType
    durations: BoxSampleType
    current_durations: BoxSampleType


class IngameStateType(TypedDict):
    player: PlayerType
    enemies: EnemiesType
    paint_buckets: PaintBucketsType
    projectiles: ProjectilesType
    time_scale_active: BoxSampleType
    time_scale: BoxSampleType
    time_scale_duration: BoxSampleType


class ObservationSpaceType(TypedDict):
    elapsed_game_time: BoxSampleType
    total_game_time: BoxSampleType
    ingame_state: IngameStateType


class ActionSpaceType(TypedDict):
    move_horisontal: spaces.Discrete
    move_vertical: spaces.Discrete
    ability: spaces.Discrete


def find_free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return int(s.getsockname()[1])


class KoloriEnv(Env[ObservationSpaceType, ActionSpaceType]):
    def __init__(
        self,
        env_config: EnvContext | dict[str, Any],
        number_of_closest_enemies: int = 5,
        number_of_closest_paint_buckets: int = 5,
        number_of_closest_projectiles: int = 5,
    ) -> None:
        self.number_of_closest_enemies = number_of_closest_enemies
        self.number_of_closest_paint_buckets = number_of_closest_paint_buckets
        self.number_of_closest_projectiles = number_of_closest_projectiles
        self.observation_space = cast(
            spaces.Space[ObservationSpaceType],
            spaces.Dict(
                elapsed_game_time=spaces.Box(low=0, high=np.inf, shape=(1,)),
                total_game_time=spaces.Box(low=0, high=np.inf, shape=(1,)),
                ingame_state=spaces.Dict(
                    player=spaces.Dict(
                        position=spaces.Box(low=16, high=784, shape=(2,)),
                        speed=spaces.Box(low=-675, high=675, shape=(2,)),
                        speed_boost=spaces.Dict(
                            speed_boost_multiplier=spaces.Box(
                                low=0, high=np.inf, shape=(1,)
                            ),
                            speed_boost_current_duration=spaces.Box(
                                low=0, high=np.inf, shape=(1,)
                            ),
                            speed_boost_duration=spaces.Box(
                                low=0, high=np.inf, shape=(1,)
                            ),
                            speed_boost_active=spaces.Box(low=0, high=1, shape=(1,)),
                        ),
                        trail_color=spaces.Box(low=0, high=1, shape=(3,)),
                        score=spaces.Box(low=0, high=np.inf, shape=(1,)),
                        health=spaces.Box(low=0, high=100, shape=(1,)),
                    ),
                    enemies=spaces.Dict(
                        count=spaces.Box(low=0, high=np.inf, shape=(1,)),
                        positions=spaces.Box(
                            low=0, high=800, shape=(self.number_of_closest_enemies, 2)
                        ),
                    ),
                    paint_buckets=spaces.Dict(
                        count=spaces.Box(low=0, high=np.inf, shape=(1,)),
                        positions=spaces.Box(
                            low=0,
                            high=800,
                            shape=(self.number_of_closest_paint_buckets, 2),
                        ),
                        types=spaces.Box(
                            low=0,
                            high=1,
                            shape=(self.number_of_closest_paint_buckets, 3),
                        ),
                    ),
                    projectiles=spaces.Dict(
                        count=spaces.Box(low=0, high=np.inf, shape=(1,)),
                        positions=spaces.Box(
                            low=0,
                            high=800,
                            shape=(self.number_of_closest_projectiles, 2),
                        ),
                        angles=spaces.Box(
                            low=0,
                            high=2 * np.pi,
                            shape=(self.number_of_closest_projectiles,),
                        ),
                        speeds=spaces.Box(
                            low=0,
                            high=np.inf,
                            shape=(self.number_of_closest_projectiles,),
                        ),
                        durations=spaces.Box(
                            low=0,
                            high=np.inf,
                            shape=(self.number_of_closest_projectiles,),
                        ),
                        current_durations=spaces.Box(
                            low=0,
                            high=np.inf,
                            shape=(self.number_of_closest_projectiles,),
                        ),
                    ),
                    time_scale_active=spaces.Box(low=0, high=1, shape=(1,)),
                    time_scale=spaces.Box(low=0, high=np.inf, shape=(1,)),
                    time_scale_duration=spaces.Box(low=0, high=np.inf, shape=(1,)),
                ),
            ),
        )

        self.observation_space._shape = (86,)

        self.action_space = cast(
            spaces.Space[ActionSpaceType],
            spaces.Dict(
                move_horisontal=spaces.Discrete(3, start=-1),
                move_vertical=spaces.Discrete(3, start=-1),
                ability=spaces.Discrete(2),
            ),
        )

        self.action_space._shape = (3,)

        self.reward_range = (0, np.inf)

        self.max_episode_steps = 30 * 300
        self.current_step_count = 0

        self.render_next = False

        if "game_path" in env_config:
            game_path: str = env_config["game_path"]
        else:
            raise ValueError("game_path not specified in env_config")

        port = find_free_port()

        if isinstance(env_config, EnvContext):
            worker_str = f"Worker {env_config.worker_index}"
        else:
            worker_str = "Custom env"
        print(f"{worker_str} - Starting game on port `{port}`")

        self.game_process: subprocess.Popen[bytes] = subprocess.Popen(
            [game_path, "--rl", "--port", str(port)],
        )

        self.context: zmq.Context[zmq.Socket[bytes]] = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        print("Connecting to server…")
        self.socket.connect(f"tcp://localhost:{port}")
        self.reset()
        print("Connected to server")

    def close(self) -> None:
        self.game_process.terminate()

    def __del__(self) -> None:
        print("Closing game")
        self.close()

    @staticmethod
    def trail_color_to_one_hot(trail_color: str) -> BoxSampleType:
        if trail_color.lower() == "red":
            return np.array([1, 0, 0])
        elif trail_color.lower() == "green":
            return np.array([0, 1, 0])
        elif trail_color.lower() == "yellow":
            return np.array([0, 0, 1])
        else:
            raise ValueError("Unknown trail color")

    @staticmethod
    def pad_or_slice_list(lst: list[Any], length: int, padding: Any) -> list[Any]:
        return lst[:length] + [padding] * (length - len(lst))

    def message_to_observation_space_sample(
        self, message: dict[str, Any]
    ) -> ObservationSpaceType:
        ingame_state = message["IngameState"]
        player = ingame_state["Player"]
        paint_buckets = ingame_state["PaintBuckets"]
        DEFAULT_POSITION = {"Position": {"X": 0, "Y": 0}}
        return ObservationSpaceType(
            elapsed_game_time=np.array([message["elapsedGameTime"]]),
            total_game_time=np.array([message["totalGameTime"]]),
            ingame_state=IngameStateType(
                player=PlayerType(
                    position=np.array(
                        [
                            player["Position"]["X"],
                            player["Position"]["Y"],
                        ]
                    ),
                    speed=np.array(
                        [
                            player["Speed"]["_speedX"],
                            player["Speed"]["_speedY"],
                        ]
                    ),
                    speed_boost=SpeedBoostType(
                        speed_boost_multiplier=np.array(
                            [player["SpeedBoost"]["_speedBoostMultiplier"]]
                        ),
                        speed_boost_current_duration=np.array(
                            [player["SpeedBoost"]["_speedBoostCurrentDuration"]]
                        ),
                        speed_boost_duration=np.array(
                            [player["SpeedBoost"]["_speedBoostDuration"]]
                        ),
                        speed_boost_active=np.array(
                            [int(player["SpeedBoost"]["_speedBoostActive"])]
                        ),
                    ),
                    trail_color=self.trail_color_to_one_hot(player["TrailColor"]),
                    score=np.array([player["Score"]]),
                    health=np.clip(np.array([player["Health"]]), 0, 100),
                ),
                enemies=EnemiesType(
                    count=np.array([ingame_state["NumberOfEnemies"]]),
                    positions=np.array(
                        [
                            [
                                np.clip(enemy["Position"]["X"], 0, 800),
                                np.clip(enemy["Position"]["Y"], 0, 800),
                            ]
                            for enemy in self.pad_or_slice_list(
                                ingame_state["Enemies"],
                                self.number_of_closest_enemies,
                                DEFAULT_POSITION,
                            )
                        ]
                    ),
                ),
                paint_buckets=PaintBucketsType(
                    count=np.array([len(paint_buckets)]),
                    positions=np.array(
                        [
                            [
                                np.clip(paint_bucket["Position"]["X"], 0, 800),
                                np.clip(paint_bucket["Position"]["Y"], 0, 800),
                            ]
                            for paint_bucket in self.pad_or_slice_list(
                                paint_buckets,
                                self.number_of_closest_paint_buckets,
                                DEFAULT_POSITION,
                            )
                        ]
                    ),
                    types=np.array(
                        [
                            self.trail_color_to_one_hot(paint_bucket["Color"])
                            for paint_bucket in self.pad_or_slice_list(
                                paint_buckets,
                                self.number_of_closest_paint_buckets,
                                {"Color": "RED"},
                            )
                        ],
                    ),
                ),
                projectiles=ProjectilesType(
                    count=np.array([len(ingame_state["Projectiles"])]),
                    positions=np.array(
                        [
                            [
                                np.clip(projectile["Position"]["X"], 0, 800),
                                np.clip(projectile["Position"]["Y"], 0, 800),
                            ]
                            for projectile in self.pad_or_slice_list(
                                ingame_state["Projectiles"],
                                self.number_of_closest_projectiles,
                                DEFAULT_POSITION,
                            )
                        ]
                    ),
                    angles=np.array(
                        [
                            projectile["Angle"]
                            for projectile in self.pad_or_slice_list(
                                ingame_state["Projectiles"],
                                self.number_of_closest_projectiles,
                                {"Angle": 0},
                            )
                        ]
                    ),
                    speeds=np.array(
                        [
                            projectile["Speed"]
                            for projectile in self.pad_or_slice_list(
                                ingame_state["Projectiles"],
                                self.number_of_closest_projectiles,
                                {"Speed": 0},
                            )
                        ]
                    ),
                    durations=np.array(
                        [
                            projectile["Duration"]
                            for projectile in self.pad_or_slice_list(
                                ingame_state["Projectiles"],
                                self.number_of_closest_projectiles,
                                {"Duration": 0},
                            )
                        ]
                    ),
                    current_durations=np.array(
                        [
                            projectile["CurrentDuration"]
                            for projectile in self.pad_or_slice_list(
                                ingame_state["Projectiles"],
                                self.number_of_closest_projectiles,
                                {"CurrentDuration": 0},
                            )
                        ]
                    ),
                ),
                time_scale_active=np.array([int(ingame_state["TimeScaleActive"])]),
                time_scale=np.array([ingame_state["TimeScale"]]),
                time_scale_duration=np.array([ingame_state["TimeScaleDuration"]]),
            ),
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        return_info: bool = False,
        options: dict[Any, Any] | None = None,
    ) -> ObservationSpaceType | tuple[ObservationSpaceType, dict[Any, Any]]:
        # print(datetime.datetime.now().strftime("%H:%M:%S"), "RESET")
        self.socket.send_json({"command": "RESET", "render_next": self.render_next})
        self.render_next = False
        message = cast(dict[str, Any], self.socket.recv_json())
        observation = self.message_to_observation_space_sample(message)
        self.current_step_count = 0
        if return_info:
            return observation, {}
        else:
            return observation

    def step(
        self, action: ActionSpaceType
    ) -> tuple[ObservationSpaceType, float, bool, dict[Any, Any]]:
        up = int(action["move_vertical"]) < 0  # type: ignore
        down = int(action["move_vertical"]) > 0  # type: ignore
        left = int(action["move_horisontal"]) < 0  # type: ignore
        right = int(action["move_horisontal"]) > 0  # type: ignore
        move = [up, down, left, right]

        action_json = {
            "command": "STEP",
            "render_next": self.render_next,
            "move": move,
            "ability": bool(cast(int, action["ability"])),
        }
        self.render_next = False
        self.socket.send_json(action_json)
        message = cast(dict[str, Any], self.socket.recv_json())
        observation_space_sample = self.message_to_observation_space_sample(message)
        self.current_step_count += 1
        score = (
            message["totalGameTime"] + message["IngameState"]["Player"]["Score"] * 200
        )
        done = message["IngameState"]["Player"]["Health"] <= 0
        if self.current_step_count >= self.max_episode_steps:
            done = True
        return (
            observation_space_sample,
            score,
            done,
            {},
        )

    def render(self, mode: str = "human") -> None:
        if mode == "human":
            self.render_next = True
        else:
            raise NotImplementedError("Only human mode is supported")
