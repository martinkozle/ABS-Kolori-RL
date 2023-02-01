# ABS Kolori RL

This is a project for the subject "Agent Based Systems" at Faculty of Computer Science and Engineering Skopje.
Made by [Martin Popovski](https://github.com/martinkozle) and [Tomislav Ignjatov](https://github.com/AnixDrone).

## Gameplay by a human player

To get a feel for the game and the mechanics, we recorded a gameplay video of a human player (YouTube link):

[![YouTube thumbnail](http://img.youtube.com/vi/i_DXAveNxDE/0.jpg)](http://www.youtube.com/watch?v=i_DXAveNxDE "Kolori RL Showcase - Gameplay")

## How the game was simplified

For more technical details on how the game was changed, see the [Kolori RL](https://github.com/martinkozle/Kolori-RL) repository.

Gameplay was simplified by removing the purple (teleport anywhere on screen) and blue (shoot in a straight line) paint buckets because they required the use of the mouse to interact with the game. We wanted the agent to be able to play the game without the use of the mouse and hence have a lot smaller action space.

## Observation space

The observation space contains information about the game state, the player, the enemies, the paint buckets, the projectiles and the time scale ability.
The observation space has a shape of `86`.

```py
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
)
```

## Action space

By removing the mouse interaction, the action space was reduced to a shape of 3.
With a total of 3 possible actions for vertical movement, 3 possible actions for horizontal movement and 2 possible actions for the ability, the total number of possible actions is `3 * 3 * 2 = 18`.

```py
spaces.Dict(
    move_horisontal=spaces.Discrete(3, start=-1),
    move_vertical=spaces.Discrete(3, start=-1),
    ability=spaces.Discrete(2),
)
```

## Framework used for reinforcement learning

The framework used for reinforcement learning is [RLlib](https://docs.ray.io/en/latest/rllib/index.html).

## Algorithm used for reinforcement learning

The algorithm used for reinforcement learning is [PPO](https://docs.ray.io/en/latest/rllib-algorithms.html#proximal-policy-optimization-ppo).

We used torch as the backend for RLlib.

For the hyperparameters, we used a fully connected network with 6 layers, the number of neurons in each layer was 512, 512, 256, 128, 64, 32.

## Reward function

The game itself has a Score system, which is based on the number of enemies killed with your abilities. But we thought that this was not good for the agent to learn, because the agent wouldn't be able to learn the beginning of the game effectively because it would have to wait for the enemies to spawn. So we decided to use a reward function that would also reward the agent for surviving for longer. This way the agent will have a linear reward earlier in the game.

The reward function is as follows:

```text
score = total_game_time + ingame_score * 200
```

## Gameplay by an untrained actor

The untrained actor got a mean reward of `~266`.

It's downfall was using the ability too ofter, which made it lose all of its health at the beginning of the game.

[![YouTube thumbnail](http://img.youtube.com/vi/TY6_URXgjXI/0.jpg)](http://www.youtube.com/watch?v=TY6_URXgjXI "Kolori RL Showcase - Episode 0 (No Training)")

## Actor trained for 1000 episodes

This training was performed on a machine with an `AMD Ryzen 7 5800X3D` processor, `NVIDIA GeForce RTX 3070` graphics card and 32 GB of RAM.

8 instances of the game were run in parallel with unbound update rates for training, and another 8 instances for evaluation for each episode. In total, 16 instances of the game were running at the same time. The training took ~36 hours to complete.

This actor got a mean reward of `~5402`.

While it is a significant improvement over the untrained actor if looking at the reward alone, if you look at the gameplay, it is clear that the agent is still not very good at the game.

The only real reason it doesn't die so quickly as the untrained actor is because it learned to use the ability a little less often. But this is hardly an improvement when not using the ability at all would give it an even better reward.

### Gameplay

[![YouTube thumbnail](http://img.youtube.com/vi/6y-6Yh_ubjs/0.jpg)](http://www.youtube.com/watch?v=6y-6Yh_ubjs "Kolori RL Showcase - Episode 1000")

## Some interesting technical details

- For communication between the game instances and the training instance we chose [ZeroMQ](https://zeromq.org/), a universal messaging library
  - For the Python implementation we used [pyzmq](https://pyzmq.readthedocs.io/en/latest/)
  - For the C# implementation we used [NetMQ](https://netmq.readthedocs.io/en/latest/)
  - We used TCP as the transport protocol
  - We used the [REQ-REP](https://learning-0mq-with-pyzmq.readthedocs.io/en/latest/pyzmq/patterns/client_server.html) pattern for communication. The training instance was the client and the game instances were the servers. The training instance would send a request to the game instance with the action to perform, and the game instance would send a reply with the observation after the action was performed. This pattern ended up being a good fit for our use case.
  - For port selection we used an uncertain solution where the training instance would request a free port from the host machine and then create a ZMQ socket on that port. This will not work 100% of the time because a race condition can occur where the port is taken by another process before the game instance can create a socket on it. But that would be a very rare occurrence that could only happen at the start of the training.
- The code is fully typed and uses [mypy](http://mypy-lang.org/) (strict) for static type checking.
  - RLlib wasn't designed with typed gym environments in mind, so we had to make some workarounds and use castings to make it work.
- Poetry was used for dependency management
  - Installing torch with GPU support using poetry is a little bit difficult and updating the lock file for the first time takes about ~15 minutes. This is because poetry downloads and tries every wheel file for the specified version of torch and cuda in order to find one for the specific platform.

## Development

Developed and tested on Fedora 37 with Python 3.10.9.

### Dependencies

Use Poetry to install the dependencies.

```bash
poetry install
```

### Kolori RL Game

The modified version of the game is available [here](https://github.com/martinkozle/Kolori-RL). A build of the game for Linux is available in the Releases section.

### Training

Run the module `kolori_rl` to start the training.

```bash
KOLORI_GAME_PATH={PATH_TO_YOUR_GAME_BINARY} python -m abs_kolori_rl; killall GJP2021
```

The game instances sometimes don't close properly, so we add a `killall GJP2021` at the end to make sure they are closed.

### Showcase

To showcase the trained actor, run the same module `kolori_rl` but with a `showcase` argument.

```bash
KOLORI_GAME_PATH={PATH_TO_YOUR_GAME_BINARY} python -m abs_kolori_rl showcase; killall GJP2021
```

RLlib for some reason will start multiple instances of the game and only the last one will showcase the game.
