![banner](imgs/Ponger.png)

We trained an AI using [NEAT (NeuroEvolution of Augmenting Topologies)](https://github.com/CodeReclaimers/neat-python/tree/master) with [ALE (The Arcade Learning Environment)](https://github.com/Farama-Foundation/Arcade-Learning-Environment) to play **Pong** from Atari 2600.

## ALE
We used [ALE](https://github.com/Farama-Foundation/Arcade-Learning-Environment) as our base, where it only gave us this to work with:
```Python
import gymnasium
import ale_py

gymnasium.register_envs(ale_py)

env = gymnasium.make("ALE/Pong-v5", render_mode="human")
env.reset()
for _ in range(100):
    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

env.close()
```
![ale example](Imgs/ALE.gif)    
The obs only gives us an image frame for each frame of the game, so we have to do image processing to get all the necessary information ourselves, e.g., the ball and paddle location, the ball speed and direction, etc.

## Manually - *Planky's AI*
First, we tried to do it manually without using an AI to play the game. We did this by using math. 