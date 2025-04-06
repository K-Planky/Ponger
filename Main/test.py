import gymnasium
import ale_py

gymnasium.register_envs(ale_py)

env = gymnasium.make(
    "ALE/Pong-v5",
    render_mode="rgb_array",
    repeat_action_probability=0,
    mode=1,
)
env = gymnasium.wrappers.RecordVideo(
    env,
    video_folder="saved-video-folder",
    name_prefix="video-",
)
env.reset()
for _ in range(1000):
    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

env.close()
