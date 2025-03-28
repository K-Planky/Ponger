import gymnasium
import ale_py
import numpy as np
from matplotlib import pyplot as plt

gymnasium.register_envs(ale_py)
# https://ale.farama.org/environments/pong/


env = gymnasium.make(
    "ALE/Pong-v5",
    obs_type="grayscale",
    # render_mode="human"
)


env.reset()
score = 0
for frame in range(15):  # 15
    action = env.action_space.sample()

    img, reward, terminated, truncated, info = env.step(action)
    score += int(reward)

    if frame > 13:  # frame > 13
        plt.imshow(img)

        paddle_y = np.mean(np.where(img[34:194, 141] == 147)) + 34
        plt.plot(141, paddle_y, "xr")

        _ball_xy = np.where(img[34:194, 20:140] == 236)
        ball_x = np.mean(_ball_xy[1]) + 20  # x
        ball_y = np.mean(_ball_xy[0]) + 34  # y
        plt.plot(ball_x, ball_y, "xr")

        # Top bottom border Y
        # plt.axhline(34, color="red")
        # plt.axhline(193, color="red")

        # Left right border X
        # plt.axvline(20, color="red")
        # plt.axvline(139, color="red")

        plt.show()

    if terminated or truncated:
        obs, info = env.reset()

print(score)
env.close()
