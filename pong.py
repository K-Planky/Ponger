import gymnasium
import ale_py
import numpy as np
from matplotlib import pyplot as plt

gymnasium.register_envs(ale_py)
# https://ale.farama.org/environments/pong/


def get_ball_loc(img):
    _ball_xy = np.where(img[34:194, 20:140] == 236)
    if len(_ball_xy[0]):  # Check if empty
        ball_x = np.mean(_ball_xy[1]) + 20  # x
        ball_y = np.mean(_ball_xy[0]) + 34  # y
        return (ball_x, ball_y)
    return (0, 0)


env = gymnasium.make(
    "ALE/Pong-v5",
    obs_type="grayscale",
    # render_mode="human"
)
env.reset()
score = 0

target_frame = 35
last_ball = (0, 0)

for frame in range(target_frame):  # 15
    action = env.action_space.sample()

    img, reward, terminated, truncated, info = env.step(action)
    score += int(reward)

    if frame > 26:  # frame > 13
        plt.imshow(img)

        paddle_y = np.mean(np.where(img[34:194, 141] == 147)) + 34

        ball = get_ball_loc(img)

        if last_ball[0] == 0 or ball[0] == 0:
            last_ball = ball
            continue

        plt.plot(141.5, paddle_y, "xr")
        plt.plot(last_ball[0], last_ball[1], "xr")

        print(ball, last_ball, paddle_y)

        displacement = np.subtract(ball, last_ball)
        print(displacement)

        plt.plot(ball[0] + displacement[0], ball[1] + displacement[1], "xg")
        # Top bottom border Y
        # plt.axhline(34, color="red")
        # plt.axhline(193, color="red")

        # Left right border X
        # plt.axvline(20, color="red")
        # plt.axvline(139, color="red")

        last_ball = ball
        plt.show()

    if terminated or truncated:
        obs, info = env.reset()

print(score)
env.close()
