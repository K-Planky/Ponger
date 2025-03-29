import gymnasium
import ale_py
import numpy as np
from matplotlib import pyplot as plt

gymnasium.register_envs(ale_py)
# https://ale.farama.org/environments/pong/


def get_ball_loc(img):
    _ball_xy = np.where(img[39:189, 23:137] == 236)
    if len(_ball_xy[0]):  # Check if empty
        ball_x = np.mean(_ball_xy[1]) + 23  # x
        ball_y = np.mean(_ball_xy[0]) + 39  # y
        return (ball_x, ball_y)
    return (0, 0)


def cal_y_bounce(y, dis_y):
    if y < 33.5:
        if dis_y == -8:
            return abs(33.5 - y) + 33.5 - 3
        elif dis_y == -4:
            return abs(33.5 - y) + 33.5 - 1
        elif dis_y == -12:
            return abs(33.5 - y) + 33.5 - 1
        return abs(33.5 - y) + 33.5

    elif y > 193.5:
        if dis_y == 8:
            return 193.5 - abs(y - 193.5) + 1
        elif dis_y == 4:
            return 193.5 - abs(y - 193.5) - 1
        elif dis_y == 12:
            return 193.5 - abs(y - 193.5) + 1
        return 193.5 - abs(y - 193.5)
    return y


def opposite_action(action):
    if action == 2:
        return 3
    return 2


def convert_action(action):
    if action == 2:
        return "up"
    elif action == 3:
        return "down"
    elif action == 0:
        return


env = gymnasium.make(
    "ALE/Pong-v5",
    obs_type="grayscale",
    render_mode="human",
)
env.reset()

score = 0
last_ball = (0, 0)
last_paddle = 113.5
img, reward, terminated, truncated, info = env.step(0)

for frame in range(10000):
    paddle = np.mean(np.where(img[34:194, 141] == 147)) + 34
    ball = get_ball_loc(img)

    if last_ball[0] == 0 or ball[0] == 0:
        last_ball = ball
        img, reward, terminated, truncated, info = env.step(0)
        continue

    displacement = np.subtract(ball, last_ball)
    frame_to_paddle = (139.5 - ball[0]) / displacement[0]
    predict = cal_y_bounce(frame_to_paddle * displacement[1] + ball[1], displacement[1])
    speed = last_paddle - paddle

    if displacement[0] < 0:
        predict = 113.5

    if predict - paddle < 0:
        action = 2
    else:
        action = 3

    if abs(speed) > 15 and abs(predict - paddle) < 14:
        action = opposite_action(action)

    if abs(predict - paddle) < 8 and abs(speed) < 10:
        action = 0

    print(predict, paddle, predict - paddle, speed, convert_action(action), "\n")

    if False:  # frame > 40
        plt.imshow(img)
        plt.plot(141.5, paddle, "xr")
        plt.plot(last_ball[0], last_ball[1], "xr")
        plt.plot(ball[0] + displacement[0] * 3, ball[1] + displacement[1] * 3, "xg")
        plt.plot(frame_to_paddle * displacement[0] + ball[0], predict, "xb")

        # Top bottom border Y
        # plt.axhline(34, color="red")
        # plt.axhline(193, color="red")

        # Left right border X
        # plt.axvline(20, color="red")
        # plt.axvline(139, color="red")

        plt.show()

    img, reward, terminated, truncated, info = env.step(action)
    score += int(reward)

    last_ball = ball
    last_paddle = paddle

    if terminated or truncated:
        obs, info = env.reset()

print(score)
env.close()
