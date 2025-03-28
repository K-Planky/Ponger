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


def cal_y_bounce(y):
    if y < 34:
        return abs(34 - y) + 34
    elif y > 193:
        return 193 - abs(y - 193)
    return y


env = gymnasium.make(
    "ALE/Pong-v5",
    obs_type="grayscale",
    render_mode="human",
)
env.reset()
score = 0

target_frame = 35
last_ball = (0, 0)
ball = (0, 0)
paddle_y = 113.5
predict = 113.5

for frame in range(10000):  # 15
    # action = env.action_space.sample()

    print(abs(predict - paddle_y))
    if predict - paddle_y < 0:
        action = 2
    else:
        action = 3

    if abs(predict - paddle_y) < 11:
        action = 0

    img, reward, terminated, truncated, info = env.step(action)
    score += int(reward)

    if True:  # frame > 13
        plt.imshow(img)

        paddle_y = np.mean(np.where(img[34:194, 141] == 147)) + 34

        ball = get_ball_loc(img)

        if last_ball[0] == 0 or ball[0] == 0:
            last_ball = ball
            continue

        plt.plot(141.5, paddle_y, "xr")
        plt.plot(last_ball[0], last_ball[1], "xr")

        # print(ball, last_ball, paddle_y)

        displacement = np.subtract(ball, last_ball)
        # print(displacement)

        plt.plot(ball[0] + displacement[0] * 3, ball[1] + displacement[1] * 3, "xg")

        x_dis_to_paddle = (139 - ball[0]) / displacement[0]
        predict = cal_y_bounce(x_dis_to_paddle * displacement[1] + ball[1])

        # print(x_dis_to_paddle, predict)

        plt.plot(141.5, predict, "xb")

        # Top bottom border Y
        # plt.axhline(34, color="red")
        # plt.axhline(193, color="red")

        # Left right border X
        # plt.axvline(20, color="red")
        # plt.axvline(139, color="red")

        last_ball = ball
        # plt.show()

    if terminated or truncated:
        obs, info = env.reset()

print(score)
env.close()
