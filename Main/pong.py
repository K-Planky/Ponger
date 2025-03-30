import gymnasium
import ale_py
import numpy as np
from matplotlib import pyplot as plt

gymnasium.register_envs(ale_py)
# https://ale.farama.org/environments/pong/


def get_ball_loc(img):
    _ball_xy = np.where(img[40:188, 23:137] == 236)  # 23:137
    if len(_ball_xy[0]):  # Check if empty
        ball_x = np.mean(_ball_xy[1]) + 23  # x
        ball_y = np.mean(_ball_xy[0]) + 40  # y
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
    if action == 0:
        return 0
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


env = gymnasium.make("ALE/Pong-v5", obs_type="grayscale", render_mode="human", mode=1)
env.reset()

win = 0
lose = 0
last_ball = (0, 0)
last_paddle = 113.5
displacement = (-1, 0)
predict = 113.5
paddle = 113.5
last_action = 0
gave_reward = True

for frame in range(10000):
    # if displacement[0] < 0:
    #     predict = 113.5

    if predict - paddle < 0:
        action = 2
    else:
        action = 3

    if abs(predict - paddle) < 9:
        action = 0

    if last_action == action and abs(predict - paddle) < 80:
        action = 0

    img, reward, terminated, truncated, info = env.step(action)
    if int(reward) > 0:
        win += 1
    elif int(reward) < 0:
        lose += 1

    paddle = np.mean(np.where(img[34:194, 141] == 147)) + 34
    ball = get_ball_loc(img)

    if not gave_reward and last_ball[0] > ball[0]:
        gave_reward = True
        print("GOT REWARD!", last_ball[0], ball[0])
    elif last_ball[0] < ball[0]:
        gave_reward = False

    if last_ball[0] == 0 or ball[0] == 0:
        last_ball = ball
        continue

    displacement = np.subtract(ball, last_ball)
    frame_to_paddle = (139.5 - ball[0]) / displacement[0]  # 139.5
    predict = cal_y_bounce(frame_to_paddle * displacement[1] + ball[1], displacement[1])
    speed = last_paddle - paddle

    # print(predict, paddle, predict - paddle, speed, convert_action(action), "\n")

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

    last_action = action
    last_ball = ball
    last_paddle = paddle

    if terminated or truncated:
        img, info = env.reset()

print(win, lose)
env.close()
