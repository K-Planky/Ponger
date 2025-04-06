import neat.population
import neat
import os
import pickle
import gymnasium
import ale_py
import numpy as np

gymnasium.register_envs(ale_py)

env = gymnasium.make(
    "ALE/Pong-v5",
    obs_type="grayscale",
    repeat_action_probability=0,
    mode=1,
)  # no need to change
env.reset()


def get_ball_loc(img):
    _ball_xy = np.where(img[34:194, :] == 236)
    if len(_ball_xy[0]):  # Check if empty
        ball_x = np.mean(_ball_xy[1])  # x
        ball_y = np.mean(_ball_xy[0]) + 34  # y
        return (ball_x, ball_y)
    return (0, 0)


def test_ai(genome):
    env = gymnasium.make(
        "ALE/Pong-v5",
        obs_type="grayscale",
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
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    action = 0

    last_y = 113.5
    last_paddle = 113.5
    last_action = 0

    run = True
    while run:
        img, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            img, info = env.reset()
            run = False
            break

        right_paddle = np.mean(np.where(img[34:194, 141] == 147)) + 34
        ball_x, ball_y = get_ball_loc(img)

        output = net.activate(
            (
                right_paddle,
                ball_y,
                139.5 - ball_x,
                ball_y - last_y,
                right_paddle - last_paddle,
                last_action,
            )
        )
        decision = output.index(max(output))

        if decision == 0:
            action = 0
            last_action = 0
        elif decision == 1:
            action = 2
            last_action = -1
        else:
            action = 3
            last_action = 1

        last_y = ball_y
        last_paddle = right_paddle

    env.close()


def train_ai(genome, config):
    env.reset()
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    action = 0

    last_x = 0
    last_y = 113.5
    last_paddle = 113.5
    gave_reward = True
    hit_counter = 0
    last_action = 0
    pos_hit = 113.5

    run = True
    while run:
        img, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            img, info = env.reset()

        right_paddle = np.mean(np.where(img[34:194, 141] == 147)) + 34
        ball_x, ball_y = get_ball_loc(img)

        output = net.activate(
            (
                right_paddle,
                ball_y,
                139.5 - ball_x,
                ball_y - last_y,
                right_paddle - last_paddle,
                last_action,
            )
        )
        decision = output.index(max(output))

        if decision == 0:
            action = 0
            last_action = 0
        elif decision == 1:
            action = 2
            last_action = -1
        else:
            action = 3
            last_action = 1

        if ball_x != 0 and last_x != 0:
            if not gave_reward and last_x - ball_x > 0:
                # if action != 0:
                #     genome.fitness += 1
                if pos_hit != right_paddle:
                    genome.fitness += 1
                pos_hit = right_paddle
                hit_counter += 1
                gave_reward = True
            elif last_x - ball_x < 0:
                gave_reward = False

        last_x = ball_x
        last_y = ball_y
        last_paddle = right_paddle

        if int(reward) < 0 or hit_counter >= 50:
            break


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 0
        train_ai(genome, config)


def run_neat(config):
    # for loading checkpoint comment out p and then uncomment the line below:
    # p = neat.Checkpointer.restore_checkpoint("neat-checkpoint-0")
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(10))

    winner = p.run(eval_genomes, 101)  # CHANGE THIS ONE!
    with open("best.pickle", "wb") as f:
        pickle.dump(winner, f)


def test_ai_w_config(config):
    with open("Ponger/Main/gen save/best-mode1.pickle", "rb") as f:
        winner = pickle.load(f)  # CHANGE THIS ONE!

    test_ai(winner)


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )
    # run_neat(config)
    test_ai_w_config(config)
