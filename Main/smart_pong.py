import neat.population
import neat
import os
import pickle
import gymnasium
import ale_py
import numpy as np


def get_ball_loc(img):
    _ball_xy = np.where(img[34:194, :] == 236)
    if len(_ball_xy[0]):  # Check if empty
        ball_x = np.mean(_ball_xy[1])  # x
        ball_y = np.mean(_ball_xy[0]) + 34  # y
        return (ball_x, ball_y)
    return (80, 105)


gymnasium.register_envs(ale_py)
env = gymnasium.make(
    "ALE/Pong-v5",
    obs_type="grayscale",
    render_mode="human",
)
env.reset()


def test_ai(genome):
    env.reset()
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    action = 0

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
                abs(139.5 - ball_x),
            )
        )
        decision = output.index(max(output))

        if decision == 0:
            action = 0
        elif decision == 1:
            action = 2
        else:
            action = 3

    env.close()


def train_ai(genome, config):
    env.reset()
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    action = 0

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
                abs(139.5 - ball_x),
            )
        )
        decision = output.index(max(output))

        if decision == 0:
            action = 0
        elif decision == 1:
            action = 2
        else:
            action = 3

        if reward != 0:
            genome.fitness += reward
            break


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 0
        train_ai(genome, config)


def run_neat(config):
    # for loading checkpoint comment out p and then uncomment the line below:
    p = neat.Checkpointer.restore_checkpoint("neat-checkpoint-49")
    # p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1))

    winner = p.run(eval_genomes, 50)  # CHANGE THIS ONE!
    with open("best.pickle", "wb") as f:
        pickle.dump(winner, f)


def test_ai_w_config(config):
    with open("Ponger/best.pickle", "rb") as f:
        winner = pickle.load(f)

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
