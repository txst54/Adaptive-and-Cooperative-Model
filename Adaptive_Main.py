import pickle
import random
import numpy as np
import gfootball.env as football_env
from gfootball.env.football_action_set import *
from Adaptive_Model import Model

env = football_env.create_environment(env_name="academy_empty_goal_close", stacked=False, logdir='/tmp/football',
                                      write_goal_dumps=False, write_full_episode_dumps=False, render=False)
env.reset()

StartingReplaySize = 50000
InputShape = (72, 96, 4)
TotalStepLimit = 5000000
ActionSpace = [
    action_idle, action_left, action_top_left, action_top,
    action_top_right, action_right, action_bottom_right,
    action_bottom, action_bottom_left, action_long_pass,
    action_high_pass, action_short_pass, action_shot,
    action_sprint, action_release_direction, action_release_sprint,
    action_sliding, action_dribble, action_release_dribble]
print(ActionSpace)


def main():
    Exp_traj = pickle.load(open("exp_traj.pkl", "rb"))
    Exp_traj.pop(0)
    print(type(Exp_traj[0][0]))
    total_step = 0
    game_model = Model(25, len(ActionSpace), Exp_traj)
    prev_score = 0
    prev_action = [0, 0]
    while True:
        env.reset()
        step = 0
        print(total_step)
        # action_env = GetEnv(inDim=InFrameDim, outDim=(1920, 1080))
        # action_frame = action_env.takeImage(1, "none")
        # mouse_action_space = create_action_space(action_frame)
        obs, rew, done, info = env.step(env.action_space.sample())
        while total_step <= TotalStepLimit:
            total_step += 1
            step += 1
            print(f"Step: {total_step}")
            action = game_model.move(obs)
            print(action)
            print(f"Action: {ActionSpace[action]}")
            nobs, rew, done, info = env.step(ActionSpace[action])
            # action_frame = action_env.takeImage(1, "none")
            # mouse_action_space = create_action_space(action_frame)
            print(f"Reward: {rew}")
            game_model.remember(obs, action, rew, nobs)
            # thread = Thread(target=game_model.step_update, args=(total_step,))
            # thread.start()
            game_model.step_update(total_step)
            obs = nobs
            if done:
                break


if __name__ == "__main__":
    main()
