from typing import Any, Dict

import gym
from sb3.env import action, reward
from stable_baselines3.common import monitor
from stable_baselines3.common.env_checker import check_env
from competition_env import CompetitionEnv

import smarts.env.wrappers.rgb_image as smarts_rgb_image
import smarts.env.wrappers.single_agent as smarts_single_agent
from smarts.core import agent as smarts_agent
from smarts.core import agent_interface as smarts_agent_interface
from smarts.core import controllers as smarts_controllers
from smarts.env import hiway_env as smarts_hiway_env 


def make_env() -> gym.Env:

    env = CompetitionEnv(scenarios=["scenarios/loop"], max_episode_steps=100)

    # Wrap env with ActionWrapper
    env = action.Action(env=env)
    # Wrap env with RewardWrapper
    env = reward.Reward(env=env)
    # Wrap env with RGBImage wrapper to only get rgb images in observation
    env = smarts_rgb_image.RGBImage(env=env, num_stack=1)
    # Wrap env with SingleAgent wrapper to be Gym compliant
    env = smarts_single_agent.SingleAgent(env=env)
    env = monitor.Monitor(env=env)
    check_env(env, warn=True)

    return env
