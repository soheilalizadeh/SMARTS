import logging
import pathlib
import d3rlpy
from d3rlpy.dataset import MDPDataset
from d3rlpy.algos import CQL, BCQ
import numpy as np
from competition_env import CompetitionEnv
import gym
from ruamel.yaml import YAML
import os
import glob
import pandas as pd

from examples.argument_parser import default_argument_parser

logging.basicConfig(level=logging.INFO)
import argparse
from datetime import datetime
yaml = YAML(typ="safe")

import csv
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.controllers import ControllerOutOfLaneException
from smarts.core.scenario import Scenario
from smarts.core.sensors import Observation
from smarts.core.smarts import SMARTS
from smarts.core.sumo_traffic_simulation import SumoTrafficSimulation
from smarts.sstudio import build_scenario
from smarts.zoo.agent_spec import AgentSpec

logging.basicConfig(level=logging.INFO)
import pathlib
import gym

from examples.argument_parser import default_argument_parser
from smarts.core.agent import Agent
from smarts.core.utils.episodes import episodes
from smarts.env.wrappers.single_agent import SingleAgent

from d3rlpy.algos import CQL
import numpy as np
from competition_env import CompetitionEnv
from smarts.sstudio import build_scenario
import shutil


def data_process():
    df_dataset = pd.DataFrame()
    cols = ['x', 'y', 'dx', 'dy', 'rewards', 'terminals']
    for filename in np.sort(os.listdir('recorded_trajectories')):
        print(filename)
        df = pd.read_csv('recorded_trajectories/' + filename)
        for i in range(1, df.shape[0]):
            x = df.iloc[i]['position_x']
            y = df.iloc[i]['position_y']
            dx = df.iloc[i]['delta_x']
            dy = df.iloc[i]['delta_y']
            if df.iloc[i]['dones']:
                reward = 10
                terminal = 1
            else:
                reward = df.iloc[i]['rewards']
                terminal = 0
            temp_df = pd.DataFrame([[x, y, dx, dy, reward, terminal]], columns=cols)
            df_dataset = pd.concat([df_dataset, temp_df])
    return df_dataset



def new_data_process():
    df_dataset = pd.DataFrame()
    cols = ['x', 'y', 'dx', 'dy', 'rewards', 'terminals']
    for filename in np.sort(os.listdir('recorded_trajectories_new')):
        print(filename)
        df = pd.read_csv('recorded_trajectories_new/' + filename)
        for i in range(1, df.shape[0]):
            x = df.iloc[i]['position_x']
            y = df.iloc[i]['position_y']
            dx = df.iloc[i]['delta_x']
            dy = df.iloc[i]['delta_y']
            if df.iloc[i]['dones']:
                reward = -5
                terminal = 1
            else:
                reward = df.iloc[i]['rewards']
                terminal = 0
            temp_df = pd.DataFrame([[x, y, dx, dy, reward, terminal]], columns=cols)
            df_dataset = pd.concat([df_dataset, temp_df])
    return df_dataset


def _record_data(
    collected_data: List[List[Any]],
    obs,
    prev_obs,
    reward,
    done
):
 
    curr_state = obs.ego['pos']

    dx, dy = None, None
    agent_id = 'single'
    event = obs.events
    prev_s = prev_obs.ego['pos']
    dx = curr_state[0] - prev_s[0]
    dy = curr_state[1] - prev_s[1]

    row = [
        agent_id,
        curr_state[0],
        curr_state[1],
        dx,
        dy,
        event,
        reward,
        done

    ]
    collected_data.append(row)


def main(args: argparse.Namespace):
    # Load config file.
    config_file = yaml.load(
        (pathlib.Path(__file__).absolute().parent / "config.yaml").read_text()
    )

    # Load env config.
    config_env = config_file["smarts"]
    config_env["mode"] = args.mode
    config_env["headless"] = not args.head
    config_env["num_epochs"] = args.num_epochs
    config_env["max_episode_steps"] = args.num_epochs
    """
    config_env["scenarios_dir"] = (
        pathlib.Path(__file__).absolute().parents[0] / "scenarios"
    )
    
    config_env["max_episode_steps"] = (
        pathlib.Path(__file__).absolute().parents[0] / "max_episode_steps"
    )
    """
    #_build_scenario()

    # Train or evaluate.
    if config_env["mode"] == "train" and not args.logdir:
        # Begin training from scratch.
        time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        logdir = pathlib.Path(__file__).absolute().parents[0] / "logs" / time
    elif (config_env["mode"] == "train" and args.logdir) or (
        config_env["mode"] == "evaluate"
    ):
        # Begin training from a pretrained model.
        logdir = pathlib.Path(args.logdir)
    elif config_env["mode"] == "keep_training":
        time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        logdir = pathlib.Path(__file__).absolute().parents[0] / "logs" / time
    else:
            raise KeyError(
                f'Expected \'train\' or \'evaluate\', but got {config_env["mode"]}.'
            )
    logdir.mkdir(parents=True, exist_ok=True)
    print("Logdir:", logdir)

    # Run training or evaluation.
    run(config_env, logdir)

def _build_scenario():
    scenario = str(pathlib.Path(__file__).absolute().parent / "evaluate_scenarios/cruise")
    build_scenario = f"scl scenario build-all --clean {scenario}"
    os.system(build_scenario)


def run(config, logdir):
    num_keep_training = 20
    num_episodes = 50
    #scenarios = config["scenarios_dir"]
    max_episode_steps = config['max_episode_steps']
    num_epochs = config['num_epochs']
    # models = os.listdir('d3rlpy_logs')
    for k in range(num_keep_training):
        
        if k == 0:
            model = CQL.from_json('d3rlpy_logs/baseline/params.json')
            model.load_model('d3rlpy_logs/baseline/model.pt')
        else:
            print('removing and adding new trajectories')
            shutil.rmtree('recorded_trajectories_new')    
            imported_model = os.listdir('d3rlpy_logs')[-1]
            model = CQL.from_json('d3rlpy_logs/' + imported_model + '/params.json')
            model.load_model('d3rlpy_logs/' + imported_model + '/model.pt')
        
        scenario_name = "scenarios/cruise"
        scenarios_iterator = Scenario.scenario_variations([scenario_name], [])
        for episode in range(num_episodes):
            scenario = next(scenarios_iterator)
            env = CompetitionEnv(["scenarios/cruise"], max_episode_steps=300, headless=False)
            _build_scenario()
            obs = env.reset()
            prev_obs = obs
            collected_data = []
            _record_data(collected_data, obs, prev_obs, None, None)

            done = False
            while not done:
                ego_pos_x = obs.ego['pos'][0]
                ego_pos_y = obs.ego['pos'][1]
                state = np.array([ego_pos_x, ego_pos_y])
                action = model.predict([state])[0]
                obs, reward, done, info = env.step(action)
                _record_data(collected_data, obs, prev_obs, reward, done)
                prev_obs = obs
            observation_folder = "recorded_trajectories_new"
            if not os.path.exists(observation_folder):
                os.makedirs(observation_folder)

            csv_filename = Path(observation_folder) / f"{scenario.name}-{episode}.csv"
            with open(csv_filename, "w", newline="") as f:
                csv_writer = csv.writer(f, delimiter=",")
                header = [
                        "agent_id",
                        "position_x",
                        "position_y",
                        "delta_x",
                        "delta_y",
                        'events',
                        'rewards',
                        'dones'
                    ]
                csv_writer.writerow(header)

                for row in collected_data:
                    csv_writer.writerow(row)

        env.close()

        df_dataset = pd.read_csv('dataset.csv')
        df_dataset_new = new_data_process()
        df_dataset = pd.concat([df_dataset, df_dataset_new])
        df_dataset.to_csv('new_dataset.csv')
        print('finish processing data')
        observations = df_dataset[['x', 'y']].to_numpy()
        actions = df_dataset[['dx', 'dy']].to_numpy()
        rewards = df_dataset[['rewards']].to_numpy()
        terminals = df_dataset[['terminals']].to_numpy()
        dataset = MDPDataset(observations, actions, rewards, terminals)
        model.fit(dataset, 
                eval_episodes=dataset, 
                n_epochs=num_epochs, 

                )
        imported_model = os.listdir('d3rlpy_logs')[-1]
        model.save_model('d3rlpy_logs/' + imported_model + '/model.pt')


if __name__ == "__main__":
    program = pathlib.Path(__file__).stem
    parser = argparse.ArgumentParser(program)
    parser.add_argument(
        "--mode",
        help="`train` or `evaluate`. Default is `train`.",
        type=str,
        default="train",
    )
    parser.add_argument(
        "--logdir",
        help="Directory path to saved RL model. Required if `--mode=evaluate`, else optional.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--head", help="Run the simulation with display.", action="store_true"
    )
    parser.add_argument(
        "--num_epochs", help="Number of training epochs.", type=int, default=200
    )
    parser.add_argument(
        "--max_episode_steps", help="Number of steps in each episode.", type=int, default=300
    )

    args = parser.parse_args()

    if args.mode == "evaluate" and args.logdir is None:
        raise Exception("When --mode=evaluate, --logdir option must be specified.")

    main(args)

