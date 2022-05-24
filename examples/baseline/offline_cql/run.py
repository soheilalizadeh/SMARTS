import logging
import pathlib
import d3rlpy
from d3rlpy.dataset import MDPDataset
from d3rlpy.algos import CQL
import numpy as np
from competition_env import CompetitionEnv
from offline_competition_env import OfflineCompetitionEnv
import gym
from ruamel.yaml import YAML
import os
import pandas as pd

from examples.argument_parser import default_argument_parser

logging.basicConfig(level=logging.INFO)
import argparse
from datetime import datetime
yaml = YAML(typ="safe")

def data_process(df_behind, df_front):
    df_process = pd.DataFrame()
    cols = ['x_behind', 'y_behind', 'x_front', 'y_front', 'dx_behind', 'dy_behind', 'dist']
    for i in range(df_behind.shape[0]):
        x_behind = df_behind.iloc[i]['position_x']
        y_behind = df_behind.iloc[i]['position_y']
        x_front = df_front.iloc[i]['position_x']
        y_front = df_front.iloc[i]['position_y']
        dx_behind = df_behind.iloc[i]['delta_x']
        dy_behind = df_behind.iloc[i]['delta_y']
        dist = np.sqrt((x_behind - x_front) ** 2 + (y_behind - y_front) ** 2)
        temp_df = pd.DataFrame([[x_behind, y_behind, x_front, y_front, dx_behind, dy_behind, dist]], columns=cols)
        df_process = pd.concat([df_process, temp_df])
    return df_process



def clean_data(df):
    df_clean = pd.DataFrame()
    i = 0
    while i < df.shape[0] - 1:
        if df.iloc[i]['sim_time'] == df.iloc[i + 1]['sim_time']:
            sub_df = df.iloc[i : i+2, :]
            df_clean = pd.concat([df_clean, sub_df])
            i += 2
        else:
            i += 1
    return df_clean

def get_vehicle_id(df):
    vehicle_behind_id = 'NA'
    vehicle_front_id = 'NA'
    for i in range(df.shape[0]):
        if 'behind' in df.iloc[i]['agent_id']:
            vehicle_behind_id = df.iloc[i]['agent_id']
        else: 
            vehicle_front_id = df.iloc[i]['agent_id']
        if vehicle_behind_id != 'NA' and vehicle_front_id != 'NA':
            break
    return vehicle_behind_id, vehicle_front_id

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
    else:
        raise KeyError(
            f'Expected \'train\' or \'evaluate\', but got {config_env["mode"]}.'
        )
    logdir.mkdir(parents=True, exist_ok=True)
    print("Logdir:", logdir)

    # Run training or evaluation.
    run(config_env, logdir)

def _build_scenario():
    scenario = str(pathlib.Path(__file__).absolute().parent / "scenarios")
    build_scenario = f"scl scenario build-all --clean {scenario}"
    os.system(build_scenario)

def run(config, logdir):
    #scenarios = config["scenarios_dir"]
    max_episode_steps = config['max_episode_steps']
    env = OfflineCompetitionEnv(scenarios)(["scenarios/follow"], max_episode_steps)
    num_epochs = config['num_epochs']
    if config["mode"] == "evaluate":
        print("Start evaluation.")
        model = CQL.from_json('d3rlpy_logs/CQL_20220520122229/params.json')
        model.load_model('d3rlpy_logs/CQL_20220520122229/model_5000.pt')
        print('finish loading model')

        observation=env.reset
        aciton = model.prediction(state)
        env.step(action)
        observation





    else:
        print('Start training using existing dataset')
        df_dataset = pd.DataFrame()
        for filename in os.listdir('recorded_trajectories'):
            df = pd.read_csv('recorded_trajectories/' + filename)
            df = clean_data(df)
            vehicle_behind_id, vehicle_front_id = get_vehicle_id(df)
            df_behind = df.loc[df['agent_id'] == vehicle_behind_id]
            df_front = df.loc[df['agent_id'] == vehicle_front_id]
            df_process = data_process(df_behind, df_front)
            df_dataset = pd.concat([df_dataset, df_process])
        df_dataset.to_csv('dataset.csv')

        observations = df_dataset[['x_behind', 'y_behind', 'x_front', 'y_front']].to_numpy()
        actions = df_dataset[['dx_behind', 'dy_behind']].to_numpy()
        rewards = - df_dataset[['dist']].to_numpy()
        terminals = np.array([0] * df_dataset.shape[0])
        episode_terminals = np.random.randint(2, size=df_dataset.shape[0])
        dataset = MDPDataset(observations, actions, rewards, terminals, episode_terminals)
        cql = d3rlpy.algos.CQL(use_gpu=False)

        cql.fit(dataset, 
                eval_episodes=dataset, 
                n_epochs=num_epochs, 

        )
    
    env.close()


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
        "--num_epochs", help="Number of training epochs.", type=int, default=1000
    )
    parser.add_argument(
        "--max_episode_steps", help="Number of steps in each episode.", type=int, default=300
    )

    args = parser.parse_args()

    if args.mode == "evaluate" and args.logdir is None:
        raise Exception("When --mode=evaluate, --logdir option must be specified.")

    main(args)

