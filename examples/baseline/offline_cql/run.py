import logging
import pathlib
import d3rlpy
from d3rlpy.dataset import MDPDataset
from d3rlpy.algos import CQL
import numpy as np
from competition_env import CompetitionEnv
import gym
from ruamel.yaml import YAML
import os

from examples.argument_parser import default_argument_parser

logging.basicConfig(level=logging.INFO)
import argparse
from datetime import datetime
yaml = YAML(typ="safe")



def main(args: argparse.Namespace):
    # Load config file.
    print('i am')
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
    _build_scenario()

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
    env = CompetitionEnv(["scenarios/follow"], max_episode_steps)
    num_epochs = config['num_epochs']
    if config["mode"] == "evaluate":
        print("Start evaluation.")
        print(logdir)
        model = CQL.from_json(logdir/"d3rlpy_log/param.json")
        model.load_model(logdir/"d3rlpy_log/model.pt")
    else:
        
        # generating data from smarts for training
        env.action_space = gym.spaces.Box(
            low=-1e6, high=1e6, shape=(2,), dtype=np.float32
        )   

        observations = []
        actions = []
        rewards = []
        terminals = []
        timeouts = []
        infos = []

        for episode in episodes(n=num_episodes):
            agent = Agent.from_function(agent_function=act)
            observation = env.reset()
            episode.record_scenario(env.scenario_log)

            done = False
            while not done:
                agent_action = agent.act(observation)
                observation, reward, done, info = env.step(agent_action)
                episode.record_step(observation, reward, done, info)
                
                # collect data
                observations.append(observation)
                actions.append(agent_action)
                rewards.append(reward)
                terminals.append(done)
                timeouts.append(done)
                infos.append(info)

        # create dataset
        dataset = MDPDataset(np.array(observations), np.array(actions), np.array(rewards), np.array(terminals))
        if config["mode"] == "train" and args.logdir:
            print("Start training from existing model.")
            model = CQL.from_json(logdir/"d3rlpy_log/param.json")
            model.load_model(logdir/"d3rlpy_log/model.pt")
            model.fit(dataset, 
            eval_episodes=dataset, 
            n_epochs=num_epochs, 
            scorers={
                'environment': d3rlpy.metrics.evaluate_on_environment(env),
                'td_error': d3rlpy.metrics.td_error_scorer
            }
            )
            model.save_model(logdir / "model.pt")
        
        else:
            print("Start training from beginning")
            model = CQL()
            model.fit(dataset, 
            eval_episodes=dataset, 
            n_epochs=num_epochs, 
            scorers={
                'environment': d3rlpy.metrics.evaluate_on_environment(env),
                'td_error': d3rlpy.metrics.td_error_scorer
            }
            )
            model.save_model("model.pt")
    
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
        "--num_epochs", help="Number of training epochs.", type=int, default=1e3
    )
    parser.add_argument(
        "--max_episode_steps", help="Number of steps in each episode.", type=int, default=300
    )

    args = parser.parse_args()

    if args.mode == "evaluate" and args.logdir is None:
        raise Exception("When --mode=evaluate, --logdir option must be specified.")

    main(args)

