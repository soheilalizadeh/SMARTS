from baselines.competition.competition_env import CompetitionEnv
from smarts.core.agent import Agent
import pathlib
import os


def act(obs, **conf):
    return (1, 1)


agent = Agent.from_function(agent_function=act)

def _build_scenario():
    scenario = str(pathlib.Path(__file__).absolute().parent / "scenarios")
    build_scenario = f"scl scenario build-all --clean {scenario}"
    os.system(build_scenario)


def main(max_steps):
<<<<<<< HEAD
<<<<<<< HEAD
    env = CompetitionEnv(
        scenarios=["scenarios/straight"],
        max_episode_steps=max_steps,
        recorded_obs_path=Path(__file__).parent,
    )
    _build_scenario()
=======
    env = CompetitionEnv(scenarios=["scenarios/straight"], max_episode_steps=max_steps)
=======
    env = CompetitionEnv(scenarios=["scenarios/loop"], max_episode_steps=max_steps)
>>>>>>> 4d3ccd3e91bd607554082df6f25d2a5a7d83339f

>>>>>>> 4207aacc7d00ae98afdbbdd5e87e8ce0bff7550b
    obs = env.reset()
    done = False

    while not done:
<<<<<<< HEAD
        #obs["target_position"] = [0.4, 0, 0]  # can manually modify the observation here
=======
>>>>>>> 4d3ccd3e91bd607554082df6f25d2a5a7d83339f
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)
        # smarts_obs = info["env_obs"] # full observations for debugging if needed
        score = info["scores"]  # TODO: metrics for environment score

    env.close()


if __name__ == "__main__":
    main(max_steps=10)
