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






def _record_data(
    collected_data: List[List[Any]],
    t: float,
    obs,
    prev_obs,
    reward,
    done
):
 
    curr_state = obs.ego_vehicle_state

    dx, dy = None, None
    agent_id = 'single'
    event = obs.events
    prev_s = prev_obs.ego_vehicle_state
    dx = curr_state.position[0] - prev_s.position[0]
    dy = curr_state.position[1] - prev_s.position[1]

    row = [
        t,
        agent_id,
        curr_state.position[0],
        curr_state.position[1],
        dx,
        dy,
        curr_state.speed,
        curr_state.heading,
        curr_state.lane_id,
        curr_state.lane_index,
        event,
        reward,
        done

    ]
    collected_data.append(row)


def run(episodes: int, max_steps: int):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    agent_spec = AgentSpec(
        interface=AgentInterface.from_type(AgentType.Laner, max_episode_steps=None),
        agent_builder=None,
    )

    scenario_name = "scenarios/cruise"
    scenarios_iterator = Scenario.scenario_variations([scenario_name], [])
    for episode in range(episodes):
        build_scenario([scenario_name])
        scenario = next(scenarios_iterator)
        obs = smarts.reset(scenario)
        prev_obs = obs

        collected_data = []
        _record_data(collected_data, smarts.elapsed_sim_time, obs, None)

        # could also include "motorcycle" or "truck" in this set if desired
        vehicle_types = frozenset({"car"})

        # filter off-road vehicles from observations
        vehicles_off_road = set()

        while smarts.step_count < max_steps:
            smarts.step({})
            current_vehicles = smarts.vehicle_index.social_vehicle_ids(
                vehicle_types=vehicle_types
            )

            if collected_data and not current_vehicles:
                print("no more vehicles.  exiting...")
                break

            for veh_id in current_vehicles:
                try:
                    smarts.attach_sensors_to_vehicles(agent_spec.interface, {veh_id})
                except ControllerOutOfLaneException:
                    logger.warning(f"{veh_id} out of lane, skipped attaching sensors")
                    vehicles_off_road.add(veh_id)

            valid_vehicles = {v for v in current_vehicles if v not in vehicles_off_road}
            obs, _, _, dones = smarts.observe_from(list(valid_vehicles))
            _record_data(collected_data, smarts.elapsed_sim_time, obs, prev_obs)
            prev_obs = obs

        # an example of how we might save the data per car
        observation_folder = "recorded_trajectories"
        if not os.path.exists(observation_folder):
            os.makedirs(observation_folder)

        csv_filename = Path(observation_folder) / f"{scenario.name}-{episode}.csv"
        with open(csv_filename, "w", newline="") as f:
            csv_writer = csv.writer(f, delimiter=",")
            header = [
                "sim_time",
                "agent_id",
                "position_x",
                "position_y",
                "delta_x",
                "delta_y",
                "speed",
                "heading",
                "lane_id",
                "lane_index",
                'events',
                'rewards',
                'dones'
            ]
            csv_writer.writerow(header)

            for row in collected_data:
                csv_writer.writerow(row)

    smarts.destroy()



class ChaseViaPointsAgent(Agent):
    def act(self, obs: Observation):
        if (
            len(obs.via_data.near_via_points) < 1
            or obs.ego_vehicle_state.road_id != obs.via_data.near_via_points[0].road_id
        ):
            return (obs.waypoint_paths[0][0].speed_limit, 0)

        nearest = obs.via_data.near_via_points[0]
        if nearest.lane_index == obs.ego_vehicle_state.lane_index:
            return (nearest.required_speed, 0)

        return (
            nearest.required_speed,
            1 if nearest.lane_index > obs.ego_vehicle_state.lane_index else -1,
        )


def main(scenarios, headless, num_episodes, max_episode_steps=None):
    agent_spec = AgentSpec(
        interface=AgentInterface.from_type(
            AgentType.LanerWithSpeed, max_episode_steps=max_episode_steps
        ),
        agent_builder=ChaseViaPointsAgent,
    )

    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=scenarios,
        agent_specs={"SingleAgent": agent_spec},
        headless=headless,
        sumo_headless=True,
    )

    # Convert `env.step()` and `env.reset()` from multi-agent interface to
    # single-agent interface.
    env = SingleAgent(env=env)
    scenario_name = "scenarios/cruise"
    scenarios_iterator = Scenario.scenario_variations([scenario_name], [])
    for episode in range(num_episodes):
        build_scenario([scenario_name])
        scenario = next(scenarios_iterator)
        agent = agent_spec.build_agent()
        obs = env.reset()
        prev_obs = obs

        collected_data = []
        _record_data(collected_data, obs.elapsed_sim_time, obs, prev_obs, None, None)

        done = False
        while not done:
            agent_action = agent.act(obs)
            obs, reward, done, info = env.step(agent_action)
            _record_data(collected_data, obs.elapsed_sim_time, obs, prev_obs, reward, done)
            prev_obs = obs
        observation_folder = "recorded_trajectories"
        if not os.path.exists(observation_folder):
            os.makedirs(observation_folder)

        csv_filename = Path(observation_folder) / f"{scenario.name}-{episode}.csv"
        with open(csv_filename, "w", newline="") as f:
            csv_writer = csv.writer(f, delimiter=",")
            header = [
                    "sim_time",
                    "agent_id",
                    "position_x",
                    "position_y",
                    "delta_x",
                    "delta_y",
                    "speed",
                    "heading",
                    "lane_id",
                    "lane_index",
                    'events',
                    'rewards',
                    'dones'
                ]
            csv_writer.writerow(header)

            for row in collected_data:
                csv_writer.writerow(row)

    env.close()


if __name__ == "__main__":
    parser = default_argument_parser("single-agent-example")
    args = parser.parse_args()

    if not args.scenarios:
        args.scenarios = [
            str(pathlib.Path(__file__).absolute().parents[0] / "scenarios" / "cruise")
        ]

    build_scenario(args.scenarios)

    main(
        scenarios=args.scenarios,
        headless=args.headless,
        num_episodes=50,
    )

