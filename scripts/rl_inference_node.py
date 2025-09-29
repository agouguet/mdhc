#!/usr/bin/env python3

import os, math
import yaml
import numpy as np
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

from stable_baselines3 import PPO, SAC, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecNormalize

from mdhc.envs.env_factory import *
from mdhc.envs.env_wrappers import *
from mdhc.policy.policy_factory import *
from mdhc.utils.config import load_ros2_package_config, get_rl_algo, get_latest_log_dir

class RLInferenceNode(Node):
    def __init__(self):
        super().__init__('rl_inference_node')

        self.config_name = "sparse/perpendicular_traffic.yaml"
        self.config = load_ros2_package_config("mdhc", "config/" + self.config_name)

        self.model_path = os.path.join(get_package_share_directory("mdhc"), "model/sparse/circle_crowd.zip")

        # --- Env construction ---
        env_fns = [self.make_env(0)]
        self.env = DummyVecEnv(env_fns)

        if self.config.env.normalize_observation or self.config.env.normalize_reward:
            self.env = VecNormalize.load(os.path.join(get_package_share_directory("mdhc"), "model/vecnormalize.pkl"), self.env)
            self.env.training = False
            self.env.norm_reward = False

        self.env = VecEnvDelayWrapper(self.env, delay_sec=(0.05 / self.config.learning.speed_time))
        self.env = JsonLoggerVecWrapper(self.env, log_path="log.json")

        policy_class = method_factory[self.config.policy.name].get_policy(self.config)[0]
        rl_algo = get_rl_algo(self.config.algo.name)

        self.model = rl_algo.load(self.model_path, env=self.env)

        self.get_logger().info("Model loaded. Inference start...")

    def make_env(self, env_id):
        def _init():
            env = env_factory[self.config.env.name](env_id, self.config, env_id_display_log=1)
            return env
        return _init

    def run_inference(self, episodes=50):
        success = 0
        episode = 0
        for ep in range(episodes):
            obs = self.env.reset()
            done = False
            total_reward = 0
            steps = 0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.env.step(action)
                total_reward += reward
                steps += 1
            self.env.write_episode_logs()
        self.env.close()

def main(args=None):
    rclpy.init(args=args)
    node = RLInferenceNode()
    try:
        node.run_inference()
    except Exception as e:
        node.get_logger().error(f"Error during Inference : {e}")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
