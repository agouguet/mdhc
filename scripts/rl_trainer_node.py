#!/usr/bin/env python3

import numpy as np
import os, yaml
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory


from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

from mdhc.envs.env_factory import *
from mdhc.envs.env_wrappers import *
from mdhc.policy.policy_factory import *
from mdhc.learning.callback import PolicyUpdateCallback, SuccessRateCallback

from mdhc.learning.curriculum.callback import CurriculumCallback
from mdhc.learning.curriculum.manager import CurriculumManager
from mdhc.learning.curriculum.wrapper import RewardTrackingVecWrapper

from mdhc.utils.config import load_ros2_package_config, get_rl_algo, get_latest_log_dir

class RLTrainerNode(Node):
    def __init__(self):
        super().__init__('rl_trainer_node')

        self.config_name = "config.yaml"
        self.config = load_ros2_package_config("mdhc", "config/"+self.config_name)
        model_name = "model/"
        self.model_path = os.path.join(get_package_share_directory("mdhc"), model_name)
        self.use_model = False

        postfix = "_norm" if self.config.env.normalize_reward or self.config.env.normalize_observation else ""
        self.name_log = f'{self.config.policy.name}_{self.config.learning.num_envs}_{self.config.env.name}{postfix}'
        if self.config.log.name != "":
            self.name_log = self.config.log.name

        self.curriculum = CurriculumManager(self.config.learning.curriculum)

        # Env creation
        env_ids = np.arange(self.config.learning.num_envs)
        env_fns = [self.make_env(i) for i in env_ids]
        self.env = DummyVecEnv(env_fns)
        self.env = VecNormalize(self.env, norm_obs=self.config.env.normalize_observation, norm_reward=self.config.env.normalize_reward)
        self.env = VecEnvDelayWrapper(self.env, delay_sec=(0.05 / self.config.learning.speed_time))
        self.env = RewardTrackingVecWrapper(self.env, self.curriculum, ros_node=self, save_path="./model/cl/" + self.name_log)

        policy, policy_kwargs = method_factory[self.config.policy.name].get_policy(self.config)

        common_kwargs = {
            'env': self.env,
            'tensorboard_log': self.config.log.log_dir,
            'verbose': 2,
            'n_epochs': self.config.algo.n_epochs,
            'n_steps': self.config.learning.n_steps,
            'batch_size': self.config.algo.batch_size,
            'learning_rate': get_schedule_fn(self.config.learning.learning_rate),
            'clip_range': get_schedule_fn(self.config.algo.clip_range),
            'ent_coef': self.config.algo.ent_coef,
            'vf_coef': self.config.algo.vf_coef
        }

        rl_algo = get_rl_algo(self.config.algo.name)

        if os.path.exists(self.model_path + ".zip"):
            self.model = rl_algo.load(self.model_path, **common_kwargs)
            self.model.num_timesteps = 0
            self.use_model = True
        else:
            self.model = rl_algo(
                policy,
                policy_kwargs=policy_kwargs,
                **common_kwargs,
                normalize_advantage=True,
                seed=self.config.algo.seed
            )
        
        self.env.set_model(self.model)

    def make_env(self, env_id):
        def _init():
            env = env_factory[self.config.env.name](env_id, self.config, env_id_display_log=0)
            env = Monitor(env, os.path.join(self.config.log.log_dir, str(env_id)))
            return env
        return _init

    def start_training(self):
        self.get_logger().info("Start training ...")
        log_interval = max(1, int(self.config.log.log_interval / self.config.learning.num_envs))

        checkpoint_callback = CheckpointCallback(
            save_freq=self.config.learning.save_model_frequency // self.config.learning.num_envs,  # fréquence = 1M steps globaux
            save_path="./model/",
            name_prefix=self.name_log+"_checkpoint"
        )

        curriculum_callback = CurriculumCallback(self.curriculum, self.env, ros_node=self)
        policy_callback = PolicyUpdateCallback(ros_node=self)
        success_callback = SuccessRateCallback()

        self.model.learn(
            total_timesteps=self.config.learning.total_timesteps,
            log_interval=log_interval,
            tb_log_name=self.name_log,
            reset_num_timesteps=not self.use_model,
            callback=[checkpoint_callback, curriculum_callback, policy_callback, success_callback] 
        )
        
        self.model.save("./model/"+self.name_log)
        self.env.close()


def main(args=None):
    NUM_TRAININGS = 1
    
    rclpy.init(args=args)
    
    for i in range(NUM_TRAININGS):

        node = RLTrainerNode()

        try:
            node.start_training()
        except Exception as e:
            node.get_logger().error(f"Exception capturée : {e}")
        finally:
            postfix = "_norm" if node.config.env.normalize_reward or node.config.env.normalize_observation else ""
            name_log = f'{node.config.policy.name}_{node.config.learning.num_envs}_{node.config.env.name}{postfix}'
            if node.config.log.name != "":
                name_log = node.config.log.name
            log_dir = get_latest_log_dir(node.config.log.log_dir, name_log)

            os.makedirs(log_dir, exist_ok=True)
            with open(os.path.join(log_dir, "model_params.yaml"), "w") as f:
                yaml.dump(node.config.to_dict(), f)

            node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
    