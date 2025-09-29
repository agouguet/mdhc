from stable_baselines3.common.vec_env import VecEnvWrapper
import numpy as np
from collections import deque
import time

class VecEnvDelayWrapper(VecEnvWrapper):
    def __init__(self, venv, delay_sec=0.1, ros_node=None):
        super().__init__(venv)
        self.delay_sec = delay_sec

    def step_async(self, actions):
        return self.venv.step_async(actions)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        # time.sleep(self.delay_sec)  #
        return obs, rewards, dones, infos

    def reset(self):
        return self.venv.reset()
        

from stable_baselines3.common.vec_env import VecEnvWrapper
import json
from geometry_msgs.msg import Pose

class RosJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Pose):
            return {
                "position": {
                    "x": obj.position.x,
                    "y": obj.position.y,
                    "z": obj.position.z,
                },
                "orientation": {
                    "x": obj.orientation.x,
                    "y": obj.orientation.y,
                    "z": obj.orientation.z,
                    "w": obj.orientation.w,
                }
            }
        if hasattr(obj, "tolist"):
            return obj.tolist()
        return super().default(obj)

class JsonLoggerVecWrapper(VecEnvWrapper):
    def __init__(self, venv, log_path="log.jsonl"):
        super().__init__(venv)
        self.log_path = log_path
        self.episode_id = -1
        self.step_count = 0
        self.file = open(self.log_path, "w")
        self.current_episode_logs = []

    def reset(self):
        obs = self.venv.reset()
        self.episode_id += 1
        self.step_count = 0
        self.current_episode_logs = []
        return obs
    
    def reset_log(self):
        self.file.close()
        self.file = open(self.log_path, "w")
        self.episode_id = -1
        self.step_count = 0

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        self.step_count += 1

        for i in range(len(obs)):
            entry = {
                "trial_id": self.episode_id,
                "info": infos[i],
            }
            self.current_episode_logs.append(entry)

        return obs, rewards, dones, infos

    def write_episode_logs(self):
        for entry in self.current_episode_logs:
            self.file.write(json.dumps(entry, cls=RosJsonEncoder) + "\n")
        self.file.flush()
        self.current_episode_logs = []

    def close(self):
        self.file.close()
        super().close()
