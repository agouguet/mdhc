from mdhc.policy.simple import SimplePolicy, MultiModalFeatureExtractor
from mdhc.policy.saf import SAFPolicy, SAFFeatureExtractor
from mdhc.policy.stg import STGPolicy, STGFeatureExtractor
from mdhc.policy.mdhc import MDHCPolicy, MDHCFeatureExtractor
from mdhc.policy.mdhc_cross_modal import MDHCCrossModalPolicy, MDHCFeatureExtractor

import torch.nn as nn

def get_activation_fn(name):
    name = name.lower()
    activations = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "elu": nn.ELU,
        "leakyrelu": nn.LeakyReLU,
        "sigmoid": nn.Sigmoid,
        "gelu": nn.GELU,
        "selu": nn.SELU
    }
    if name not in activations:
        raise ValueError(f"Activation '{name}' non reconnue. Choisis parmi : {list(activations.keys())}")
    return activations[name]

class Method():
    def __init__(self, policy, features_extractor_class):
        self.policy = policy
        self.features_extractor_class = features_extractor_class

    def get_policy(self, config):
        # PPO Policy
        if self.features_extractor_class != None:
            policy_kwargs = dict(
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=dict(features_dim=config.policy.features_dim, config=config),
                net_arch=[dict(pi=config.policy.pi_layers, vf=config.policy.vf_layers)],
                activation_fn=get_activation_fn(config.policy.activation_fn)
            )
            return self.policy, policy_kwargs
        return self.policy, None


method_factory = dict()
def none_policy():
    return None

method_factory['none'] = Method("MlpPolicy", None)
method_factory['simple'] = Method(SimplePolicy, MultiModalFeatureExtractor)
method_factory['saf'] = Method(SAFPolicy, SAFFeatureExtractor)
method_factory['stg'] = Method(STGPolicy, STGFeatureExtractor)
method_factory['mdhc'] = Method(MDHCPolicy, MDHCFeatureExtractor)
method_factory['mdhc_cross_modal'] = Method(MDHCCrossModalPolicy, MDHCFeatureExtractor)
