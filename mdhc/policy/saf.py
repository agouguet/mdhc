import torch as th
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
import math

# ---------- Social Angular Field Module ----------
class SocialAngularField(nn.Module):
    def __init__(self, nb_bins=72, max_range=10.0, sigma_deg=10.0, softmin_alpha=10.0, device='cpu'):
        """
        nb_bins: nombre de bins angulaires (p.ex. 72 -> 5° par bin)
        max_range: distance max à considérer (clamp)
        sigma_deg: largeur angulaire de diffusion (en degrés) pour pondération
        softmin_alpha: paramètre pour approx soft-min (plus grand -> plus proche du vrai min)
        """
        super().__init__()
        self.nb_bins = nb_bins
        self.max_range = float(max_range)
        self.sigma = math.radians(sigma_deg)
        self.softmin_alpha = float(softmin_alpha)
        # precompute bin angles centers in [-pi, pi)
        bin_centers = th.linspace(-math.pi, math.pi, steps=nb_bins+1)[:-1] + (math.pi*2/nb_bins)/2.0
        # register as buffer so it moves with .to(device)
        self.register_buffer("bin_centers", bin_centers)
        self.device = device

    def forward(self, humans):
        """
        humans: (B, N, H, 5) (dx, dy, vx, vy, r)
        returns: tensor (B, C, nb_bins) where C = 3 (occupancy, distance, ttc)
        """
        B, N, H, Fh = humans.shape
        device = humans.device

        humans = th.nan_to_num(humans, nan=0.0, posinf=10.0, neginf=-10.0)

        assert th.isfinite(humans).all()

        # latest (current) positions/velocities
        dx = humans[..., -1, 0]  # (B,N)
        dy = humans[..., -1, 1]
        vx = humans[..., -1, 2]
        vy = humans[..., -1, 3]
        # recompute distance for stability
        dist = th.sqrt(dx*dx + dy*dy).clamp(min=1e-6)  # (B,N)

        # angles in [-pi, pi)
        ang = th.atan2(dy, dx)  # (B,N)

        # Time-to-collision (TTC) relative along radial direction
        # radial velocity = projection of velocity onto radial vector
        radial_vel = (dx*vx + dy*vy) / dist  # (B,N)
        # ttc positive if closing (radial_vel < 0), else large (no collision)
        eps = 1e-6
        ttc = th.full_like(dist, float('inf'))
        closing_mask = radial_vel < -1e-3
        ttc_val = dist / (-radial_vel.clamp(min=eps))
        ttc = th.where(closing_mask, ttc_val, th.full_like(dist, 1e6))  # large value if not closing

        # For stability clamp distances and ttc
        dist_clamped = dist.clamp(min=1e-6, max=self.max_range)
        ttc_clamped = ttc.clamp(min=1e-6, max=1e6)

        assert th.isfinite(dist_clamped).all()
        assert th.isfinite(ttc_clamped).all()

        # Prepare bin centers (1D tensor) and compute angular differences
        # bin_centers shape (nb_bins,)
        bin_centers = self.bin_centers.to(device)  # (nb_bins,)
        # We want angle difference for broadcasting: (B, N, nb_bins)
        ang_exp = ang.unsqueeze(-1)  # (B,N,1)
        bin_centers_exp = bin_centers.view(1, 1, -1)  # (1,1,nb_bins)
        # shortest circular difference in [-pi, pi]
        ang_diff = (ang_exp - bin_centers_exp + math.pi) % (2*math.pi) - math.pi  # (B,N,nb_bins)

        # angular weights: gaussian on angle difference
        w_ang = th.exp(-0.5 * (ang_diff / (self.sigma + 1e-9))**2)  # (B,N,nb_bins)

        # Optional: mask absent humans using large dist (assumes dist>max_range ~ absent)
        present_mask = (dist_clamped < self.max_range).float().unsqueeze(-1)  # (B,N,1)
        w = w_ang * present_mask  # zero out absent humans

        # occupancy per bin: sum of weights (normalize by max per bin to keep in [0,1])
        occupancy = w.sum(dim=1)  # (B, nb_bins)
        # normalize occupancy to [0,1] by dividing by max possible (approx N)
        occupancy_norm = occupancy / (N + 1e-6)

        # soft-min for distance per bin:
        # we compute exp(-alpha * dist) * weight -> gives large weight to small distances
        alpha = self.softmin_alpha
        score_dist = w * th.exp(-alpha * dist_clamped.unsqueeze(-1))  # (B,N,nb)
        sum_score = score_dist.sum(dim=1) + 1e-9  # (B, nb)
        # weighted average of distances with weight proportional to score -> smaller dist dominates
        dist_weighted = ( (score_dist * dist_clamped.unsqueeze(-1)).sum(dim=1) ) / sum_score  # (B, nb)
        # if sum_score == 0 (no humans), set dist to max_range
        no_presence = (sum_score < 1e-9)
        dist_weighted = th.where(no_presence, th.full_like(dist_weighted, self.max_range), dist_weighted)

        # soft-min for ttc similarly (small TTC = imminent collision)
        score_ttc = w * th.exp(-alpha * ttc_clamped.unsqueeze(-1))
        sum_score_ttc = score_ttc.sum(dim=1) + 1e-9
        ttc_weighted = ( (score_ttc * ttc_clamped.unsqueeze(-1)).sum(dim=1) ) / sum_score_ttc
        ttc_weighted = th.where((sum_score_ttc < 1e-9), th.full_like(ttc_weighted, 1e6), ttc_weighted)

        # final channel stacking: (B, C, nb_bins)
        # normalize channels:
        occ = occupancy_norm.unsqueeze(1)  # (B,1,nb)
        d = (dist_weighted / self.max_range).unsqueeze(1)  # normalized [0,1]
        # for TTC, normalize by some horizon (e.g., 5s). Values >5 -> saturate as non-imminent.
        ttc_horizon = 5.0
        t = (ttc_weighted.clamp(max=ttc_horizon) / ttc_horizon).unsqueeze(1)  # (B,1,nb)
        out = th.cat([occ, d, t], dim=1)  # (B,3,nb_bins)
        return out  # channels-first for Conv1D

# ---------- Small CNN to encode the angular field ----------
class AngularFieldEncoder(nn.Module):
    def __init__(self, nb_bins=72, in_channels=3, out_dim=64):
        super().__init__()
        # simple 1D conv stack (keeps nb_bins length)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=5, padding=2, stride=1),
            nn.SiLU(),
            nn.Conv1d(16, 32, kernel_size=5, padding=2, stride=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),  # pool over angle -> (B,32,1)
            nn.Flatten(),
            nn.Linear(32, out_dim),
            nn.SiLU()
        )

    def forward(self, angular_field):
        # angular_field: (B, C, nb_bins)
        return self.net(angular_field)  # (B, out_dim)


class Swish(nn.Module):
    def forward(self, x):
        return x * th.sigmoid(x)

class SAFFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512, config=None):
        super().__init__(observation_space, features_dim)
        self.config = config

        # Dimensions robot / goal
        self.goal_size = 1 + int(self.config.env.obs.goal_social) + int(self.config.env.obs.goal_dist)
        self.vel_size = 2 if self.config.env.obs.robot_velocity else 0
        self.robot_obs_size = self.goal_size + self.vel_size

        # Humans
        self.human_size = 4
        self.human_history = self.config.env.obs.human_history if self.config.env.obs.human else 0
        self.number_max_human = self.config.env.obs.human_number if self.config.env.obs.human else 0

        # Scan
        self.scan_size = self.config.env.obs.scan_dim
        self.nb_slice = self.config.env.obs.scan_slice
        self.scan_history = self.config.env.obs.scan_history
        self.scan_tile = self.config.env.obs.scan_tile
        self.scan_obs_size = int(self.scan_size * self.scan_history * self.scan_tile)
        size = int(self.config.env.obs.scan_avg_pool) + int(self.config.env.obs.scan_min_pool)
        if size > 0:
            self.scan_obs_size = int((self.scan_size / self.nb_slice) * self.scan_history * self.scan_tile * size)

        self.nb_angular_bins = 36 #72
        self.angular_field_out_dim = 32 #64
        self.social_field_mod = SocialAngularField(nb_bins=self.nb_angular_bins, max_range=10.0, sigma_deg=10.0)
        self.social_encoder = AngularFieldEncoder(nb_bins=self.nb_angular_bins, in_channels=3, out_dim=self.angular_field_out_dim)

        # ---------------- Scan CNN ----------------
        self.scan_net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2),
            Swish(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            Swish(),
        )

        with th.no_grad():
            dummy_input = th.zeros(1, 1, self.scan_obs_size)
            dummy_out = self.scan_net(dummy_input)
            conv_output_size = dummy_out.numel()

        self.scan_fc = nn.Sequential(
            nn.Linear(conv_output_size, 256),
            Swish()
        )

        # ---------------- Robot MLP ----------------
        self.robot_fc = nn.Sequential(
            nn.Linear(self.robot_obs_size, 32),
            Swish(),
            nn.Linear(32, 32),
            Swish()
        )

        # ---------------- Final FC ----------------
        # ajuste final_fc input size : before it was 256 + 32 + 32 = features_dim input
        # now we add social encoder 64
        self.final_fc = nn.Sequential(
            nn.Linear(256 + 32 + 32, features_dim),
            Swish()
        )

    def forward(self, observations):
        # découpage des observations
        scan = observations[:, :self.scan_obs_size]
        robot = observations[:, self.scan_obs_size:self.scan_obs_size + self.robot_obs_size]
        humans = observations[:, self.scan_obs_size + self.robot_obs_size:]

        # ---- Scan branch ----
        x_scan = scan.unsqueeze(1)  # (batch, 1, scan_size)
        x_scan = self.scan_net(x_scan)
        x_scan = x_scan.flatten(start_dim=1)
        x_scan = self.scan_fc(x_scan)

        # ---- Robot branch ----
        x_robot = self.robot_fc(robot)  # (batch, 32)

        # ---- SAF branch ----
        humans = humans.reshape(-1, self.number_max_human, self.human_history, self.human_size)
        angular_field = self.social_field_mod(humans)  # (B, 3, nb_bins)
        x_social = self.social_encoder(angular_field)

        # ---- Fusion ----
        x = th.cat([x_scan, x_robot, x_social], dim=1)
        x = self.final_fc(x)

        return x


class SAFPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            **kwargs,
        )

        latent_dim = self.mlp_extractor.latent_dim_pi
        
        self.mu = nn.Linear(latent_dim, action_space.shape[0])
        self.log_std = nn.Parameter(th.zeros(action_space.shape[0]))

    def forward_actor(self, features):
        mu = self.mu(features)
        std = th.exp(self.log_std)
        return mu, std

    def _get_action_dist_from_latent(self, latent_pi, latent_vf=None):
        mean_actions, std = self.forward_actor(latent_pi)
        return self.action_dist.proba_distribution(mean_actions, std)

    def forward_critic(self, features):
        return self.value_net(features)
