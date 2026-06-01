import argparse
import json
import os
import sys

import torch
from stable_baselines3 import PPO


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


class DeterministicActorWrapper(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        dist = self.policy.get_distribution(obs)
        return dist.distribution.mean


def load_policy_entry(name_or_path):
    policies_path = os.path.join(REPO_ROOT, "configs", "policies.json")
    if not os.path.exists(policies_path):
        return None
    with open(policies_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("policies", {}).get(name_or_path)


def resolve_path(path):
    if os.path.isabs(path):
        return path

    cwd_path = os.path.abspath(path)
    if os.path.exists(cwd_path):
        return cwd_path

    repo_path = os.path.join(REPO_ROOT, path)
    return os.path.abspath(repo_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy",
        default="v8_selected",
        help="Policy registry name or explicit .zip model path",
    )
    parser.add_argument(
        "--output",
        default="runs/exported/aqrl_v8_actor_cpu.ts",
        help="TorchScript output path",
    )
    args = parser.parse_args()

    entry = load_policy_entry(args.policy)
    model_path = entry["model"] if entry is not None else args.policy
    model_path = resolve_path(model_path)

    ppo = PPO.load(model_path, device="cpu")
    policy = ppo.policy
    policy.eval()

    obs_shape = policy.observation_space.shape
    if obs_shape is None or len(obs_shape) != 1:
        raise ValueError(f"Expected flat observation shape, got {obs_shape}")

    actor = DeterministicActorWrapper(policy)
    actor.eval()

    example_obs = torch.zeros((1, obs_shape[0]), dtype=torch.float32)
    traced = torch.jit.trace(actor, example_obs)

    output_path = args.output
    if not os.path.isabs(output_path):
        output_path = os.path.join(REPO_ROOT, output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    traced.save(output_path)

    print("exported:", output_path)
    print("model:", model_path)
    print("obs_dim:", obs_shape[0])
    print("action_dim:", int(policy.action_space.shape[0]))
    if entry is not None and "env" in entry:
        print("env_config:", json.dumps(entry["env"], indent=2))


if __name__ == "__main__":
    main()
