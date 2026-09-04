import gymnasium as gym

from . import agents, parallel_env_cfg

##
# The public package supports one Luna task: the closed-loop articulation with the deployment
# observation contract and projected gravity.
##

gym.register(
    id="Tracking-Flat-HU-L04-Parallel-Deploy-Gravity-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": parallel_env_cfg.HuL04ParallelDeployGravityEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HuL04ParallelDeployGravityPPORunnerCfg",
    },
)
