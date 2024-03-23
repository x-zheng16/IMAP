from gym.envs.registration import register

register(
    id="GridWorld-v0",
    entry_point="ap.envs.classic.gridworld:GridWorldContinuous",
    max_episode_steps=200,
)
