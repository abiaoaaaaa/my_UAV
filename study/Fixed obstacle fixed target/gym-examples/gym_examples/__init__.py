from gym.envs.registration import register

register(
    id="gym_examples/GridWorld-v0",
    entry_point="gym_examples.envs:GridWorldEnv",
)

register(
    id="gym_examples/env_test1-v0",
    entry_point="gym_examples.envs:DroneEnv",
)