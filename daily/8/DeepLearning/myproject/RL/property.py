from gym import spaces,envs

space = spaces.Discrete(8) # Set with 8 elements {0, 1, 2, ..., 7}
x = space.sample()
assert space.contains(x)
assert space.n == 8

env_names=sorted(envs.registry.all(),key=lambda x:x.id)
for name in env_names:
    print(name)