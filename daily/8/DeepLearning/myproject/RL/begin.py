import gym


def print_env_base_info(env):
    print('# of states:{},# of action {}'.format(env.observation_space.shape[0],env.action_space.n))
    print('low value of state', env.observation_space.low)
    print('high value of state',env.observation_space.high)

# MountainCar-v0, MsPacman-v0,CartPole-v0
env=gym.make('Hopper-v2')

print_env_base_info(env)


for i_episode in range(200):
    env.reset()
    reward=0
    for _ in range(1000):
        env.render()
        a=env.action_space.sample()
        s,r,done,_=env.step(a)
        reward+=r
        if done:
            print('episode {} finish,get reward {}'.format(i_episode,reward))
            break
input()
env.close()