import gym
env = gym.make('MountainCar-v0')

class TileCoder:
    def __init__(self, layers, features):
        self.layers = layers
        self.features = features
        self.codebook = {} 

    def get_feature(self, codeword):
        if codeword in self.codebook:
            return self.codebook[codeword]
        count = len(self.codebook)
        if count>=self.features:
            return hash(codeword)
        else:
            self.codebook[codeword] = count
            return count
    
    def __call__(self, floats=(), ints=()):
        dim = len(floats)
        scaled_floats = tuple( f * self.layers * self.layers for f in floats)
        features = []
        for layer in range(self.layers):
            codeword = (layer,) + tuple(int((f+(1+dim * i)*layer)/self.layers) for i,f in enumerate(scaled_floats))+ints
            feature = self.get_feature(codeword)
            features.append(feature)
        return features
    

class SARSAAgent:
    def __init__(self,env, layers=8, features=1893, gamma=1.0, learning_rate=0.03, epsilon=0.001):
        self.action_n = env.action_space.n # 动作数
        self.obs_low = env.observation_space.low
        self.obs_scale = env.observation_space.high - env.observation_space.low # 观测空间范围
        self.encoder = TileCoder(layers, features)
        