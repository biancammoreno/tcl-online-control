import gym
from gym import spaces
import numpy as np
import os, random
import pandas as pd
from abc import abstractmethod

# fixed parameters
vol = 0.2
height = 1.37
EI = 0.035
EI = EI/4
rho = 1000
capWater = 4185
CI = 0.033
Sect = vol/height
ray = np.sqrt(Sect/3.14)
coefLoss = CI/EI * 2 * 3.14 * ray
powerRes = 2200
stepSize = 1/6
lossH = (coefLoss * 3600)/(capWater * rho * Sect)
e_unit = vol * rho * capWater
rhoMax = powerRes * 3600

class HeatersEnv(gym.Env):

    def __init__(self, max_steps, **kwargs):
        self.max_steps = max_steps
        self.T_min = 50 + 273
        self.T_max = 65 + 273
        self.T_amb = 25 + 273

        self.n_temperatures = self.T_max - self.T_amb + 2

        # Observations are dictionaries with the agent's location.
        # Each location is encoded as an element of {0,1} X {integer temperature}
        self.observation_space = spaces.Dict(
            {
                "oper_state": spaces.Discrete(2),
                "temperature": spaces.Discrete(self.n_temperatures, start=self.T_amb),
            }
        )

        # We have 2 actions, corresponding to "turn on" and "turn off"
        self.action_space = spaces.Discrete(2)

        # useful as a shortcut
        s = 1
        for key in list(self.observation_space.keys()):
            s = s*self.observation_space[key].n
        self.S = s
        self.A = self.action_space.n

        # initial state-action distribution
        rho_0 = np.ones(2 * self.n_temperatures)/(2 * self.n_temperatures)
        self.rho_0 = kwargs.get("rho_0", rho_0)


    def _get_obs(self):
        return {"oper_state": self._heater_oper_state, "temperature": self._heater_temperature}


    def obs_to_state(self, obs):
        """
        Transform observation to state
        """
        operational_state = obs["oper_state"]
        temperature = obs["temperature"]
        ind_temperature = temperature - self.T_amb
        return operational_state * self.n_temperatures + ind_temperature

    def reset(self, seed=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's initial state
        state = np.random.choice(self.rho_0.shape[0], p=self.rho_0)
        self._heater_oper_state = int(state // self.n_temperatures)
        self._heater_temperature = int(state % self.n_temperatures) + self.T_amb

        # Step count since episode start
        self.step_count = 0

        observation = self._get_obs()

        return observation


    def P(self, week_day):
        """
        Build a ground truth probability transition kernel from data
        week_day = which day we consider to build P
        """
        P = np.zeros((self.max_steps, self.S, self.A, self.S))
        # get number of agents
        n_agents = len( os.listdir('drain_trajs'))
        # vectorize g
        g_vec = np.vectorize(self.g)
        # get drain trajectory from oracle
        for filename in os.listdir('drain_trajs'):
            df_drain = pd.read_csv('drain_trajs/' + filename, sep=',')
            self.drain_traj = np.asarray(df_drain.loc[:, 'drain'])[self.max_steps *(week_day -1):self.max_steps * week_day]
            for x in range(self.S):
                for a in range(self.A):
                    x_next = g_vec(x, a, self.drain_traj) 
                    for n in range(self.max_steps):
                        P[n, x, a, int(x_next[n])] += 1

        P = P/n_agents

        return P


    def P_no_noise(self):
        """
        Build a probability transition kernel that uses the deterministic dynamics but suppose the noise is always 0.
        """
        P = np.zeros((self.max_steps, self.S, self.A, self.S))
        for x in range(self.S):
            for a in range(self.A):
                x_next = self.g(x,a,0)
                P[:,x,a,int(x_next)] = 1

        return P
             

    def g(self, state, action, noise):
        """Heater's deterministic dynamics
        """
        oper_state = int(state // self.n_temperatures)
        old_temperature = state % self.n_temperatures + self.T_amb

        new_temperature = old_temperature + stepSize * (-lossH * (old_temperature - self.T_amb) + rhoMax * oper_state / e_unit) - \
        noise / e_unit
        
        new_oper_state, ind_new_temperature = self.next_state(action, oper_state, new_temperature, old_temperature)

        return new_oper_state * self.n_temperatures + ind_new_temperature


    def next_state(self, action, oper_state, new_temperature, old_temperature):
        # heater is on
        ind_new_temperature = new_temperature - self.T_amb
        ind_old_temperature = old_temperature - self.T_amb
        if oper_state == 1:
            if new_temperature > self.T_max:
                return 0, ind_old_temperature
            elif new_temperature < self.T_min:
                if ind_new_temperature > 0:
                    return 1, ind_new_temperature
                else:
                    return 1, 0
            else:
                return action, ind_new_temperature

        # heater is off
        if oper_state == 0:
            if new_temperature < self.T_min:
                if ind_new_temperature > 0:
                    return 1, ind_new_temperature
                else:
                    return 1, 0
            elif new_temperature > self.T_max:
                return 0, ind_old_temperature
            else:
                return action, ind_new_temperature


    def step(self, action, week_day=[2]):
        """Simulates one heater choosing an action. 
        week_day = list of which week days we consider. If multiple, the episode does not restart until the end of the list.
        """
        if self.step_count == 0:
            # get drain trajectory from oracle
            filename = random.choice(os.listdir('drain_trajs')) 
            df_drain = pd.read_csv('drain_trajs/' + filename, sep=',')
            self.drain_traj = np.asarray(df_drain.loc[:, 'drain'])[self.max_steps *(week_day[0] -1):self.max_steps * week_day[-1]]

        # Get random noise

        epsilon = self.drain_traj[self.step_count]

        # Update agent's state
        observation = self._get_obs()
        state = self.obs_to_state(observation)
        new_state = self.g(state, action, epsilon)
        self._heater_oper_state = int(new_state // self.n_temperatures)
        self._heater_temperature = int(new_state % self.n_temperatures) + self.T_amb

        # update step count
        self.step_count += 1
        
        # An episode is done iff we reach the max number of steps
        truncated = False
        if self.step_count >= self.max_steps * len(week_day):
            truncated = True

        observation = self._get_obs()

        return observation, truncated, epsilon

