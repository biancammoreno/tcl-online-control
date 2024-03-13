import gym
from gym import spaces
import numpy as np
from HeaterGym import HeatersEnv


class TargetTrackingEnv():

    """
    ## Builds target from simulating the average consumption
    of a population of heaters over some days using a realistic data-set.
    Also computes the initial state distribution to define the true environment.
    """

    def __init__(self, env, n_heaters=1000, n_days=2):
        """env is a heater enviroment defined with any initial mu_0.
        n_days >= 2 as the first day have noise from the initial state distribution estimation
        """
        self.env = env
        self.n_heaters = n_heaters
        self.n_days = n_days

        # useful as a shortcut
        s = 1
        for key in list(env.observation_space.keys()):
            s = s*env.observation_space[key].n
        self.S = s
        self.A = env.action_space.n
        self.N_steps = env.max_steps


    def nominal_policy(self):
        """Computes the nominal policy
        """
        pi = np.zeros((self.N_steps, self.S, self.A))
        # proba 1 of keeping On or keeping Off if temperature between min and max 
        for temperature in range(self.env.T_min, self.env.T_max):
            for oper_state in range(2):
                ind_temperature = temperature - self.env.T_amb
                state = oper_state * self.env.n_temperatures + ind_temperature
                pi[:, state, oper_state] = 1
        # proba 1 of turning On if temperature below min
        for temperature in range(self.env.T_amb, self.env.T_min):
            ind_temperature = temperature - self.env.T_amb
            state_on = self.env.n_temperatures + ind_temperature
            state_off = ind_temperature
            pi[:,state_on, 1] = 1
            pi[:,state_off, 1] = 1
        # proba q of turning Off if temperature above max 
        for temperature in range(self.env.T_max, self.env.n_temperatures + self.env.T_amb):
            ind_temperature = temperature - self.env.T_amb
            state_on = self.env.n_temperatures + ind_temperature
            state_off = ind_temperature
            pi[:,state_on,0] = 1
            pi[:,state_off,0] = 1

        return pi

    def sample_policy(self, n, state, policy):
        return np.random.choice(self.A, p=policy[n, state,:])

    def nominal_consumption(self):
        """Simulates the nominal consumption of n_heaters 
        """
        # Compute the nominal policy
        nominal_pi = self.nominal_policy()
        # To store the consumption for n_days
        conso = np.zeros(self.N_steps * self.n_days + 1)
        # To store the initial conso and temperature of day 2 for all agents
        oper_state_0 = np.zeros(self.n_heaters)
        temperature_0 = np.zeros(self.n_heaters)
        # Simulate the nominal policy for n_heaters during n_days
        for m in range(self.n_heaters):
            observation = self.env.reset()
            conso[0] += observation['oper_state']
            state = self.env.obs_to_state(observation)
            for n in range(self.N_steps * self.n_days):
                # 1. Sample an action using the nominal policy
                action = self.sample_policy(n % self.N_steps, state, nominal_pi)
                # 2. Step in the env using this action
                observation, _, _ = self.env.step(action, week_day=np.arange(self.n_days)+1)
                next_state = self.env.obs_to_state(observation)
                conso[n+1] += observation['oper_state']
                # 3.  Update state
                state = next_state
                # 4. Save initial state at day 2 to build rho_0
                if n == (self.N_steps-1):
                    oper_state_0[m] = observation['oper_state']
                    temperature_0[m] = observation['temperature']

        # Compute average consumption during n_days
        conso = conso/self.n_heaters

        # Build rho_0
        rho_0 = np.zeros(self.S)
        for m in range(self.n_heaters):
            ind_temperature = temperature_0[m] - self.env.T_amb
            state = int(oper_state_0[m] * self.env.n_temperatures + ind_temperature)
            rho_0[state] += 1
        rho_0 = rho_0/self.n_heaters

        return conso, rho_0

    def deviation(self, signal):
        """"Computes a deviation curve
        """
        dev = np.zeros(self.N_steps * self.n_days)
        delta_t = 24*60/self.N_steps # size of time step in minutes

        if signal == 'one_hour_step':
            for k in range(self.n_days):
                # deviations everyday over a week (suppose at the same time)
                n1 = int((5 * 60 + 24 * 60 * k)/delta_t)  # deviation start at 5h
                n2 = int((6 * 60 + 24 * 60 * k)/delta_t)  # goes until 6h
                n3 = int((24 * 60 + 24 * 60 * k)/delta_t)  # gets negative until 24h
                dev[n1:n2] = 10
                dev[n2:n3] = -10 / (24 - 6)
        elif signal == 'eight_hours_step':
            for k in range(self.n_days):
                # deviations everyday over a week (suppose at the same time)
                n1 = int((11 * 60 + 24 * 60 * k)/delta_t)  # deviation is positive from 11h
                n2 = int((19 * 60 + 24 * 60 * k)/delta_t)  # until 19h
                n3 = int((24 * 60 + 24 * 60 * k)/delta_t)  # gets negative until 24h
                n4 = int((7 * 60 + 24 * 60 * k)/delta_t) # gets negative from 7h to 11h also
                dev[n1:n2] = 5
                dev[n2:n3] = 5*(-16 / 9)
                dev[n4:n1] = 5*(-16 / 9)
        elif signal == 'TSO':
            dev_TSO = np.load('curves/signal_TSO.npy') * 5
            if self.env.max_steps == dev_TSO.shape[0]:
                dev = dev_TSO
                for _ in range(self.n_days-1):
                    dev = [j for i in [dev, dev_TSO] for j in i]
        elif signal == 'none':
            return dev

        return dev

    def build_target(self, conso, signal):
        """Builds the target. Return the target and the initial state distribution from the target.
        """
        dev = self.deviation(signal)

        return (conso[0:-1] *100 + dev)/100





    

    

            




    



