"""Models for the original algorithm (MD) and for the MFG algorithms from the literature (OMD and FP)"""

import numpy as np


# Mirror Descent adaption algorithm
class MD_CURL_changing_costs:
    """
   Mirror descent object for a given environment.
    """

    def __init__(self, env, algo, n_agents, targets, initial_pi, learn_noise=True, lr=1, **kwargs):

        self.env = env
        self.lr = lr
        self.targets = targets  # number of tagets X env.max_steps
        self.n_targets = targets.shape[0]
        self.n_agents = n_agents

        # useful as a shortcut
        s = 1
        for key in list(env.observation_space.keys()):
            s = s*env.observation_space[key].n
        self.S = s
        self.A = env.action_space.n
        self.N_steps = env.max_steps
        self.algo = algo
        # observe 10% of the whole population
        self.sub_n_agents = int(0.10 * self.n_agents)

        # initial probability transition
        uniform_P = np.ones((self.N_steps, self.S, self.A, self.S))/self.S
        self.P = kwargs.get("P", uniform_P)

        # initial policy
        self.policy = initial_pi
        if self.algo == 'accelerated_MD-CURL':
            self.aux_policy = initial_pi
        if self.algo == 'FP':
            self.av_policy = initial_pi

        # initial state distribution sequence
        self.mu = np.zeros((self.n_targets, self.N_steps, self.S))
        if self.algo == 'accelerated_MD-CURL':
            self.nu = np.zeros((self.n_targets, self.N_steps, self.S))
        if self.algo == 'FP':
            self.av_mu = np.zeros((self.n_targets, self.N_steps, self.S))

        # initial state-action value function
        self.Q = np.zeros((self.n_targets, self.N_steps, self.S, self.A))
        if self.algo == 'accelerated_MD-CURL':
            self.aux_Q = np.zeros((self.n_targets, self.N_steps, self.S, self.A))

        # to count the iterations
        self.global_count_step = 0
        self.count_step = np.zeros(self.n_targets)

        # vectorize g function from env to update P
        self.g_vec = np.vectorize(self.env.g)

        # wheter we want to learn the model external noise or not
        self.learn_noise = learn_noise

        # to gather the error
        self.global_error = []
        self.error = [[] for _ in range(self.n_targets)]
        
    def nu0(self):
        return self.env.rho_0

    def softmax(self, y, pi):
        """softmax function
        Args:
          y: vector of len |A|
          pi: vector of len |A|
        """
        max_y = max(y)
        exp_y = [np.exp(self.lr * (y[a] - max_y)) for a in range(y.shape[0])]
        norm_exp = sum(exp_y)
        return [l / norm_exp for l in exp_y]

    
    def softmax_original(self, y, pi):
        """softmax function
        Args:
          y: vector of len |A|
          pi: vector of len |A|
          k: iteration
        """
        size = y.shape[0]
        gamma = np.zeros((size))
        for a in range(size):
            gamma[a] = pi[a] * np.exp(y[a] * self.lr)

        return gamma / np.sum(gamma)


    def policy_from_logit(self, Q, prev_policy):
        """Compute policy from Q function
        """
        policy = np.zeros((self.N_steps, self.S, self.A))
        for n in range(self.N_steps):
            for x in range(self.S):
                policy[n,x,:] = self.softmax_original(Q[n,x,:], prev_policy[n,x,:])
                # assert np.sum(policy[n,x,:]) == 1,  'policy should sum to 1'
        
        return policy

    def reward(self, mu, x):
        """Target tracking reward
        """
        # Get proportion of heaters in state with oper_state = 1 and any temperature
        conso = np.sum(mu[:,1 * self.env.n_temperatures:2* self.env.n_temperatures], axis=1) 
        # Get current oper_state
        oper_state = int(x // self.env.n_temperatures)
        # Compute reward
        return -2 * (conso - self.target) * oper_state


    def state_action_value(self, mu, policy):
        """
        Computes the state-action value function
        (without updating pi)
        """
        Q = np.zeros((self.N_steps, self.S, self.A))

        reward = np.zeros((self.N_steps, self.S))
        for x in range(self.S):
            reward[:,x] = self.reward(mu,x)
            Q[self.N_steps-1,x,:] = reward[self.N_steps-1,x]

        for n in range(self.N_steps - 1, 0, -1):
            for x in range(self.S):
                for a in range(self.A):
                    Q[n-1,x,a] = reward[n-1,x] 
                    for x_next in range(self.S):
                        Q[n-1,x,a] += self.P[n, x,a,x_next] * np.dot(policy[n, x_next,:], Q[n,x_next,:])

        return Q

    def mu_induced(self, policy, P):
        """
        Computes the state distribution induced by a policy
        """
        mu = np.zeros((self.N_steps, self.S))
        mu[0,:] = self.nu0()
        for n in range(1,self.N_steps):
            for x in range(self.S):
                for x_prev in range(self.S):
                    mu[n, x] += mu[n-1, x_prev] * np.dot(policy[n-1, x_prev, :], P[n, x_prev, :, x])   

        # np.testing.assert_array_equal(np.sum(mu, axis=1), np.ones(self.N_steps), 'proba density should sum to 1')
        return mu 

    
    def fit(self, target_sequence, week_day):
        """
        Computes one iteration of MD
        n_iterations = number of iterations
        algo = MD-CURL or OMD-MFG
        target_ind = which target index
        week_day = which day we use to compute drains
        """
        self.week_day = week_day

        # initialize parameters 
        for i in range(self.n_targets):
            self.mu[i,:] = self.mu_induced(self.policy[i,:], self.P)
            if self.algo == 'accelerated_MD-CURL':
                self.nu[i,:] = self.mu_induced(self.aux_policy[i,:], self.P)
            if self.algo == 'FP':
                self.av_mu[i, :] = self.mu_induced(self.policy[i,:], self.P)
                self.sum_mu = self.mu.copy()
            
                
        for target_ind in target_sequence:
            print(self.global_count_step)
            self.target = self.targets[target_ind,:]
            self.global_count_step += 1
            self.count_step[target_ind] += 1
            # 1) compute policy
            if self.algo == 'MD-CURL':
                # 1b) Update the state-value function
                self.Q[target_ind,:] = self.state_action_value(self.mu[target_ind,:], self.policy[target_ind,:])
                # 2b) Compute the policy associated
                self.policy[target_ind,:] = self.policy_from_logit(self.Q[target_ind,:], self.policy[target_ind,:])
            elif self.algo == 'accelerated_MD-CURL':
                # Update the state-action value function
                self.Q[target_ind,:] = self.state_action_value(self.nu[target_ind, :], self.aux_policy[target_ind,:])
                # Compute the policy associated
                self.policy[target_ind,:] = self.policy_from_logit(self.Q[target_ind,:], self.aux_policy[target_ind, :])
            elif self.algo == 'FP':
                # compute the best policy greedy with respect to the Q function
                self.Q[target_ind,:] = self.state_action_value(self.av_mu[target_ind,:], self.policy[target_ind,:])
                self.policy[target_ind,:] = self.greedy_policy(self.Q[target_ind,:])

            # 3) Play policy
            if self.algo == 'FP':
                # for FP the final policy is the average policy
                conso, noise_matrix = self.play_policy(self.av_policy[target_ind,:], self.week_day)
            else:
                conso, noise_matrix = self.play_policy(self.policy[target_ind,:], self.week_day)

            # 4) Update probability transition kernel if P is unknown
            if self.learn_noise == True:
                self.P = self.update_prob_transition(noise_matrix, self.global_count_step)

            # 5) Compute error
            error = self.estimated_objective_function(conso, self.target)
            self.global_error.append(error)
            self.error[target_ind].append(error)
            
            # 6) Update the state-action distribution
            self.mu[target_ind,:] = self.mu_induced(self.policy[target_ind,:], self.P)
            if self.algo == 'accelerated_MD-CURL':
                # Update the auxiliary state-action value function
                self.aux_Q[target_ind,:] = self.state_action_value(self.mu[target_ind, :], self.aux_policy[target_ind,:])
                # Compute the policy associated
                self.aux_policy[target_ind,:] = self.policy_from_logit(self.aux_Q[target_ind,:], self.aux_policy[target_ind, :])
                # Update the auxiliary state-action distribution
                self.nu[target_ind,:] = self.mu_induced(self.aux_policy[target_ind,:], self.P)
            if self.algo == 'FP':
                # compute average policy
                self.av_policy[target_ind,:] = self.average_pi(self.av_policy[target_ind,:], self.sum_mu[target_ind,:], self.mu[target_ind,:], self.policy[target_ind,:])
                # compute average distribution
                self.av_mu[target_ind,:] =  self.average_mu(self.av_mu[target_ind,:], self.mu[target_ind,:], self.count_step[target_ind])
                # update sum of distributions
                self.sum_mu[target_ind,:] += self.mu[target_ind,:]
                if self.global_count_step == len(target_sequence):
                    self.policy = self.av_policy


    def sample_policy(self, n, state, pi):
        return np.random.choice(self.A, p=pi[n, state,:])

    def greedy_policy(self, Q):
        """Computes policy greedy with respect to the Q function
        """
        greedy_policy = np.zeros((self.N_steps, self.S, self.A))
        for n in range(self.N_steps):
            for state in range(self.S):
                greedy_policy[n, state, np.argmax(Q[n,state,:])] = 1.0

        return greedy_policy

    def play_policy(self, pi, week_day):
        """
        Play self.policy over a day for M agents. Observe a sub-population to update self.P. Compute error using all agents.
        t = episode
        """
        # To store the consumption
        conso = np.zeros(self.N_steps + 1)
        # To store noise
        noise_matrix = np.zeros((self.sub_n_agents, self.N_steps))
        # Simulate policy pi for n_heaters during n_days
        for m in range(self.n_agents):
            observation = self.env.reset()
            conso[0] += observation['oper_state']
            state = self.env.obs_to_state(observation)
            for n in range(self.N_steps):
                # 1. Sample an action using the policy
                action = self.sample_policy(n, state, pi)
                # 2. Step in the env using this action
                observation, _, epsilon = self.env.step(action, week_day=[week_day])
                next_state = self.env.obs_to_state(observation)
                conso[n+1] += observation['oper_state']
                # 3. Append noise for a sub-population of agents
                if m < self.sub_n_agents:
                    noise_matrix[m,n] = epsilon
                # 4. Update state
                state = next_state


        # Compute average consumption during n_days
        conso = conso/self.n_agents

        return conso, noise_matrix

    def update_prob_transition(self, noise_matrix, t):
        """Updates the probability transition kernel using the observed noise matrix.
        Here we suppose the deterministic dynamics are the same for all time steps.
        noise_matrix = matrix n_agents X N (noise trajectory for each observed agent)
        t = episode
        """
        next_P = (t-1)/t * self.P
        # compute next state conditioned in (x,a) for all time steps and agents' noises
        for x in range(self.S):
            for a in range(self.A):
                x_next = self.g_vec(x, a, noise_matrix[:,:]) 
                for n in range(self.N_steps):
                    for m in range(self.sub_n_agents):
                        next_P[n, x, a, int(x_next[m,n])] += 1/(self.sub_n_agents * t) 
    
        
        return next_P

    def estimated_objective_function(self, conso, target):
        """Compute the estimated objective function (not the mean field one as we suppose in practice there is no ground truth P)
        """
        return np.sum((conso[0:-1] - target)**2)

    def average_pi(self, av_pi, sum_mu, best_mu, best_pi):
        """
        Computes step j of FP policy generating the average distribution.
            Args:
                av_pi:  (N x |X| x |A|)
                    policy generating step j-1 average distribution
                sum_mu: (N x |X|)
                    sum of first j-1 best mu induced by the j-1 best policy
                best_mu: (N x |X|)
                    jth mu induced by jth best policy
                best_pi:  (N x |X| x |A|)
                    jth best policy
        """
        new_av_pi = np.zeros_like(av_pi)
        for n in range(self.N_steps):
            for s in range(self.S):
                    for a in range(self.A):
                        if sum_mu[n, s] + best_mu[n, s] == 0:
                            new_av_pi[n,s,a] = av_pi[n,s,a]
                        else:
                            new_av_pi[n, s, a] = (best_mu[n, s] * best_pi[n, s, a] + sum_mu[n, s] * av_pi[
                            n, s, a]) / (sum_mu[n, s] + best_mu[n, s])
        return new_av_pi

    def average_mu(self, av_mu, best_mu, j):
        """
        Compute step j of FP average distribution.
            Args:
                av_mu: (N x |X|)
                    average distribution at step j-1
                best_mu: (N x |X|)
                    distribution induced by best policy at step j
                j: scalar
                    step
        """
        new_av_mu = (j - 1) / j * av_mu + best_mu / j
        return new_av_mu
