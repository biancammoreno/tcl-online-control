

from HeaterTarget import TargetTrackingEnv
from HeaterGym import HeatersEnv
from online_md_changing_costs import MD_CURL_changing_costs
import matplotlib.pyplot as plt
import os
import numpy as np


def main(signal, n_heaters, n_iterations, algo, learn, p_value, epsilon):

    # UNCOMMENT TO BUILD TARGET

    # # # define initial env for building the target using the uniform distribution for rho_0
    # initial_env = HeatersEnv(max_steps)
    # # define target and the corresponding initial distribution
    # TargetEnv = TargetTrackingEnv(initial_env, n_heaters, n_days)
    # conso, rho_0 = TargetEnv.nominal_consumption()
    # np.save('curves/rho_0', rho_0)
    # np.save('curves/nominal_conso', conso)
    # for dev in signal:
    #     print(dev)
    #     target = TargetEnv.build_target(conso, dev)
    #     plt.plot(target[max_steps:], label='target', c='red', linewidth=1.5)
    #     plt.plot(conso[max_steps:], label='conso', c='blue', linewidth=1.5)
    #     plt.title('Target and Nominal consumptions')
    #     plt.xlabel('Time')
    #     plt.ylabel('Average cons. (% of max. cons.)')
    #     plt.savefig('images/target_' + dev + '_conso_curve.png')
    #     np.save('curves/target_' + dev, target)
    #     plt.close()

    # number of days we consider
    n_days = 1
    week_day = 2
    # number of time steps of 10 minutes in a day
    max_steps = int(24 * 60 * n_days/10)

    # UNCOMMENT TO RUN MODEL
    # load nominal consumption and heater's initial state
    rho_0 = np.load('curves/rho_0.npy')
    conso = np.load('curves/nominal_conso.npy')

    # define real env 
    env = HeatersEnv(max_steps, rho_0=rho_0)

    # load possible targets
    targets = np.zeros((len(signal), max_steps))
    for i, dev in enumerate(signal):
        target = np.load('curves/target_' + dev + '.npy')
        targets[i,:] = target[max_steps:max_steps*2]

    # # UNCOMMENT TO BUILD PROBABILITY KERNELS
    # # true_P = env.P(week_day)
    # # np.save('curves/true_P', true_P)
    # # zero_noise_P = env.P_no_noise()
    # # np.save('curves/zero_noise_P', zero_noise_P)
    
    # load probability kernels 
    true_P = np.load('curves/true_P.npy')
    zero_noise_P = np.load('curves/zero_noise_P.npy')


    if p_value == True: 
        P = true_P
    else:
        P = zero_noise_P


    # MD-CURL WITH CHANGING COSTS
    # getting initial policy
    if epsilon == 0: # uniform initialization
        initial_pi = np.ones((targets.shape[0], env.max_steps, env.S, env.A))/env.A
    else:
        pi = np.load('curves/nominal_policy_epsilon_' + str(epsilon) + '.npy')
        initial_pi = np.zeros((1, env.max_steps, env.S, env.A))
        initial_pi[0,:] = pi
    # define model - learn_noise = if to learn P or not; P = the value of P to initate if learn is True, or the value of P to use all iter. if learn is False
    model = MD_CURL_changing_costs(env, algo, n_heaters, targets, initial_pi, learn_noise=learn, P=P)
    # create sequence of targets over all episodes
    sequence_targets = np.random.randint(0, len(signal), n_iterations)
    # fit model
    model.fit(sequence_targets, week_day)
    
    # simulate and save the best policy for each target over n_heaters for the target day
    for i, dev in enumerate(signal):
        policy_conso, _  = model.play_policy(model.policy[i,:], week_day)
        plt.plot(policy_conso, label='best policy conso', c='blue', linewidth=1.5)
        plt.plot(conso[max_steps:max_steps*week_day],label='nominal', c='purple', linewidth=1.5)
        plt.plot(targets[i,:], label='target', c='red', linewidth=1.5)
        plt.title('Target vs Consumption simulated with optimal policy for target' + dev)
        plt.xlabel('Time')
        plt.ylabel('Average cons. (% of max. cons.)')
        plt.legend()
        plt.savefig('images/consumption_curve_optimal_policies_' + dev + '_' + algo + '_learn_' + str(learn) + '_proba_' + str(p_value) + '_epsilon_' + str(epsilon) + '.png')
        plt.close()
        plt.plot(model.error[i])
        plt.savefig('images/error_'+ dev + '_' + algo + '_learn_' + str(learn) + '_proba_' + str(p_value) + '_epsilon_' + str(epsilon) + '.png')
        plt.close()
        np.save('policies/policy_' + dev + '_' + algo + '_learn_' + str(learn) + '_proba_' + str(p_value) + '_epsilon_' + str(epsilon) + '.npy', model.policy[i, :])


    
if __name__ == '__main__':
    import argparse
    import ast
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--signal', type=str, nargs='+', required=True, help='List of signals to build targets: TSO, one_hour_step, eight_hours_step')
    parser.add_argument('--n_heaters', type=int, required=True, help='Number of heaters to simulate')
    parser.add_argument('--n_iterations', type=int, required=True, help='Number of iterations')
    parser.add_argument('--algo', type=str, required=True, help='Algorithm to run: MD-CURL, accelerated_MD-CURL, FP')
    parser.add_argument('--learn', type=ast.literal_eval, required=True, help='True or False: if the external noise dynamics should be learned')
    parser.add_argument('--p_value', type=ast.literal_eval, required=True, help='True or False: probability kernel is known')
    parser.add_argument('--epsilon', type=float, required=True, help='how to initialize the policy: if 0, uniform; if in (0,1), deviation from nominal.')
    args = parser.parse_args()

    main(signal=args.signal, n_heaters=args.n_heaters, n_iterations=args.n_iterations, algo=args.algo, learn=args.learn, p_value=args.p_value, epsilon=args.epsilon)  