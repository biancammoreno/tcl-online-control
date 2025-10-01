# (Online) Convex Optimization for Demand-Side Management

Code for running experiments on the article: (Online) Convex Optimization for Demand-Side Management: Application to Thermostatically Controlled Loads.

Unzip 'drain_trajs' to obtain a folder with all the data used in the simulations. It is made up of csv files of individual water heater consumption in Joules over a week with 10-minute time steps. This data is adapted from a dataset simulated by the SMACH platform (https://hal.science/hal-03195500/document). The original dataset reports water consumption in liters at a 1-minute step. We averaged these values to obtain a 10-minute step. To compute the `drain` column in our trajectories, we take the water consumption (in liters) from the original SMACH dataset multiplied by the water capacity, the time step, and the temperature difference between the current water temperature and the cold-water temperature entering the water-heater (as also provided in the original dataset).

If you use the data from this repository, please cite:
Moreno, B., Brégère, M., Gaillard, P., Oudjane, N., **Online Convex Optimization for Demand-Side Management: Application to Thermostatically Controlled Loads.** *J. Optim. Theory Appl.* 205, 43 (2025). [https://doi.org/10.1007/s10957-025-02658-9](https://doi.org/10.1007/s10957-025-02658-9)

The 'curves' folder contains the nominal consumption curve, the three possible targets ('one_hour_step', 'eight_hours_step', and 'TSO' - the target constructed with the transmission system operator's balancing signal).  This folder also contains the probability transition kernel simulated using the dataset drains.

To run an experiment save all files to a folder, create a virtual environment, install the requirements in requirements.txt and execute

python main_heater.py --signal 'TSO' --n_heaters 1000 --n_iterations 20 --algo 'MD-CURL' --learn False --p_value True --epsilon 0

where the variable signal can receive one or more types of signal, depending on whether the test involves changing targets or not.

For an explanation of the parameters execute

python main.py -h

A new directory results is created to save images and graphs
