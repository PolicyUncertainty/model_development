import sys

sys.path.insert(0, "submodules/dc-egm/src/")
import numpy as np
import jax.numpy as jnp
from dcegm.solve import get_solve_function


start_age = 25
end_age = 75
n_periods = end_age - start_age + 1
resolution_age = 60
max_retirement_age = 72
minimum_SRA = 67
# you can retire four years before minimum_SRA
min_retirement_age = minimum_SRA - 4
# you can retire from min retirement age until max retirement age
n_possible_retirement_ages = max_retirement_age - min_retirement_age + 1
# when you are (start_age) years old, there can be as many policy states as there are years until (resolution_age)
n_possible_policy_states = resolution_age - start_age + 1
# choices: 0 = unemployment, , 1 = work, 2 = retire
choices = np.array([0, 1, 2])

options_test = {
    "state_space": {
        "n_periods": n_periods,
        "choices": np.array([0, 1, 2]),
        "endogenous_states": {
            "experience": np.arange(n_periods, dtype=int),
            "policy_state": np.arange(n_possible_policy_states, dtype=int),
            "retirement_age_id": np.arange(n_possible_retirement_ages, dtype=int),
        },
    },
    "model_params": {
        # info from state spoace used in functions
        "n_periods": n_periods,
        "n_possible_policy_states": n_possible_policy_states,
        # mandatory keywords
        "quadrature_points_stochastic": 5,
        # custom: model structure
        "start_age": start_age,
        "resolution_age": resolution_age,
        # custom: policy environment
        "minimum_SRA": minimum_SRA,
        "max_retirement_age": max_retirement_age,
        "min_retirement_age": min_retirement_age,
        "unemployment_benefits": 5,
        "pension_point_value": 0.3,
        "early_retirement_penalty": 0.036,
        # custom: params estimated outside model
        "belief_update_increment": 0.05,
        "gamma_0": 10,
        "gamma_1": 1,
        "gamma_2": -0.1,
    },
}

params_dict_test = {
    "mu": 0.5,  # Risk aversion
    "delta": 4,  # Disutility of work
    "interest_rate": 0.03,
    "lambda": 1e-16,  # Taste shock scale/variance. Almost equal zero = no taste shocks
    "beta": 0.95,  # Discount factor
    "sigma": 1,  # Income shock scale/variance.
}


def sparsity_condition(
    period, lagged_choice, policy_state, retirement_age_id, experience, options
):
    min_ret_age = options["min_retirement_age"]
    start_age = options["start_age"]
    max_ret_age = options["max_retirement_age"]
    n_policy_states = options["n_possible_policy_states"]

    age = start_age + period
    actual_retirement_age = min_ret_age + retirement_age_id
    # You cannot retire before the earliest retirement age
    if (age <= min_ret_age) & (lagged_choice == 2):
        return False
    # After the maximum retirement age, you must be retired
    elif (age > max_ret_age) & (lagged_choice != 2):
        return False
    # If you weren't retired last period, your actual retirement age is kept at minimum
    elif (lagged_choice != 2) & (retirement_age_id > 0):
        return False
    # If you are retired, your actual retirement age can at most be your current age
    elif (lagged_choice == 2) & (age <= actual_retirement_age):
        return False
    # Starting from resolution age, there is no more adding of policy states.
    elif policy_state > n_policy_states - 1:
        return False
    # If you have not worked last period, you can't have worked all your live
    elif (lagged_choice != 1) & (period == experience) & (period > 0):
        return False
    # You cannot have more experience than your age
    elif experience > period:
        return False
    # The policy state we need to consider increases by one increment
    # per period.
    elif policy_state > period:
        return False
    else:
        return True


def update_state_space(
    period, choice, lagged_choice, policy_state, retirement_age_id, experience, options
):
    next_state = dict()

    next_state["period"] = period + 1
    next_state["lagged_choice"] = choice

    age = period + options["start_age"]

    if age < options["resolution_age"]:
        next_state["policy_state"] = policy_state + 1
    else:
        next_state["policy_state"] = policy_state

    if lagged_choice == 2:  # Retirement
        next_state["retirement_age_id"] = retirement_age_id
    elif choice == 2:  # Retirement
        next_state["retirement_age_id"] = age - options["min_retirement_age"]

    if choice == 1:  # Work
        next_state["experience"] = experience + 1

    return next_state


def state_specific_choice_set(period, lagged_choice, policy_state, options):
    age = period + options["start_age"]
    min_individual_retirement_age = (
        options["min_retirement_age"]
        + policy_state * options["belief_update_increment"]
    )

    if age < min_individual_retirement_age:
        return np.array([0, 1])
    elif age >= options["max_retirement_age"]:
        return np.array([2])
    elif lagged_choice == 2:  # retirement is absorbing
        return np.array([2])
    else:
        return np.array([0, 1, 2])


state_space_functions = {
    "update_endog_state_by_state_and_choice": update_state_space,
    "get_state_specific_choice_set": state_specific_choice_set,
}

# put sparsity condition into endogenous states dict within options
options_test["state_space"]["endogenous_states"][
    "sparsity_condition"
] = sparsity_condition


def utility_func(consumption, choice, params):
    mu = params["mu"]
    delta = params["delta"]
    is_working = choice == 1
    utility = consumption ** (1 - mu) / (1 - mu) - delta * is_working
    return utility


def marg_utility(consumption, params):
    mu = params["mu"]
    marg_util = consumption**-mu
    return marg_util


def inverse_marginal(marginal_utility, params):
    mu = params["mu"]
    return marginal_utility ** (-1 / mu)


utility_functions = {
    "utility": utility_func,
    "inverse_marginal_utility": inverse_marginal,
    "marginal_utility": marg_utility,
}


def solve_final_period_scalar(
    choice,
    begin_of_period_resources,
    params,
    options,
    compute_utility,
    compute_marginal_utility,
):
    """Compute optimal consumption policy and value function in the final period.

    In the last period, everything is consumed, i.e. consumption = savings.

    Args:
        state (np.ndarray): 1d array of shape (n_state_variables,) containing the
            period-specific state vector.
        choice (int): The agent's choice in the current period.
        begin_of_period_resources (float): The agent's begin of period resources.
        compute_utility (callable): Function for computation of agent's utility.
        compute_marginal_utility (callable): Function for computation of agent's
        params (dict): Dictionary of model parameters.
        options (dict): Options dictionary.

    Returns:
        tuple:

        - consumption (float): The agent's consumption in the final period.
        - value (float): The agent's value in the final period.
        - marginal_utility (float): The agent's marginal utility .

    """

    # eat everything
    consumption = begin_of_period_resources

    # utility & marginal utility of eating everything
    value = compute_utility(
        consumption=begin_of_period_resources, choice=choice, params=params
    )

    marginal_utility = compute_marginal_utility(
        consumption=begin_of_period_resources, params=params
    )

    return marginal_utility, value, consumption


def budget_constraint(
    lagged_choice,  # d_{t-1}
    experience,
    policy_state,  # current applicable SRA identifyer
    retirement_age_id,
    savings_end_of_previous_period,  # A_{t-1}
    income_shock_previous_period,  # epsilon_{t - 1}
    params,
    options,
):
    # fetch necessary parameters (gammas for wage, pension_point_value & ERP for pension)
    gamma_0 = options["gamma_0"]
    gamma_1 = options["gamma_1"]
    gamma_2 = options["gamma_2"]
    pension_point_value = options["pension_point_value"]
    ERP = options["early_retirement_penalty"]

    # generate actual retirement age and SRA at resolution
    SRA_at_resolution = (
        options["minimum_SRA"] + policy_state * options["belief_update_increment"]
    )
    actual_retirement_age = options["min_retirement_age"] + retirement_age_id

    # calculate applicable SRA and pension deduction/increase factor
    # (malus for early retirement, bonus for late retirement)

    pension_factor = 1 - (actual_retirement_age - SRA_at_resolution) * ERP

    # decision bools
    is_unemployed = lagged_choice == 0
    is_worker = lagged_choice == 1
    is_retired = lagged_choice == 2

    # decision-specific income
    unemployment_benefits = options["unemployment_benefits"]
    labor_income = (
        gamma_0
        + gamma_1 * experience
        + gamma_2 * experience**2
        + income_shock_previous_period
    )
    retirement_income = pension_point_value * experience * pension_factor

    income = (
        is_unemployed * unemployment_benefits
        + is_worker * labor_income
        + is_retired * retirement_income
    )

    # calculate beginning of period wealth M_t
    wealth = (1 + params["interest_rate"]) * savings_end_of_previous_period + income

    return wealth


savings_grid = jnp.arange(start=0, stop=100, step=0.5)

import time

start_pre = time.time()
solve_func = get_solve_function(
    options=options_test,
    exog_savings_grid=savings_grid,
    utility_functions=utility_functions,
    budget_constraint=budget_constraint,
    state_space_functions=state_space_functions,
    final_period_solution=solve_final_period_scalar,
)
end_pre = time.time()
print("Preprocessing time: ", end_pre - start_pre)

solve_func(params_dict_test)
end_first = time.time()
print("First time_call: ", end_first - end_pre)

solve_func(params_dict_test)
end_second = time.time()
print("Second time_call: ", end_second - end_first)


f = open("runtime.txt", "w")
lines_to_write = [
    f"Pre processing time on max machine was {end_pre - start_pre} " f"seconds \n",
    f"First time_call was {end_first - end_pre} seconds \n",
    f"Second time_call was {end_second - end_first} seconds \n",
]
f.writelines(lines_to_write)
f.close()
