import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import stats

# %%
########  ####  ######  ##    ##    ###    ########  ##     ##
##     ##  ##  ##    ## ##   ##    ## ##   ##     ## ###   ###
##     ##  ##  ##       ##  ##    ##   ##  ##     ## #### ####
########   ##  ##       #####    ##     ## ########  ## ### ##
##         ##  ##       ##  ##   ######### ##   ##   ##     ##
##         ##  ##    ## ##   ##  ##     ## ##    ##  ##     ##
##        ####  ######  ##    ## ##     ## ##     ## ##     ##


def pick_arm(
    q_values,
    counts,
    strategy,
    success,
    failure,
    costs,
    share_elapsed,
    eta=0.1,
    epsilon=0.2,
    psi=0.5,
    return_what="arm_choice",
):
    if strategy == "random":
        arm_choice = np.random.randint(0, len(q_values))

    if strategy == "greedy":
        best_arms_value = np.max(q_values)
        best_arms = np.argwhere(q_values == best_arms_value).flatten()
        arm_choice = best_arms[np.random.randint(0, len(best_arms))]

    if strategy == "efirst":
        if share_elapsed < eta:  # explore phase
            arm_choice = np.random.randint(0, len(q_values))
        else:  # exploit phase
            best_arms_value = np.max(q_values)
            best_arms = np.argwhere(q_values == best_arms_value).flatten()
            arm_choice = best_arms[np.random.randint(0, len(best_arms))]

    if strategy == "egreedy":
        if np.random.random() > epsilon:
            best_arms_value = np.max(q_values)
            best_arms = np.argwhere(q_values == best_arms_value).flatten()
            arm_choice = best_arms[np.random.randint(0, len(best_arms))]
        else:
            arm_choice = np.random.randint(0, len(q_values))

    if strategy == "ucb":
        total_counts = np.sum(counts)
        q_values_ucb = q_values + np.sqrt(
            np.reciprocal(counts + 0.001) * 2 * math.log(total_counts + 1.0)
        )
        best_arms_value = np.max(q_values_ucb)
        best_arms = np.argwhere(q_values_ucb == best_arms_value).flatten()
        arm_choice = best_arms[np.random.randint(0, len(best_arms))]

    if strategy == "fracKUBE":
        total_counts = np.sum(counts)
        q_values_ucb = q_values + np.sqrt(
            2 * math.log(total_counts + 1.0) / (counts + 0.001)
        )
        q_values_to_cost_ratio = q_values_ucb / costs
        best_arms_value = np.max(q_values_to_cost_ratio)
        best_arms = np.argwhere(q_values_to_cost_ratio == best_arms_value).flatten()
        choice = best_arms[np.random.randint(0, len(best_arms))]
        arm_choice = choice

    if strategy == "thompson":
        draws = np.zeros(len(counts))
        for i in range(len(counts)):
            draws[i] = np.random.beta(success[i] + 1, failure[i] + 1)
        arm_choice = np.argmax(draws)

    # costy thompson
    if strategy == "thompsonBC":
        draws = np.zeros(len(counts))
        for i in range(len(counts)):
            draws[i] = np.random.beta(success[i] + 1, failure[i] + 1)
        # scale costs to be on [0, 1]
        scaled_costs = costs / np.sum(costs)
        rats = draws / scaled_costs
        arm_choice = np.argmax(rats)

    if strategy == "thompsonTopTwo":
        draws = np.zeros(len(counts))
        for i in range(len(counts)):
            draws[i] = np.random.beta(success[i] + 1, failure[i] + 1)
        I = np.argmax(draws)
        if np.random.binomial(1, psi) == 1:
            arm_choice = I
        else:
            J = None
            while J != I:
                for i in range(len(counts)):
                    draws[i] = np.random.beta(success[i] + 1, failure[i] + 1)
                J = np.argmax(draws)
            arm_choice = J

    if strategy == "kasySautmann":
        draws = np.zeros(len(counts))
        for i in range(len(counts)):
            draws[i] = np.random.beta(success[i] + 1, failure[i] + 1)
        exp_draws = draws * (1 - draws)
        arm_choice = np.argmax(exp_draws)

    if return_what == "arm_choice":
        return arm_choice
    else:
        if strategy in ["thompson", "thompsonBC", "thompsonTopTwo", "kasySautmann"]:
            return draws
        else:
            raise Exception(f"{strategy} does not return draws")


# %%
########  ##     ## ##    ##  ######  #### ##     ##
##     ## ##     ## ###   ## ##    ##  ##  ###   ###
##     ## ##     ## ####  ## ##        ##  #### ####
########  ##     ## ## ## ##  ######   ##  ## ### ##
##   ##   ##     ## ##  ####       ##  ##  ##     ##
##    ##  ##     ## ##   ### ##    ##  ##  ##     ##
##     ##  #######  ##    ##  ######  #### ##     ##


def sim_runner(
    budget,
    arm_means,
    costs,
    pay_after=False,
    bandits=["greedy", "random", "egreedy", "efirst", "ucb", "thompson", "fracKUBE"],
    ax=None,
    number_of_arms=10,
    number_of_pulls=30_000,
    cumulative=True,
    vax=True,
):
    run_lengths = []
    for bt in bandits:
        # initialise large arrays for best arm, reward, and pull sequences
        q_values = np.zeros(number_of_arms)
        counts = np.zeros(number_of_arms)
        success = np.zeros(number_of_arms)
        failure = np.zeros(number_of_arms)
        j = 0
        if budget:  # budgeted bandit
            B = budget  # reset budget to initial
            rewards = np.zeros(B * 10)
            while B >= 0:
                a = pick_arm(
                    q_values,
                    counts,
                    bt,
                    success,
                    failure,
                    costs,
                    share_elapsed=j / number_of_pulls,
                )
                reward = np.random.binomial(1, arm_means[a])
                rewards[j] = (
                    reward  # append to sequence of rewards for eventual plotting
                )
                counts[a] += 1.0
                q_values[a] += (reward - q_values[a]) / counts[a]
                success[a] += reward
                failure[a] += 1 - reward
                j += 1
                if pay_after == False:
                    B -= costs[a]
                else:
                    if reward == 1:
                        B -= costs[a]
        else:  # specified number of runs
            rewards = np.zeros(number_of_pulls)
            for j in range(number_of_pulls):
                a = pick_arm(
                    q_values,
                    counts,
                    bt,
                    success,
                    failure,
                    costs=None,
                    share_elapsed=j / number_of_pulls,
                )
                reward = np.random.binomial(1, arm_means[a])
                rewards[j] = (
                    reward  # append to sequence of rewards for eventual plotting
                )
                counts[a] += 1.0
                q_values[a] += (reward - q_values[a]) / counts[a]
                success[a] += reward
                failure[a] += 1 - reward
        run_lengths.append(j)  # keep tally of how many iterations each algo ran
        ys = rewards[:j]
        xs = np.arange(1, len(ys) + 1)
        if cumulative:
            ax.plot(xs, np.cumsum(ys), label=bt, alpha=0.8, linewidth=1.5)
            if vax:
                ax.set_ylabel("Total Obs Collected")
        else:
            ax.plot(xs, np.cumsum(ys) / xs, label=bt, alpha=0.8, linewidth=1.5)
            if vax:
                ax.set_ylabel("Average Reward")
            ax.set_ylim((0, 1.1))
    if budget:
        ax.set_title(f"Budget = {budget}")
    ax.set_xlabel("Steps")
    ax.set_xlim((10, max(run_lengths)))


# %%
###    ########  ##     ##  ######  ########  #######
## ##   ##     ## ###   ### ##    ## ##       ##     ##
##   ##  ##     ## #### #### ##       ##       ##     ##
##     ## ########  ## ### ##  ######  ######   ##     ##
######### ##   ##   ##     ##       ## ##       ##  ## ##
##     ## ##    ##  ##     ## ##    ## ##       ##    ##
##     ## ##     ## ##     ##  ######  ########  ##### ##


def arm_sequence(
    budget,
    bandit,
    arm_means,
    costs=None,
    ret=False,
    pay_after=False,
    number_of_pulls=10_000,
    number_of_arms=10,
):
    run_lengths = []
    # initialise large arrays for best arm, reward, and pull sequences
    q_values = np.zeros(number_of_arms)
    counts = np.zeros(number_of_arms)
    success = np.zeros(number_of_arms)
    failure = np.zeros(number_of_arms)
    j = 0
    if budget:  # budgeted bandit
        B = budget  # reset budget to initial
        rewards = np.zeros(B * 10)
        pull_mat = np.zeros(number_of_arms)
        while B >= 0:
            a = pick_arm(
                q_values,
                counts,
                bandit,
                success,
                failure,
                costs=costs,
                share_elapsed=j / number_of_pulls,
            )
            reward = np.random.binomial(1, arm_means[a])
            rewards[j] = reward  # append to sequence of rewards for eventual plotting
            counts[a] += 1.0
            pull_mat = np.vstack([pull_mat, counts])
            q_values[a] += (reward - q_values[a]) / counts[a]
            success[a] += reward
            failure[a] += 1 - reward
            if pay_after == False:
                B -= costs[a]
            else:
                if reward == 1:
                    B -= costs[a]
            j += 1
    else:  # specified number of runs
        rewards = np.zeros(number_of_pulls)
        pull_mat = np.zeros((number_of_pulls, number_of_arms))
        for j in range(number_of_pulls):
            a = pick_arm(
                q_values,
                counts,
                bandit,
                success,
                failure,
                costs=None,
                share_elapsed=j / number_of_pulls,
            )
            reward = np.random.binomial(1, arm_means[a])
            rewards[j] = reward  # append to sequence of rewards for eventual plotting
            counts[a] += 1.0
            pull_mat[j, :] = counts
            q_values[a] += (reward - q_values[a]) / counts[a]
            success[a] += reward
            failure[a] += 1 - reward
    return [pull_mat, rewards, counts]


# %% plotter for pull sequence
########  ##     ## ##       ##        ######  ########  #######
##     ## ##     ## ##       ##       ##    ## ##       ##     ##
##     ## ##     ## ##       ##       ##       ##       ##     ##
########  ##     ## ##       ##        ######  ######   ##     ##
##        ##     ## ##       ##             ## ##       ##  ## ##
##        ##     ## ##       ##       ##    ## ##       ##    ##
##         #######  ######## ########  ######  ########  ##### ##


def pull_sequence(
    pull_mat, title, ax, means, costs=None, costly=False, xticks=True, logx=True
):
    # store index of best arm
    if costly:
        best_arm = np.argmax(means / costs)
    else:
        best_arm = np.argmax(means)
    # plot each arm
    for j in range(pull_mat.shape[1]):
        ys = pull_mat[:, j]
        xs = np.arange(1, len(ys) + 1)
        if costly:
            lab = f"{j} ({round(means[j], 2)}, {round(costs[j], 2)})"
        else:
            lab = f"{j} ({round(means[j], 2)})"
        if j == best_arm:
            lab = "".join([r"\textbf{", lab, r"}"])
            ax.plot(xs, ys, label=lab, alpha=0.8, linewidth=2.5)
        else:
            ax.plot(xs, ys, label=lab, alpha=0.8, linewidth=1.5)
    if logx == True:
        ax.set_xscale("log")
    ax.set_title(title)
    if xticks == False:
        ax.axes.get_xaxis().set_ticklabels([])


# %% best arm runtime
########  ########  ######  ########    ###    ########  ##     ##
##     ## ##       ##    ##    ##      ## ##   ##     ## ###   ###
##     ## ##       ##          ##     ##   ##  ##     ## #### ####
########  ######    ######     ##    ##     ## ########  ## ### ##
##     ## ##             ##    ##    ######### ##   ##   ##     ##
##     ## ##       ##    ##    ##    ##     ## ##    ##  ##     ##
########  ########  ######     ##    ##     ## ##     ## ##     ##


def best_arm(arm_means, bandit, M=1_000, conf_threshold=0.8, beta=0.6):
    # init arrays
    number_of_arms = len(arm_means)
    success = np.zeros(number_of_arms)
    failure = np.zeros(number_of_arms)
    uncertain = True
    # best arm posterior matrix
    alpha_post_mat = np.matrix(np.zeros(number_of_arms))
    j = 0
    # while loop for termination when confidence level reached
    while uncertain:
        post_draws = np.zeros(number_of_arms)
        b = []
        # init random variables
        for i in range(number_of_arms):
            b.append(stats.beta(success[i] + 1, failure[i] + 1))
        ############################
        # montecarlo alphas
        ############################
        draws = np.zeros((M, number_of_arms))
        # draws for estimating α
        for i in range(number_of_arms):
            # draw from posterior
            draws[:, i] = b[i].rvs(size=M)
        best_arm_vec = np.argmax(draws, axis=1)
        # create discrete distribution of argmaxes
        final_counts = np.array(
            [np.count_nonzero(best_arm_vec == i) for i in range(number_of_arms)]
        )
        # final_counts = np.unique(best_arm_vec, return_counts = True)[1]
        alphas = final_counts / M
        # stack
        alpha_post_mat = np.vstack((alpha_post_mat, alphas))
        current_best_arm_prob = np.max(alphas)
        # choose arm
        if bandit == "Thompson":
            a = np.argmax(alphas)
        elif bandit == "ThompsonTopTwo":
            if np.random.uniform() < beta:
                # pick current best arm
                a = np.argmax(alphas)
            else:
                # pick second highest arm
                a = np.argsort(-alphas)[1]
        # pull arm and update rewards
        reward = np.random.binomial(1, arm_means[a])
        success[a] += reward
        failure[a] += 1 - reward
        # this flips to true once one of the arm posterior alphas exceeds the threshold
        uncertain = current_best_arm_prob < conf_threshold
        j += 1
        # abort search after 5k runs
        if j >= 2_000:
            print(f"aborted")
            uncertain = False
    return alpha_post_mat


# %%
########     ###    ##    ## ########  #### ########  ######   #######   ######  ########
##     ##   ## ##   ###   ## ##     ##  ##     ##    ##    ## ##     ## ##    ##    ##
##     ##  ##   ##  ####  ## ##     ##  ##     ##    ##       ##     ## ##          ##
########  ##     ## ## ## ## ##     ##  ##     ##    ##       ##     ##  ######     ##
##     ## ######### ##  #### ##     ##  ##     ##    ##       ##     ##       ##    ##
##     ## ##     ## ##   ### ##     ##  ##     ##    ##    ## ##     ## ##    ##    ##
########  ##     ## ##    ## ########  ####    ##     ######   #######   ######     ##


def rep_bandit_cost(
    bandit, arm_means, pay_levels, B, target_shares, gamma=2, payafter=False
):
    # if bandit not in ['thompsonBC', 'random', 'egreedy', 'thompson']:
    #     raise ValueError(f"{bandit} Bandit not supported")
    number_of_arms = arm_means.shape[0]
    # init costs and shares
    q_values, counts, success, failure = [np.zeros(number_of_arms) for _ in range(4)]
    # init share matrix
    shares = np.zeros((1, len(target_shares)))
    # initialise costs to pay level_j
    init_costs = np.tile(pay_levels, 2)
    costs = np.copy(init_costs)
    # init cost matrix
    costmat = costs[np.newaxis, :]
    # run until money runs out
    if B:  # budgeted bandit
        b = B  # initial budget
        i = 0
        while b >= 0:
            # pick new arm
            a = pick_arm(
                q_values, counts, bandit, success, failure, costs, share_elapsed=1
            )
            # return reward
            reward = np.random.binomial(1, arm_means[a])
            q_values[a] += (reward - q_values[a]) / counts[a]
            success[a] += reward
            failure[a] += 1 - reward
            # update tally of responses from arm
            counts[a] += 1.0
            # group shares
            id = len(pay_levels)
            share_a = np.sum(success[0:id]) / np.sum(
                success
            )  # current share of group a in sample
            if np.isnan(share_a):  # beginning - set to 0
                shares = np.vstack([shares, np.array([0, 0])])
                costmat = np.vstack([costmat, init_costs])
            else:  # once shares exist, start scaling
                shares = np.vstack([shares, np.array([share_a, 1 - share_a])])
                # revise costs for group A
                gap_a = share_a - target_shares[0]  # extent of overshooting
                # costs increase over time with overshooting
                scalecosts = ((1 + ((B - b) / B) * gap_a) ** gamma) * init_costs[:id]
                # truncate negatives
                costs[:id] = np.maximum(scalecosts, 0.01)
                # store cost
                costmat = np.vstack([costmat, costs])
            # increment counter
            i += 1
            # deduct budget
            if payafter:
                if reward == 1:
                    b -= init_costs[a]
            else:
                b -= init_costs[a]
    # return shares and costs matrix
    return [shares, costmat, counts]


# %%
def rep_bandit_rake(
    bandit, arm_means, pay_levels, B, target_shares, payafter=False, alpha=0.5
):
    number_of_arms = arm_means.shape[0]
    # init costs and shares
    q_values, counts, success, failure = [np.zeros(number_of_arms) for _ in range(4)]
    # init share matrix
    shares = np.zeros((1, len(target_shares)))
    # loss matrix
    losses = np.zeros((1, number_of_arms))
    # initialise costs to pay level_j
    costs = np.tile(pay_levels, 2)
    # run until money runs out
    b = B  # initial budget
    i = 0
    while b >= 0:
        if bandit == "KUBE":
            total_counts = np.sum(counts)
            q_values_ucb = q_values + np.sqrt(
                2 * math.log(total_counts + 1.0) / (counts + 0.001)
            )
            rats = q_values_ucb / costs
        elif bandit == "ThompsonBC":
            # thompson part
            draws = np.zeros(len(counts))
            for i in range(len(counts)):
                draws[i] = np.random.beta(success[i] + 1, failure[i] + 1)
            # scale costs to be on [0, 1]
            scaled_costs = costs / np.sum(costs)
            rats = draws / scaled_costs
        # weights
        id = len(pay_levels)
        share_a = np.sum(success[0:id]) / np.sum(
            success
        )  # current share of group a in sample
        if np.isnan(share_a):  # beginning - set to 0
            shares = np.vstack([shares, np.array([0, 0])])
            a = np.argmax(rats)
        else:  # once shares exist, start scaling
            shares = np.vstack([shares, np.array([share_a, 1 - share_a])])
            # revise costs for group A
            gap_a = target_shares[0] / share_a - 1  # extent of overshooting
            # choose arm
            wted_loss = alpha * rats - (1 - alpha) * share_a * np.log(share_a)
            a = np.argmax(wted_loss)
            losses = np.vstack([losses, wted_loss])
        # return reward
        reward = np.random.binomial(1, arm_means[a])
        # update tally of responses from arm
        counts[a] += 1.0
        q_values[a] += (reward - q_values[a]) / counts[a]
        success[a] += reward
        failure[a] += 1 - reward
        # increment counter
        i += 1
        # deduct budget
        if payafter:
            if reward == 1:
                b -= costs[a]
        else:
            b -= costs[a]
    # return shares and costs matrix
    return [shares, counts, losses]


# %%
