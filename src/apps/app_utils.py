import numpy as np

def update_lr(current_round: int, total_rounds, start_lr: float, end_lr: float):
    """Applies exponential learning rate decay using the defined start_lr and end_lr.
     The basic eq is as follows:
    init_lr * exp(-round_i*gamma) = lr_at_round_i
     A more common one is :
        end_lr = start_lr*gamma^total_rounds"""

    # first we need to compute gamma, which will later be used
    # to obtain the lr for the current round

    gamma = np.power(end_lr / start_lr, 1.0 / total_rounds)
    current_lr = start_lr * np.power(gamma, current_round)
    return current_lr, gamma