import numpy as np

def rho_clipped_linear(input, max_d):
    return 1 - (1-input/max_d).relu()


def rho_clipped_parabola(input, max_d):
    return 1 - (1 - (input/max_d)**2).relu()


def rho_schoenberg(input, max_d):
    return (1 - (-input**2/(max_d/2)**2).exp()).sqrt()


def w_clipped_linear(input, beta):
    return (2-input / beta).relu() - 1


def w_clipped_cos(input, beta):
    return (np.pi / 2 * input).cos() * (2*beta - input).step() - 1 * (input-2*beta).step()



