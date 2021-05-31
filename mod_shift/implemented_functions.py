import numpy as np

def rho_clipped_linear(d, max_d):
    return 1 - (1 - d / max_d).relu()


def rho_clipped_linear_derivative(d, max_d):
    return 0.5 * ((max_d - d).sign() + 1) / max_d  # step function in both pytorch and pykeops


def rho_clipped_parabola(d, max_d):
    return 1 - (1 - (d/max_d)**2).relu()


def rho_clipped_parabola_derivative(d, max_d):
    return 2 * d/max_d**2 * ((max_d - d).sign() + 1)


def rho_schoenberg(d, max_d):
    return (1 - (-d**2/(max_d/2)**2).exp()).sqrt()


def rho_schoenberg_derivative(d, max_d):
    d = d + 0.001  # torch has some numerical problem around 0
    return 1/rho_schoenberg(d, max_d) * 4 * d / max_d**2 * (-d**2/(max_d/2)**2).exp()


def w_clipped_linear(d, beta):
    return (2-d / beta).relu() - 1


def w_clipped_linear_derivative(d, beta):
    return - 0.5 * ((2*beta - d).sign() + 1) / beta


def w_clipped_cos(d, beta):
    return (np.pi / 2 * d / beta).cos() * 0.5 * ((2*beta - d).sign() + 1) - 0.5 * ((d-2*beta).sign() + 1)


def w_clipped_cos_derivative(d, beta):
    return - (np.pi / 2 * d / beta).sin() * 0.5 * ((2*beta - d).sign() + 1) * np.pi / (2 * beta)




