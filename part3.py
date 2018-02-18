"""
For part III of this assignment, you'll implement a model from scratch has a vague relationship to the Weiss et al.
ambiguous-motion model.  The model will try to infer the direction of motion from some observations.
I'll assume that a rigid motion is being observed involving an object that has two distictinctive visual features.
The figure below shows a snapshot of the object at two nearby points in time.
The distinctive features are the red triangle and blue square.  Let's call them R and B for short.
Because the features are distinctive, determining the correspondence between features at two snapshots in time is
straightforward, and the velocity vector can be estimated.
Assume that these measurements are noisy however, such that the x and y components of the velocity are each corrupted
by independent, mean zero Gaussian noise with standard deviation σ.
Thus the observation consists of four real valued numbers: Rx, Ry, Bx, and By -- respectively,
the red element x and y velocities, and the blue element x and y velocities.
The goal of the model is to infer the direction of motion.

To simplify, let's assume there are only four directions: up, down, left, and right.
Further, the motions will all be one unit step.
Thus, if the motion is to the right, then noise-free observations would be:  Rx=1, Ry=0, Bx=1, By=0.
If the motion is down, then the noise-free observations would be: Rx=0, Ry=-1, Bx=0, By=-1.

Formally, the model must compute P(Direction | Rx, Ry, Bx, By).
"""

from scipy.stats import norm

DIRECTIONS = ['up', 'right', 'down', 'left']


def get_prior():
    """
    Assuming no noise in the direction
    Assuming the directions are up, down, left and right
    Prior = 1/4
    :return: {float} prior over direction
    """
    prior = 0.25
    return prior


def get_likelihood(direction, obs, likelihood):
    """
    Get P( velocity | direction)
    :return: P( velocity | direction) for both the shapes.
    """

    return likelihood['x_{}'.format(direction)].pdf(obs['x_red']),\
           likelihood['y_{}'.format(direction)].pdf(obs['y_red']), \
           likelihood['x_{}'.format(direction)].pdf(obs['x_blue']), \
           likelihood['y_{}'.format(direction)].pdf(obs['y_blue'])


def init_likelihood(sig):
    gauss_zero = norm(0, sig)
    gauss_one = norm(1, sig)
    gauss_minus_one = norm(-1, sig)


    likelihood = dict()

    likelihood['x_up'] = gauss_zero
    likelihood['x_right'] = gauss_one
    likelihood['x_left'] = gauss_minus_one
    likelihood['x_down'] = gauss_zero

    likelihood['y_up'] = gauss_one
    likelihood['y_down'] = gauss_minus_one
    likelihood['y_left'] = gauss_zero
    likelihood['y_right'] = gauss_zero

    return likelihood


def task1(sig, obs):
    prior = get_prior()
    velocity_given_dir = init_likelihood(sig)
    posterior = {}
    for direction in DIRECTIONS:
        likelihood_x_red, likelihood_y_red, likelihood_x_blue, likelihood_y_blue = get_likelihood(direction, obs,
                                                                                                  velocity_given_dir)

        posterior[direction] = likelihood_x_red * likelihood_y_red * likelihood_x_blue * likelihood_y_blue * prior

    norm_factor = sum(posterior.values())
    posterior = {key: posterior[key] / norm_factor for key in DIRECTIONS}

    return max(posterior, key=posterior.get)


if __name__ == "__main__":
    print(task1(1, {'x_red': 0.75, 'y_red': -0.6, 'x_blue': 1.4, 'y_blue': -0.2}))
    print(task1(5, {'x_red': 0.75, 'y_red': -0.6, 'x_blue': 1.4, 'y_blue': -0.2}))
