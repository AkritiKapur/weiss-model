from data.data import exampleA1, exampleA2, exampleB1, exampleB2, exampleC1, exampleC2
from matplotlib import colors

import matplotlib.pyplot as plt
import numpy as np

LOWER_LIMIT = 1
UPPER_LIMIT = 10
VELOCITY = [-2, -1, 0, 1, 2]


def get_colormapped_snap(snap_1, snap_2):
    mapped_image = [[3 for _ in range(LOWER_LIMIT - 1, UPPER_LIMIT)]
                    for _ in range(LOWER_LIMIT - 1, UPPER_LIMIT)]

    for i in range(LOWER_LIMIT - 1, UPPER_LIMIT):
        for j in range(LOWER_LIMIT - 1, UPPER_LIMIT):
            if snap_1[i][j] == 1 and snap_2[i][j] == 1:
                mapped_image[i][j] = 2
            elif snap_1[i][j] == 1:
                mapped_image[i][j] = 0
            elif snap_2[i][j] == 1:
                mapped_image[i][j] = 1

    return mapped_image


def plot_motion_graphs_examples(snap_1, snap_2):
    x_range = np.arange(LOWER_LIMIT - 1, UPPER_LIMIT)
    y_range = x_range

    cmap = colors.ListedColormap(['red', 'green', 'yellow', 'black'])
    colormapped_image = get_colormapped_snap(snap_1, snap_2)
    plt.imshow(colormapped_image, cmap=cmap)

    plt.grid()
    plt.xticks(x_range, np.arange(LOWER_LIMIT, UPPER_LIMIT + 1))
    plt.yticks(y_range, np.arange(LOWER_LIMIT, UPPER_LIMIT + 1))
    plt.show()


def get_likelihood(snap_1, snap_2):
    likelihood_matrix = []

    for x in VELOCITY:
        ys = []
        for y in VELOCITY:
            likelihood = 1
            for p_x in range(LOWER_LIMIT - 1, UPPER_LIMIT):
                for p_y in range(LOWER_LIMIT - 1, UPPER_LIMIT):
                    x_later = p_x + x
                    y_later = p_y + y
                    if x_later < LOWER_LIMIT - 1 or y_later < LOWER_LIMIT - 1 or \
                        x_later >= UPPER_LIMIT or y_later >= UPPER_LIMIT:
                        intensity_diff = np.square(snap_1[p_x][p_y])
                    else:
                        intensity_diff = np.square(snap_1[p_x][p_y] - snap_2[x_later][y_later])
                    likelihood *= np.exp(-intensity_diff)

            ys.append(np.log(likelihood))
        likelihood_matrix.append(ys)

    return likelihood_matrix


def task1():
    matrix = get_likelihood(exampleA1, exampleA2)
    create_contour(matrix)
    matrix = get_likelihood(exampleB1, exampleB2)
    create_contour(matrix)
    matrix = get_likelihood(exampleC1, exampleC2)
    create_contour(matrix)


def create_contour(likelihood):
    z = plt.imshow(likelihood)
    plt.colorbar(z)
    v = plt.axis()
    plt.axis(v)
    plt.xticks(np.arange(len(VELOCITY)), VELOCITY)
    plt.yticks(np.arange(len(VELOCITY)), VELOCITY)
    plt.show()


def get_priors():
    priors = []
    for x in VELOCITY:
        ys = []
        for y in VELOCITY:
            ys.append(np.exp(-(x*x + y*y)/2))
        priors.append(ys)

    priors = np.array(priors)
    print(priors)
    denom = np.sum(priors)
    priors = priors / denom
    priors = np.log(priors)
    print(priors)
    create_contour(priors)
    return priors


def get_posterior(snap_1, snap_2, priors):
    likelihood = get_likelihood(snap_1, snap_2)

    posterior = []
    for i in range(len(likelihood)):
        ys = []
        for j in range(len(likelihood[0])):
            ys.append(likelihood[i][j] + priors[i][j])
        posterior.append(ys)

    # posterior = np.array(posterior)
    # denom = np.sum(posterior)
    # posterior = posterior / denom
    return posterior


def task2():
    priors = get_priors()
    matrix = get_posterior(exampleA1, exampleA2, priors)
    create_contour(matrix)
    matrix = get_posterior(exampleB1, exampleB2, priors)
    create_contour(matrix)
    matrix = get_posterior(exampleC1, exampleC2, priors)
    create_contour(matrix)


if __name__ == "__main__":
    # plot_motion_graphs_examples(exampleA1, exampleA2)
    # plot_motion_graphs_examples(exampleB1, exampleB2)
    # plot_motion_graphs_examples(exampleC1, exampleC2)
    task2()


