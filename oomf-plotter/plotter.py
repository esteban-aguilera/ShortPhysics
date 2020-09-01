import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------------
FILENAME = 'test'


# --------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------
def main():
    # add variables to the file
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--filename', '-f', action='store', type=str,
                        default=FILENAME)
    args = parser.parse_args()

    # obtain arguments
    filename = args.filename

    # load data
    pos, mag = load_data(filename)

    # plot loaded data
    fig, ax = plot_magnetization(pos, mag)

    fig.tight_layout()
    # show the created plot
    plt.show()


# --------------------------------------------------------------------------------
# functions
# --------------------------------------------------------------------------------
def load_data(filename):
    """Loads data.  Depending on the extension of the file, different loading
    protocols are executed.  If filename has no extension, the file will be assumed
    to come from OOMF and load_oomf(filename) will be executed.

    Parameters
    ----------
    filename: str
        Name of the file where the data is located.

    Returns
    -------
    pos: np.array
        Array with the position's data.

    mag: np.array
        Array with the magnetization's data.
    """
    print(f'loading {filename} ...')
    if('.' in filename[-5:]):  # check if filename has an extension.
        if(filename[-4:] == '.csv'):  # filename corresponds to a csv.
            data = load_csv(filename)
        else:
            raise ValueError(f'{filename} has an invalid extension.')
    else:  # if there is no extension, data was extracted directly from OOMF.
        data = load_oomf(filename)

    return data


def load_csv(filename, header=False):
    """Loads a .csv file and extracts from it the positions and magnetizations.
    In the file, the columns should be structured as [Rx, Ry, Rz, Mx, My, Mz]

    Parameters
    ----------
    filename: str
        Name of the file where the data is located.  It should be a .csv file

    header: bool (default: False)
        Specifies if the file has a header with the column names on them.

    Returns
    -------
    pos: np.array
        Array with the position's data.

    mag: np.array
        Array with the magnetization's data.
    """
    if(filename[-4:] != '.csv'):
        raise ValueError('filename does not have .csv extension.')

    if(header is False):
        # There is no header.
        df = pd.read_csv(filename, header=None)
    else:
        df = pd.read_csv(filename)

    data = np.array(df)

    # check data integrity
    if(data.shape[1] != 6):
        raise ValueError('Data in the .csv is not structured correctly.  File must have 6 columns.')

    pos = data[:, :3]  # first three columns must be the position.
    mag = data[:, 3:]  # remaining three columns must be the magnetization.

    return pos, mag


def load_oomf(filename):
    """Loads the file exported from OOMF and extracts from it the positions
    and magnetizations.  In the file, the columns should be structured as
    [Rx, Ry, Rz, Mx, My, Mz].

    Parameters
    ----------
    filename: str
        Name of the file where the data is located.

    Returns
    -------
    pos: np.array
        Array with the position's data.

    mag: np.array
        Array with the magnetization's data.
    """
    data = np.loadtxt(filename)

    # check data integrity
    if(data.shape[1] != 6):
        raise ValueError( 'Data in the .csv is not structured correctly.  File must have 6 columns.')

    pos = data[:, :3]  # first three columns are the positions.
    mag = data[:, 3:]  # remaining three columns are the magnetizations.

    return pos, mag


def plot_magnetization(pos, mag, figax=None):
    """Plots the magnetization data

    Parameters
    ----------
    pos: np.array
        array with the positions data.  The rows must correspond to different sites and the
        columns are the components X, Y, Z.
    
    mag: np.array
        array with the magnetization data.  The rows must correspond to different sites and
        the columns are the components X, Y, Z.
    
    figax: (matplotlib.pyplot.Figure, matplotlib.pyplot.Axis)
        Tuple of figure and axis.  Used in case the user want to combine this plot with a
        previously created plot.

    Returns
    -------
    fig: matplotliy.pyplot.Figure
        Figure used to plot the magnetization.

    ax: matplotliy.pyplot.Axis
        Axis used to plot the magnetization.
    """
    if(figax is None):
        # create figure and axis.
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        fig, ax = figax

    # filter the data that has total magnetization equal to zero.
    mask = np.where(mag.any(axis=1))[0]
    pos, mag = pos[mask,:], mag[mask,:]

    # extract the data from position and magnetization.
    X, Y, Z = pos[:,0], pos[:,1], pos[:,2]
    Mx, My, Mz = mag[:,0], mag[:,1], mag[:,2]
    
    # delete unused variable
    del Z

    # normalize Mz to obtain colors.
    color = 2 * Mz / (np.max(Mz) - np.min(Mz))

    cr = ax.scatter(X, Y, s=1, c=color)

    # create quiver plot
    ax.quiver(X, Y, Mx, My, color, units='xy')

    # add colorbar.  The color range will be the one used in the quiver plot.
    fig.colorbar(cr)

    # return figure and axis
    return fig, ax


# --------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------
if __name__ == "__main__":  # if this is the executed file, execute main()
    main()
