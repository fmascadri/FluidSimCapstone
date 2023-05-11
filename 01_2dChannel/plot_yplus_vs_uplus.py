"""
Plot yplus vs uplus velocity profiles.
Usage:
    plot_snapshots.py <files>... [--output=<dir>]

Options:
    --output=<dir>  Output directory [default: ./frames]
    
"""


import h5py
import numpy as np
from matplotlib.path import Path
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from dedalus.extras import plot_tools


def main(filename, start, count, output):
    """Save plot of specified tasks for given range of analysis writes."""

    with h5py.File(filename, mode='r') as file:

        Nx = 256
        Nz = 64
        delta = 0.5
        Retau = 180
        nu = 1.8e-06
        utau = Retau * nu / delta

        # Data setup
        U = file['tasks']['velocity']
        u_x = U[:,0,:,:] # select x component of velocity
        u_x_time_averaged = np.mean(u_x,0) # select time axis to average over
        u_x_averaged_in_x_direction = np.mean(u_x_time_averaged,0) # select x axis to average over
        z_height = np.linspace(0,1,np.shape(u_x_averaged_in_x_direction)[0])

        # scale to wall units        
        Uplus = u_x_averaged_in_x_direction / utau
        zplus = z_height * utau / nu

        # slice up to channel half-height
        Uplus_halfheight = Uplus[0:int(Nz/2)]
        zplus_halfheight = zplus[0:int(Nz/2)]
        
        #Plotting
        fig, ax = plt.subplots(figsize=(6,4), layout='constrained')
        Retau180 = ax.semilogx(zplus_halfheight, Uplus_halfheight, '-b', linewidth=1.5)
        ax.legend(handles=Retau180, labels=[r'$Re_\tau = 180$'])
        ax.set_xlabel(r'$y^+$')
        ax.set_ylabel(r'$U^+$')

        # Add time title
        title = "Mean velocity profile in wall units"
        fig.suptitle(title, ha='center')

        # Compare with KMM data
        #retau180_refdata = np.loadtxt("./kmmdata/retau180.csv", delimiter=",", dtype=float)
        #ax.semilogx(retau180_refdata, '--k')

        # Show plot for interactive look
        plt.show()

        # Save figure
        savename = "yplusvsuplus"
        savepath = output.joinpath(savename)
        fig.savefig(str(savepath), dpi=300)
        fig.clear()
        plt.close(fig)


if __name__ == "__main__":

    import pathlib
    from docopt import docopt
    from dedalus.tools import logging
    from dedalus.tools import post
    from dedalus.tools.parallel import Sync

    args = docopt(__doc__)

    output_path = pathlib.Path(args['--output']).absolute()
    # Create output directory if needed
    with Sync() as sync:
        if sync.comm.rank == 0:
            if not output_path.exists():
                output_path.mkdir()
    post.visit_writes(args['<files>'], main, output=output_path)