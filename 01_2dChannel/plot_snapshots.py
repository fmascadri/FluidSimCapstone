"""
Plot 2D cartesian snapshots.
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

    # Plot writes
    with h5py.File(filename, mode='r') as file:
        
        # Data setup
        U = file['tasks']['velocity']
        u_x = U[:,0,:,:] # select x component of velocity
        u_x_time_averaged = np.mean(u_x,0) # select time axis to average over
        u_x_averaged_in_x_direction = np.mean(u_x_time_averaged,0) # select x axis to average over
        z_height = np.linspace(0,1,np.shape(u_x_averaged_in_x_direction)[0])
        
        #Plotting
        fig, ax = plt.subplots(figsize=(6,4), layout='constrained')
        mean_u = ax.plot(u_x_averaged_in_x_direction, z_height, '-b', linewidth=1.5)
        ax.legend(handles=mean_u, labels=[r'$\bar U$ [m/s]'])
        ax.set_xlabel('U [m/s]')
        ax.set_ylabel('z [m]')
        ax.set_ylim(-0.05,1.05)
        ax.set_xlim(0, 0.04)
        # Add time title
        title = r"Mean velocity profile averaged in time, Re$_\tau$ = 180"
        fig.suptitle(title, ha='center')
        
        #Add channel walls to plot
        verts_bottom = [
            (0,-0.05), # left bottom
            (0,0), # left top
            (4,0), # right top
            (4,-0.05), # right bottom
            (0,0), # ignored
        ]
        verts_top = [
            (0,1), # left bottom
            (0,1.05), # left top
            (4,1.05), # right top
            (4,1), # right bottom
            (0,0), # ignored
        ]

        codes = [
            Path.MOVETO,
            Path.LINETO,
            Path.LINETO,
            Path.LINETO,
            Path.CLOSEPOLY,
        ]

        bottom_path = Path(verts_bottom, codes)
        bottom_patch = patches.PathPatch(bottom_path, facecolor='grey', hatch='/', linewidth=1)
        ax.add_patch(bottom_patch)

        top_path = Path(verts_top, codes)
        top_patch = patches.PathPatch(top_path, facecolor='grey', hatch='/', linewidth=1)
        ax.add_patch(top_patch)
        
        # Show plot for interactive look
        plt.show()

        # Save figure
        savename = "Mean velocity profile"
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