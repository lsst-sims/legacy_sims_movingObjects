import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == '__main__':
    resid_file = 'residuals.dat'
    d = pd.read_table(resid_file, delim_whitespace=True)
    # Set up colors.
    colors = {}
    for ncoeff in d.ncoeff.unique():
        colors[ncoeff] = {}
        for ngran in d.ngran.unique():
            if ngran == 32:
                colors[ncoeff][ngran] = [0, 0, 0]
            elif ngran == 64:
                colors[ncoeff][ngran] = [1, 0, 0]
            elif ngran == 128:
                colors[ncoeff][ngran] = [0, 0, 1]
            else:
                colors[ncoeff][ngran] = np.random.rand(1, 3)
            if ncoeff == 7:
                colors[ncoeff][ngran][1] = 1
            elif ncoeff == 28:
                colors[ncoeff][ngran][1] = 0.5
    # Set up marker style:
    markerstyle = {}
    for otype in d.otype.unique():
        if otype == 0:
            markerstyle[otype] = 'o'
        elif otype == 1:
            markerstyle[otype] = 'x'
        elif otype == 2:
            markerstyle[otype] = '+'
        else:
            markerstyle[otype] = '.'
    # Make plots.
    plt.figure()
    ngranPlot = d.ngran.unique() 
    ngranPlot = ([64, 128])
    ncoeffPlot = d.ncoeff.unique()
    ncoeffPlot = ([14])
    for ngran in ngranPlot:
        for ncoeff in ncoeffPlot:
            for otype in d.otype.unique():
                sub = d.query('(ncoeff == @ncoeff) and (ngran == @ngran) and (otype == @otype)')
                plt.plot(sub['resid'], sub['length'],
                         color=colors[ncoeff][ngran],
                         linestyle='-', marker=markerstyle[otype],
                         label='%d_%d_%d' % (ncoeff, ngran, otype))
    plt.xlabel('Residuals (mas)')
    #plt.xlabel('Calculation time (seconds)')
    plt.ylabel('Segment length (days)')
    plt.legend(loc=(0.95, 0), fontsize='smaller')
    plt.xlim(0, 1000)
    #plt.ylim(0, 2)
    plt.show()
