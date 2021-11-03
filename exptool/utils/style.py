'''
STYLE FILE FOR PLOTTING USING EXPTOOL



'''



import matplotlib as mpl


# want a secondary style that allows for adjusting line thicknesses
# (e.g. a macro parameter)

stylepar = 'medium'

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.serif'] = 'Helvetica'
mpl.rcParams['figure.figsize'] = (8,6)

mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['axes.labelsize'] = 24

cmap = mpl.cm.inferno

# change some plotting parameters
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.serif'] = 'Helvetica'
mpl.rcParams['font.weight'] = 'medium'

mpl.rcParams['axes.linewidth'] = 1.0
mpl.rcParams['xtick.major.width'] = 1.0
mpl.rcParams['xtick.minor.width'] = 0.5
mpl.rcParams['xtick.minor.visible'] = True

mpl.rcParams['ytick.major.width'] = 1.0
mpl.rcParams['ytick.minor.width'] = 0.5
mpl.rcParams['ytick.minor.visible'] = True

mpl.rcParams['figure.figsize'] = (5,3.5)
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['axes.labelsize'] = 12

if stylepar == 'bold':

    mpl.rcParams['axes.linewidth'] = 2.0
    mpl.rcParams['xtick.major.width'] = 2.0
    mpl.rcParams['xtick.minor.width'] = 1.0
    mpl.rcParams['xtick.minor.visible'] = True

    mpl.rcParams['ytick.major.width'] = 2.0
    mpl.rcParams['ytick.minor.width'] = 1.0
    mpl.rcParams['ytick.minor.visible'] = True


elif stylepar == 'medium':
    mpl.rcParams['font.weight'] = 'medium'
    mpl.rcParams['axes.linewidth'] = 1.5
    mpl.rcParams['xtick.major.width'] = 1.5
    mpl.rcParams['xtick.minor.width'] = 0.75
    mpl.rcParams['xtick.minor.visible'] = True

    mpl.rcParams['ytick.major.width'] = 1.5
    mpl.rcParams['ytick.minor.width'] = 0.75
    mpl.rcParams['ytick.minor.visible'] = True



else:
    print('exptool.utils.style: No style parameter accepted.')

    

