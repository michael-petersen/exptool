#from .context import exptool


import exptool

import pkg_resources


# do a test of the importing capabilities
import exptool.io.psp_io as psp_io

data_file = pkg_resources.resource_filename('exptool', 'io/initial001dump.dat')

O = psp_io.Input(data_file,'star')

print(O.time)


