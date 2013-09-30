#!/usr/bin/env python

# This script is used for multiplying already written tabular forces by a conversion factor.
# It relies on the plot_forces.py script. Execute with -h flag for usage information.
#
#
#

from plot_list import * 
import argparse, os

parser = argparse.ArgumentParser(description='Given a directory which is the result of a ForcePy run and a new directory name, this script will convert the force tables by the given floating point conversion factors')
parser.add_argument('input_directory')
parser.add_argument('output_directory')
parser.add_argument('-dist_convert', type=float, default=1.0)
parser.add_argument('-force_convert', type=float, default=1.0)
parser.add_argument('-energy_convert', type=float, default=1.0)

pargs = parser.parse_args()


#check directory
if(not os.path.exists(pargs.input_directory)):
    raise RuntimeError('Could not locate input_directory {}'.format(pargs.input_directory))

#prepare output 
if(not os.path.exists(pargs.output_directory)):
    os.mkdir(pargs.output_directory)

for k,v in pot_types.iteritems():
    input = os.path.join(pargs.input_directory, v)
    output = os.path.join(pargs.output_directory, v)
    with open(output, 'w') as outf:
        with open(input, 'r') as inf:
            for matrix,name, lammps_header in gen_tables(inf, return_extra=True):
                write_table(outf, matrix, name, lammps_header, (pargs.dist_convert, pargs.force_convert, pargs.energy_convert))

