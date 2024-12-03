import os
import numpy as np
import warnings
import montepython.io_mp as io_mp
from montepython.likelihood_class import Likelihood
import scipy.constants as conts

#  adapted from bao_boss_dr12 likelihood
class bao_desi_all(Likelihood):

    # initialization routine
    def __init__(self, path, data, command_line):

        Likelihood.__init__(self, path, data, command_line)
        # Note: need to check for conflicting experiments manually

        # define arrays for values of z and data points
        self.z = np.array([], 'float64')
        self.data_array = np.array([], 'float64')
        self.quantity = []

        # read redshifts and data points
        with open(os.path.join(self.data_directory, self.data_file), 'r') as filein:
            for i, line in enumerate(filein):
                if line.strip() and line.find('#') == -1:
                    this_line = line.split()
                    self.z = np.append(self.z, float(this_line[0]))
                    self.data_array = np.append(self.data_array, float(this_line[1]))
                    self.quantity.append(str(this_line[2]))

        # read covariance matrix
        self.cov_data = np.loadtxt(os.path.join(self.data_directory, self.cov_file))

        # number of bins
        self.num_bins = np.shape(self.z)[0]

        # number of data points
        self.num_points = np.shape(self.cov_data)[0]

        # end of initialization

    # compute likelihood

    def loglkl(self, cosmo, data):

        # for each point, compute comoving angular diameter distance D_M = (1 + z) * D_A,
        # Hubble distance D_H = 1 / H(z),
        # sound horizon at baryon drag rs and
        # angle-averaged distance D_V = (z * D_M^2 * D_H)^(1/3)
        
        diff = np.zeros(self.num_bins)
        for i in range(self.num_bins):

            DM_at_z = cosmo.angular_distance(self.z[i]) * (1. + self.z[i])
            H_at_z = cosmo.Hubble(self.z[i])
            rs = cosmo.rs_drag()

            theo_DM_over_rs = DM_at_z / rs
            theo_DH_over_rs = 1. / H_at_z / rs
            theo_DV_over_rs = (self.z[i] * DM_at_z**2 / H_at_z)**(1./3.) / rs

            # calculate difference between the sampled point and observations
            if self.quantity[i] == 'DV_over_rs':
                diff[i] = theo_DV_over_rs - self.data_array[i]
            elif self.quantity[i] == 'DM_over_rs':
                diff[i] = theo_DM_over_rs - self.data_array[i]
            elif self.quantity[i] == 'DH_over_rs':
                diff[i] = theo_DH_over_rs - self.data_array[i]
        
        # compute chi squared
        inv_cov_data = np.linalg.inv(self.cov_data)
        chi2 = np.dot(np.dot(diff,inv_cov_data),diff)

        # return ln(L)
        loglkl = - 0.5 * chi2

        return loglkl
