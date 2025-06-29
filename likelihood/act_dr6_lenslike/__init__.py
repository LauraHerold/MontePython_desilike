# Adpated from https://github.com/ACTCollaboration/act_dr6_lenslike for MontePython
# Requires to install act_dr6_lenslike first
import act_dr6_lenslike as alike
import montepython.io_mp as io_mp
from montepython.likelihood_class import Likelihood

class act_dr6_lenslike(Likelihood):

    def __init__(self, path, data, command_line):
        
        Likelihood.__init__(self, path, data, command_line)

        self.data_dict = alike.load_data(self.variant,lens_only=self.lens_only,like_corrections=self.like_corrections)

        # This dict will now have entries like `data_binned_clkk` (binned data vector), `cov`
        #(covariance matrix) and `binmat_act` (binning matrix to be applied to a theory
        # curve starting at ell=0).

    def loglkl(self, cosmo, data):
        # Get cl_kk, cl_tt, cl_ee, cl_te, cl_bb predictions from your Boltzmann code.
        # These are the CMB lensing convergence spectra (not potential or deflection)
        # as well as the TT, EE, TE, BB CMB spectra (needed for likelihood corrections)
        # in uK^2 units. All of these are C_ell (not D_ell), no ell or 2pi factors.

        T_CMB = cosmo.T_cmb() * 1e6 # in uK
        cls = cosmo.lensed_cl(4000) # [dimensionless]
        ell = cls['ell']
        
        # Convert C_l^\phi\phi to C_l^\kappa\kappa, Eq. A11 in https://journals.aps.org/prd/pdf/10.1103/PhysRevD.62.043007 or Eq. 1 in https://arxiv.org/pdf/2103.05582
        kk_pref = ell**2 * (ell+1)**2 / 4.
        cl_kk = kk_pref * cls['pp']
        cl_tt = cls['tt'] * T_CMB**2
        cl_ee = cls['ee'] * T_CMB**2
        cl_te = cls['te'] * T_CMB**2
        cl_bb = cls['bb'] * T_CMB**2

        lnlike = alike.generic_lnlike(self.data_dict, ell, cl_kk, ell, cl_tt, cl_ee, cl_te, cl_bb)
        return lnlike
