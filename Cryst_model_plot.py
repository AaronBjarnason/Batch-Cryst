# Function to plot experimental results for batch_crystallizer_VSC_01.ipynb

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Results:
    ''' This object stores and plots results.
    '''

    def __init__(self,m):
        self.time= None
        self.mu0 = None
        self.mu1 = None
        self.mu2 = None
        self.mu3 = None
        self.mu4 = None
        # self.B = None
        self.Bp = None
        self.Bs = None
        self.G = None
        self.C = None
        self.MT = None

        self._load_scaling(m)

    def _load_scaling(self,m):

        self.mu0_scale = m.mu0_scale
        self.mu1_scale = m.mu1_scale
        self.mu2_scale = m.mu2_scale
        self.mu3_scale = m.mu3_scale
        self.mu4_scale = m.mu4_scale
        # self.B_scale = m.B_scale
        self.Bp_scale = m.Bp_scale
        self.Bs_scale = m.Bs_scale
        self.G_scale = m.G_scale


        # Not implemented (yet?)
        # self.C_scale = self.C_scale
        # self.MT_scale = self.MT_scale

    def to_pandas(self):
      self.undo_scaling=True
      # pandas data frame
      df = pd.DataFrame(columns=['time', 'mu0', 'mu1','mu2','mu3','mu4','Bp','Bs','G','C','MT','T'])
      df.time=self.time
      df.mu0=self.mu0*self._get_scaling(self.mu0_scale)
      df.mu1=self.mu1*self._get_scaling(self.mu1_scale)
      df.mu2=self.mu2*self._get_scaling(self.mu2_scale)
      df.mu3=self.mu3*self._get_scaling(self.mu3_scale)
      df.mu4=self.mu4*self._get_scaling(self.mu4_scale)
      df.Bp=self.Bp*self._get_scaling(self.Bp_scale)
      df.Bs=self.Bs*self._get_scaling(self.Bs_scale)
      df.G=self.G*self._get_scaling(self.G_scale)
      df.C=self.C
      df.MT=self.MT
      df.T=self.T
      return df

    def load_from_simulator(self,tsim,profiles):
        self.time = tsim
        print(tsim)
        self.mu0 = profiles[:,0]
        self.mu1 = profiles[:,1]
        self.mu2 = profiles[:,2]
        self.mu3 = profiles[:,3]
        self.mu4 = profiles[:,4]
        # self.B = profiles[:,8]
        # self.G = profiles[:,9]
        self.Bp = profiles[:,8]
        self.Bs = profiles[:,9]
        self.G = profiles[:,10]
        self.C = profiles[:,5]
        self.MT = profiles[:,6]
        self.T = profiles[:,7]

    def load_from_pyomo_model(self,m):
        self.time = np.array([t for t in m.t])
        self.mu0 = np.array([m.mu0[t]() for t in m.t])
        self.mu1 = np.array([m.mu1[t]() for t in m.t])
        self.mu2 = np.array([m.mu2[t]() for t in m.t])
        self.mu3 = np.array([m.mu3[t]() for t in m.t])
        self.mu4 = np.array([m.mu4[t]() for t in m.t])
        # self.B = np.array([m.B[t]() for t in m.t])
        self.Bp = np.array([m.Bp[t]() for t in m.t])
        self.Bs = np.array([m.Bs[t]() for t in m.t])
        self.G = np.array([m.G[t]() for t in m.t])
        self.C = np.array([m.C[t]() for t in m.t])
        self.MT = np.array([m.MT[t]() for t in m.t])
        self.T = np.array([m.T[t]() for t in m.t])

    def _get_scaling(self,scaling_factor):
        if self.undo_scaling:
            return scaling_factor
        else:
            return 1

    def _get_label_string(self,name):

        units = {'\mu_0':'[mL^{-1}]',
                 '\mu_1':'[mm.mL^{-1}]',
                 '\mu_2':'[mm^{2}.mL^{-1}]',
                 '\mu_3':'[mm^{3}.mL^{-1}]',
                 '\mu_4':'[mm^{4}.mL^{-1}]',
                 'B':'[mL^{-1}.s^{-1}]',
                 'G': '[mm.s^{-1}]'}

        units = {'\mu_0':'[mL^{-1}]',
                 '\mu_1':'[mm.mL^{-1}]',
                 '\mu_2':'[mm^{2}.mL^{-1}]',
                 '\mu_3':'[mm^{3}.mL^{-1}]',
                 'Bp':'[\#.mL^{-1}.s^{-1}]',
                 'Bs': '[\#.mL^{-1}.s^{-1}]',
                 'G': '[mm.s^{-1}]'}


        if not self.undo_scaling:
            return '$'+name+'~[scaled]$'
        else:
            return '$'+name+' '+units[name]+'$'


    def plot(self,undo_scaling=False):
        self.undo_scaling = undo_scaling

        fig, axes = plt.subplots(2, 4, figsize=(12, 8))

        axes[0,0].plot(self.time, self.mu0*self._get_scaling(self.mu0_scale), label = 'Zero moment')
        #axes[0,0].set_ylabel('$\mu_0 [mL^{-1}]$')
        axes[0,0].set_ylabel(self._get_label_string('\mu_0'))

        axes[0,1].plot(self.time, self.mu1*self._get_scaling(self.mu1_scale), label = 'First moment')
        #axes[0,1].set_ylabel('$\mu_1 [mm.mL^{-1}]$')
        axes[0,1].set_ylabel(self._get_label_string('\mu_1'))

        axes[0,2].plot(self.time, self.mu2*self._get_scaling(self.mu2_scale), label = 'Second moment')
        #axes[0,2].set_ylabel('$\mu_2 [mm^{2}.mL^{-1}]$')
        axes[0,2].set_ylabel(self._get_label_string('\mu_2'))

        axes[0,3].plot(self.time, self.mu3*self._get_scaling(self.mu3_scale), label = 'Third moment')
        #axes[0,3].set_ylabel('$\mu_3 [mm^{3}.mL^{-1}]$')
        axes[0,3].set_ylabel(self._get_label_string('\mu_3'))

        # axes[1,0].plot(self.time, self.mu4*self._get_scaling(self.mu4_scale), label = 'Fourth moment')
        # #axes[1,0].set_ylabel('$\mu_4 [mm^{4}.mL^{-1}]$')
        # axes[1,0].set_ylabel(self._get_label_string('\mu_4'))

        # axes[1,1].plot(self.time, self.B*self._get_scaling(self.B_scale), label = 'Nucleation rate')
        # #axes[1,1].set_ylabel('$B [mL^{-1}.s^{-1}]$')
        # axes[1,1].set_ylabel(self._get_label_string('B'))

        axes[1,0].plot(self.time, self.Bp*self._get_scaling(self.Bp_scale), label = 'Primary Nucleation rate')
        #axes[1,1].set_ylabel('$B [mL^{-1}.s^{-1}]$')
        axes[1,0].set_ylabel(self._get_label_string('Bp'))

        axes[1,1].plot(self.time, self.Bs*self._get_scaling(self.Bs_scale), label = 'Secondary Nucleation rate')
        #axes[1,1].set_ylabel('$B [mL^{-1}.s^{-1}]$')
        axes[1,1].set_ylabel(self._get_label_string('Bs'))

        axes[1,2].plot(self.time, self.G*self._get_scaling(self.G_scale), label = 'Growth rate')
        #axes[1,2].set_ylabel('$G [mm.s^{-1}]$')
        axes[1,2].set_ylabel(self._get_label_string('G'))

        axes[1,3].plot(self.time, self.C, label = 'C')
        # axes[1,3].plot(self.time, self.MT, label = 'M_T')
        axes[1,3].set_ylabel('$[g.cm^{-3}]$')
        #axes[1,3].legend()

        # axes[1].plot(tsim, (profiles[:,4]/profiles[:,3])*1e6, label = 'Mean particle size', color = 'green')
        # axes[1].set_xlabel('Time (s)')
        # axes[1].set_ylabel('Particle size $(\mu m)$')
        # axes[1].legend()

        fig.tight_layout()
        plt.show()