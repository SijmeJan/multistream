#!/usr/bin/python

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

class MSI():
    """Multi-species streaming instability class

    Class for calculating growth rates of the multi-species streaming instability. Follows notation of Benitez-Llambay et al. (2019, BLKP19).

    Args:
        eps: array of dust-to-gas ratios, one element for each species
        Ts: array of stopping times, one element for each species
        h0 (optional): aspect ratio of the disc. Defaults to 0.05.
        cs_over_eta (optional): sound speed in dimensionless units is h0/cs_over_eta. Defaults to unity.
    """
    def __init__(self, eps, Ts, h0=0.05, cs_over_eta=1.0):
        # Convert to arrays
        self.Ts = np.atleast_1d(Ts)
        self.eps = np.atleast_1d(eps)
        self.h0 = h0
        self.soundspeed = h0/cs_over_eta

        # Calculate equilibrium velocities using eqs (79)-(84) of BLKP19
        An = np.sum(self.eps*self.Ts/(1 + self.Ts*self.Ts))
        Bn = np.sum(self.eps/(1 + self.Ts*self.Ts)) + 1.0

        chi0 = 2*self.h0*self.h0
        psi = 1/(An*An + Bn*Bn)

        # Equilibrium gas velocity (units: Keplerian velocity box!)
        self.vgx = An*chi0*psi
        self.vgy = -0.5*Bn*chi0*psi

        # Equilibrium dust velocity (note: vectors!)
        self.vdx = (self.vgx + 2*self.Ts*self.vgy)/(1 + self.Ts*self.Ts)
        self.vdy = (self.vgy - 0.5*self.Ts*self.vgx)/(1 + self.Ts*self.Ts)

    def matrix(self, Kx, Kz):
        """Construct MSI matrix

        Args:
            Kx: non-dimensional wave number x
            Kz: non-dimensional wave number z
        """

        N = len(self.Ts)    # Number of dust species

        # Matrix: 4 equations per species (and gas!)
        M = np.zeros((4*N + 4, 4*N + 4), dtype=complex)

        # Scale velocities (see appendix E of BLKP19)
        vgx = self.vgx/self.h0**2
        vgy = self.vgy/self.h0**2
        vdx = self.vdx/self.h0**2
        vdy = self.vdy/self.h0**2

        # Relative velocities
        dvx = vgx - vdx
        dvy = vgy - vdy

        # Gas continuity equation
        M[0, 0] = 1j*Kx*vgx
        M[0, 1] = 1j*Kx
        M[0, 3] = 1j*Kz

        # Gas momentum x
        M[1, 0] = 1j*Kx/self.soundspeed**2 - np.sum(self.eps*vdx/self.Ts)
        M[1, 1] = 1j*Kx*vgx + np.sum(self.eps/self.Ts)
        M[1, 2] = -2
        M[1, 4:4*N+4:4] = dvx/self.Ts
        M[1, 5:4*N+4:4] = -self.eps/self.Ts

        # Gas momentum y
        M[2, 0] = -np.sum(self.eps*vdy/self.Ts)
        M[2, 1] = 0.5
        M[2, 2] = 1j*Kx*vgx + np.sum(self.eps/self.Ts)
        M[2, 4:4*N+4:4] = dvy/self.Ts
        M[2, 6:4*N+4:4] = -self.eps/self.Ts

        # Gas momentum z
        M[3, 0] = 1j*Kz/self.soundspeed**2
        M[3, 3] = 1j*Kx*vgx + np.sum(self.eps/self.Ts)
        M[3, 7:4*N+4:4] = -self.eps/self.Ts

        # Deal with all dust species
        for j in range(0, N):
            i = 4*j + 4

            # Dust continuity
            M[i, i] = 1j*Kx*vdx[j]
            M[i, i+1] = 1j*Kx*self.eps[j]
            M[i, i+3] = 1j*Kz*self.eps[j]

            # Dust momentum x
            M[i+1, 1] = -1/self.Ts[j]
            M[i+1, i+1] = 1j*Kx*vdx[j] + 1/self.Ts[j]
            M[i+1, i+2] = -2

            # Dust momentum y
            M[i+2, 2] = -1/self.Ts[j]
            M[i+2, i+1] = 0.5
            M[i+2, i+2] = 1j*Kx*vdx[j] + 1/self.Ts[j]

            # Dust momentum z
            M[i+3, 3] = -1/self.Ts[j]
            M[i+3, i+3] = 1j*Kx*vdx[j] + 1/self.Ts[j]

        # Return matrix
        return M

    def eigvals(self, Kx, Kz):
        """Calculate eigenvalues of MSI matrix

        Args:
            Kx: non-dimensional wave number x
            Kz: non-dimensional wave number z
        """
        return linalg.eigvals(self.matrix(Kx, Kz))

    def eig(self, Kx, Kz):
        """Calculate eigenvalues and eigenvectors of MSI matrix

        Args:
            Kx: non-dimensional wave number x
            Kz: non-dimensional wave number z
        """

        return linalg.eig(self.matrix(Kx, Kz), left=False, right=True)

    def max_growth(self, Kx, Kz, eigenvector=False):
        """Calculate maximum growth rate

        Args:
            Kx: non-dimensional wave number x (may be an array)
            Kz: non-dimensional wave number z (may be an array)
        """
        Kx = np.asarray(Kx)
        Kz = np.asarray(Kz)

        # Make sure we can handle both vector and scalar K's
        scalar_input = False
        if Kx.ndim == 0:
            Kx = Kx[None]  # Makes 1D
            Kz = Kz[None]  # Makes 1D
            scalar_input = True
        else:
            original_shape = np.shape(Kx)
            Kx = np.ravel(Kx)
            Kz = np.ravel(Kz)

        ret = np.zeros(len(Kx), dtype=complex)
        evector = []
        # Calculate maximum growth rate for each K
        for i in range(0, len(Kx)):
            ev = self.eigvals(Kx[i], Kz[i])
            ret[i] = ev[ev.real.argmin()]

            if eigenvector is True:
                evalue, evector = self.eig(Kx[i], Kz[i])
                evector = evector[:,ev.real.argmin()]/evector[4,ev.real.argmin()]

        # Return value of original shape
        if eigenvector is False:
            if scalar_input:
                return np.squeeze(ret)

            return np.reshape(ret, original_shape)

        if scalar_input:
            return np.squeeze(ret), evector

        return np.reshape(ret, original_shape), evector



# Test Case 1: single dust fluid, LinA from Table 4 in BLKP19
msi = MSI(eps=[3.0], Ts=[0.1])
print(msi.max_growth(30,30))
