import numpy as np 
import matplotlib.pyplot as plt 
from sys import exit
import camb
import Params_CosmoBasic as params
from scipy import integrate
from scipy.interpolate import interp1d, interp2d
from subprocess import call

#==============================================================================

class CosmoPowerLib(object):
	"""
	InputParams: 
	Returns:
	"""
	
#------------------------------------------------------------------------------

	def __init__(self, CosmoParams=[0.3,0.8,0.7,0.96,0.046,-1.0,0.0,0.0,0.0], \
					computeGF=True):

		# Initializing parameters from the parameter file: Params_CosmoBasic.py
		self.CosmoParams = CosmoParams
		[self.Omega_m, self.Sigma_8, self.h, self.n_s, self.Omega_b, \
						self.w0, self.wa, self.Omega_r, self.Omega_k] = self.CosmoParams
		self.Omega_l = 1.0 - self.Omega_m - self.Omega_r - self.Omega_k

		# Computing GrowthFactor
		if computeGF:
			[self.z_gf, self.gff] = self.growthfactor()

#------------------------------------------------------------------------------

	def PKL_Camb_SingleRedshift(self, kk, zz=0.0):
		pars = camb.CAMBparams()
		pars.set_cosmology(H0=self.h*100, ombh2=self.Omega_b*self.h**2, \
							omch2=(self.Omega_m-self.Omega_b)*self.h**2)
		pars.set_dark_energy(w=self.w0) #re-set defaults
		pars.InitPower.set_params(ns=self.n_s)
		pars.set_matter_power(redshifts=[zz], kmax=10.0)
		results = camb.get_results(pars)
		kh, z, pk = \
			results.get_matter_power_spectrum(minkh=1e-4,maxkh=10.0,npoints=500)
		s8 = np.array(results.get_sigma8())
		pklin = pk[0]*(self.Sigma_8/s8)**2	
		return np.interp(kk, kh, pklin)

#------------------------------------------------------------------------------

	def PKL_Camb_MultipleRedshift(self, kk, zz):
		if len(zz)<2:
			print "zz must be a list, if you need to compute \
			only for single redshift, use PowerSpectrumLinear_SingleRedshift"
			exit()
		pklin = self.PKL_Camb_SingleRedshift(kk, zz=0.0)
		return np.outer(pklin, np.interp(zz, self.z_gf, self.gff)**2)

#------------------------------------------------------------------------------

	def PKNL_CAMB_SingleRedshift(self, kk, zz=0.0):
		pars = camb.CAMBparams()
		pars.set_cosmology(H0=self.h*100, ombh2=self.Omega_b*self.h**2, \
								omch2=(self.Omega_m-self.Omega_b)*self.h**2)
		pars.set_dark_energy(w=self.w0) #re-set defaults
		pars.InitPower.set_params(ns=self.n_s)
		pars.set_matter_power(redshifts=[zz], kmax=10.0)
		results = camb.get_results(pars)
		pars.NonLinear = camb.model.NonLinear_both
		results.calc_power_spectra(pars)
		kh_nonlin, z_nonlin, pk_nonlin = \
		results.get_matter_power_spectrum(minkh=1e-4,maxkh=10.0,npoints=500)
		s8 = np.array(results.get_sigma8())
		pknonlin = pk_nonlin[0]*(self.Sigma_8/s8)**2
		return np.interp(kk, kh_nonlin, pknonlin)

#------------------------------------------------------------------------------

	def PKNL_CAMB_MultipleRedshift(self, kk, zz):
		if len(zz)<2:
			print "zz must be a list, if you need to compute \
			only for single redshift, use PowerSpectrumNonLinear_SingleRedshift"
			exit()
		pars = camb.CAMBparams()
		pars.set_cosmology(H0=self.h*100, ombh2=self.Omega_b*self.h**2, \
								omch2=(self.Omega_m-self.Omega_b)*self.h**2)
		pars.set_dark_energy(w=self.w0) #re-set defaults
		pars.InitPower.set_params(ns=self.n_s)
		pars.set_matter_power(redshifts=zz, kmax=10.0)
		results = camb.get_results(pars)
		pars.NonLinear = camb.model.NonLinear_both
		results.calc_power_spectra(pars)
		kh_nonlin, z_nonlin, pk_nonlin = \
		results.get_matter_power_spectrum(minkh=1e-4,maxkh=10.0,npoints=500)
		s8 = np.array(results.get_sigma8())
		for i in range(len(s8)):
			pk_nonlin[:,i] = pk_nonlin[:,i]*(self.Sigma_8*np.interp(zz[i], self.z_gf, self.gff)/s8[i])**2
		function = interp2d(kh_nonlin, z_nonlin, pk_nonlin)
		return np.transpose(function(kk, zz))

#------------------------------------------------------------------------------

	def GetInputFile4FrankenEMU(self,redshift=0.0,\
		pkfilename='pk.pk',inputfilename='inputfile4emu.ini'):
		cosmology = self.CosmoParams
		f = open(inputfilename,'w')
		for i in range(10):
			if i==0:
				f.write('%s \n'%pkfilename)
			if i==1:
				f.write('0 \n')
			if i==2:
				f.write('%1.5f \n'%(cosmology[4]*cosmology[2]**2))
			if i==3:
				f.write('%1.5f \n'%(cosmology[0]*cosmology[2]**2))
			if i==4:
				f.write('%1.5f \n'%cosmology[3])
			if i==5:
				f.write('%1.5f \n'%(cosmology[2]*100.0))
			if i==6:
				f.write('%1.5f \n'%cosmology[5])
			if i==7:
				f.write('%1.5f \n'%cosmology[1])
			if i==8:
				f.write('%1.5f \n'%redshift)
			if i==9:
				f.write('2')
		f.close()
		return inputfilename

	def PKNL_FrankenEmu_SingleRedshift(self, k, redshift=0.0, pkfile='pk.pk'):
		from subprocess import call
		inputfilename = self.GetInputFile4FrankenEMU(redshift,pkfile)
		call('/Users/mohammed/Dropbox/work/berkeley/'+\
			'visit2_March2015/work/FrankenEmu/emu.exe < %s'\
			%inputfilename,shell=True)
		kpk = np.genfromtxt(pkfile,skip_header=5)
		call('rm %s %s'%(inputfilename, pkfile),shell=True)
		kpk[:,0] = kpk[:,0] / self.CosmoParams[2]
		kpk[:,1] = kpk[:,1] * self.CosmoParams[2]**3
		return np.interp(k,kpk[:,0],kpk[:,1])

#------------------------------------------------------------------------------

	def GetInputFile4EMU(self,redshift=0.0,\
		pkfilename='pk.pk',inputfilename='inputfile4emu.ini'):
		cosmology = self.CosmoParams
		f = open(inputfilename,'w')
		for i in range(10):
			if i==0:
				f.write('%s \n'%pkfilename)
			if i==1:
				f.write('%1.5f \n'%(cosmology[0]*cosmology[2]**2))
			if i==2:
				f.write('%1.5f \n'%(cosmology[4]*cosmology[2]**2))
			if i==3:
				f.write('%1.5f \n'%cosmology[3])
			if i==4:
				f.write('%1.5f \n'%cosmology[1])
			if i==5:
				f.write('%1.5f \n'%cosmology[5])
			if i==6:
				f.write('%1.5f \n'%redshift)
			if i==7:
				f.write('2')
		f.close()
		return inputfilename

	def PKNL_Emu_SingleRedshift(self, k, redshift=0.0, pkfile='pk.pk'):
		from subprocess import call
		inputfilename = self.GetInputFile4EMU(redshift,pkfile)
		call('/Users/mohammed/Dropbox/work/berkeley/'+\
			'visit2_March2015/work/emulator/emu.exe < %s'\
			%inputfilename,shell=True)
		kpk = np.genfromtxt(pkfile,skip_header=5)
		call('rm %s %s'%(inputfilename, pkfile),shell=True)
		kpk[:,0] = kpk[:,0] / self.CosmoParams[2]
		kpk[:,1] = kpk[:,1] * self.CosmoParams[2]**3
		return np.interp(k,kpk[:,0],kpk[:,1])

#------------------------------------------------------------------------------

	def growthfactor(self): #Uses numpy
	    from numpy import exp
	#    Cosmology
	    Om = self.Omega_m
	    Ol = self.Omega_l
	    w0 = self.w0
	    wa = self.wa
	    Ok = self.Omega_k
	    Or = self.Omega_r
	    
	    #initial conditions
	    a0 = 1.0e-4     #initial scale factor, can be a small number
	    g0 =  a0        #growthfactor(a0) = a0
	    zz0 = 1.0       #first derivative of growthfactor: 
	    				#d(growthfactor)/da (@ a0) = 1.0
	    
	    a = []
	    z = []
	    Ez = []
	    dEdz = []
	    zdim = 1000
	    for i in range(zdim):
	        a.append(a0 + float(i)/(zdim-1)*(1.0-a0))
	        z.append((1.0-a[i])/(a[i]))

	        Ez.append(self.Ez(z[i]))

	        dEdz.append((1.0/2.0/Ez[i]) * (-3.0*Om/a[i]**4 \
	                -2.0*Ok/a[i]**3 - 4.0*Or/a[i]**5 + \
	                Ol*(a[i]**(-3.0*(1.+w0+wa))) * exp(-3.*wa*(1.-a[i])) * \
	                (3.0*wa-3.0*(1.0+w0+wa)/a[i])))
	    zz = []
	    g = []
	    zz.append(zz0)
	    g.append(g0)
	    for i in range(1,zdim): #incrementing wih Euler method.
	        h = a[i] - a[i-1]
	        k1y = zz[i-1]*h
	        k1z = (-(3.0/a[i-1] + dEdz[i-1]/Ez[i-1])*zz[i-1] + \
	            3.0*Om*g[i-1]/2.0/(a[i-1]**5)/Ez[i-1]**2)*h
	        g.append(g[i-1] + k1y)
	        zz.append(zz[i-1] + k1z)
	        
	    for i in range(zdim):   #reversing the arrays
	        g[i] = g[i]/g[zdim-1]
	    g.reverse()
	    z.reverse()
	    return [z,g]

#------------------------------------------------------------------------------

	def get_gf(self, z=0.0):
		if z==0.0:
			return 1.0
		else:
			return np.interp(z, self.z_gf, self.gff)

#------------------------------------------------------------------------------

	def Ez(self, z=0.0):
		if z==0.0:
			return 1.0
		else:
			a = 1.0 / (1.0 + z)
			return (self.Omega_m/a**3 + \
						self.Omega_k/a**2 + \
						self.Omega_r/a**4 +\
						self.Omega_l/a**(3.0*(1.0+self.w0+self.wa))/\
						np.exp(3.0*self.wa*(1.0-a)))**0.5

#==============================================================================

if __name__=="__main__":

	co = CosmoPowerLib()
	kk = 10**np.linspace(-3.0, 1.0, 100)
	zz = [0.0, 0.5, 1.0, 2.0]

	pk = co.PKNL_FrankenEmu_SingleRedshift(kk, 0.0)
	pk2 = co.PKNL_Emu_SingleRedshift(kk, 0.0)
	pkk = co.PKNL_CAMB_SingleRedshift(kk)

	plt.loglog(kk, pk, 'b', lw=2)
	plt.loglog(kk, pkk, 'r', lw=2)
	plt.loglog(kk, pk2, 'g', lw=2)
	plt.show()




