import numpy as np
import matplotlib.pyplot as plt 
import os
import pandas as pd 
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.table import  hstack
from astropy.table import  vstack
from astropy.table import Column
from astropy.table import join
from astropy.table import unique
import modules as mod 
import celerite
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--magCut", type=float, default=24)
parser.add_argument("--Npts", type=int, default=1000) 
parser.add_argument("--fsf", type=float, default=1.0) # SFinf increase factor 
parser.add_argument("--sampleAugmentationFactor", type=str, default='10')

args = parser.parse_args()

print('Using the following settings: ')
print('- input catalog cut at i < %.2f'%args.magCut)
print('- light curves with %d'%args.Npts)
print('- Paper2-derived SFinf increased by a factor of %s'%args.fsf)
print('- the quasar sample increased by a factor of %s, \n'%args.sampleAugmentationFactor)
print('  by simulating that number of DRW light curves for each value of seed r-mag, tau, SFinf')

# read in the catalog 
print('Reading in the catalog ')
qso = Table.read('../catalogs/simQuasars_v1.1.dat', format='ascii',
                names=['ra', 'dec','MB','Mi','MBH','redshift',
                       'DM','alpha','u','g','r','i','z','y', 
                       'tau','SFu','SFg','SFr','SFi','SFz','SFy']
                )
# add an identifier 
qso['dbId'] = np.arange(len(qso)) 


print('Calculate SFinf, tau ')
# calculate SFinf, tau  given paper2 relations 
lambdaOBSr = 6204 # angstroms 
qso['lambdaRFr'] = lambdaOBSr / (1+qso['redshift'])

log10tau = 2.597 + 0.17 * np.log10(qso['lambdaRFr'] / 4000) +\
  0.035 * (qso['Mi']+23) + 0.141*np.log10(qso['MBH'] / 1e9)
qso['tauRF_r'] = np.power(10,log10tau)

qso['tauOBS_r'] = qso['tauRF_r'] * (1+qso['redshift'])

log10sf = -0.476 -0.479 * np.log10(qso['lambdaRFr'] / 4000) + \
0.118 * (qso['Mi']+23) + 0.118*np.log10(qso['MBH'] / 1e9)
qso['sf_r'] = np.power(10,log10sf)

# pre-selection
mag_cut = args.magCut
m = qso['i'] < mag_cut
print('There are %d quasars brighter than %f'%(np.sum(m), mag_cut))
sample = qso[m]


# Use mag-dependent LSST model  for photometric uncertainty ... 
#LSST error model 
def calc_lsst_error(m):
    # Calculate LSST error based on the magnitude of the observed light curve ...
    # we use magnitude-dependent single-visit 
    # formula for r-band from 
    # https://www.lsst.org/sites/default/files/docs/sciencebook/SB_3.pdf
    # https://www.lsst.org/scientists/scibook
    # https://arxiv.org/abs/0912.0201
    # See Chap.3, Sec. 3.5, Eq. 3.1, Table. 3.2
     # mag 
    sigma_sys = 0.005
    gamma = 0.039 
    m5 = 24.7
    x = np.power(10,0.4 * (m-m5))
    sigma_rand = np.sqrt(  (0.04-gamma) * x + gamma * x*x ) 

    # adding in quadrature SDSS error
    sigma_sdss = 0.03 
    sigma = np.sqrt(sigma_sys**2.0 + sigma_rand**2.0 + sigma_sdss**2.0)
    return sigma



# a 20-year light curve, such as 
# between 1998 and 2018, i.e. SDSS to HSC, 
# with Npts epochs 
Npts = args.Npts
t = np.linspace(0, 20*365, Npts) 


fSfinf = str(args.fsf)
faugmentSample = args.sampleAugmentationFactor

name = str(Npts)+'pts_'+fSfinf+'SFr' + faugmentSample+'x-increase_SDSS_err'

outDir = '../data_products/qso_'+name+'/' # : 
if not os.path.exists(outDir):
    os.mkdir(outDir)

col0=[] ; col1=[] ; col2 = [] ; col3 = [] ; col4 = []  ; col5 = [] ; col6 = [] ; col7 = []
col8 = [] ; col9=[] ; col10 = []

print('Simulate light curves... ')

for i in np.arange(len(sample)):
    # sf_r, tauOBS_r  are derived values 
    SF_inf = float(fSfinf)*sample['sf_r'][i]  
    tau_in = sample['tauOBS_r'][i] 
    mean_mag = sample['r'][i]

    dbId = str(sample['dbId'][i])
    if i % 1000 == 0 : 
    	print('%d / %d\n' %(i, len(sample)))
    for j in range(int(faugmentSample)):
        lcname = dbId + '_'+str(j)
        col0.append(sample['ra'][i])
        col1.append(sample['dec'][i])
        col2.append(sample['Mi'][i])
        col3.append(sample['redshift'][i])
        col4.append(sample['r'][i])
        col5.append(tau_in)
        col6.append(lcname)
        col7.append(SF_inf)
        
        #simulate ideal light curve
        y_true = mod.sim_DRW_lightcurve(t, SF_inf, tau_in , mean_mag)

        # add noise corresponding to mag-dependent error model...
        y_err = calc_lsst_error(y_true)

        # simulate the observed light curve 
        y_obs = y_true + y_err * np.random.normal(loc=0,scale=1,size=len(y_true))

        # store the r-band observed light curve ... (time, mag, mag_err )
        lc_obs = Table([t, y_true, y_obs, y_err], 
                      names=['t','magtrue', 'magobs', 'magerr'])
        lc_obs.write(outDir+lcname+'.txt', format='ascii', overwrite=True)

        #store the dmag 
        delta_mag = lc_obs['magobs'][-1] - lc_obs['magobs'][0]
        col8.append(delta_mag)
        #sample['dmag'][i] = delta_mag
        col9.append(lc_obs['magobs'][0]) # SDSS epoch : 1998
        col10.append(lc_obs['magobs'][-1] ) # HSC epoch  : 2018 
        #sample['rmag0yr'][i] = lc_obs['magobs'][0]    # SDSS epoch : 1998
        #sample['rmag20yr'][i] = lc_obs['magobs'][-1]  # HSC epoch  : 2018 
    
# store the catalog ... 
#sample.write('../catalogs/qso_lsst_catalog_i_lt_24_'+name+'.txt', format='ascii')

catalog = Table(data = [col0,col1,col2,col3,col4,col5,
                       col6,col7,col8,col9,col10],
               names = ['ra','dec', 'Mi', 'redshift', 'r','tauOBS_r',
                       'lcname', 'sf_r_input', 'dmag', 'r0yr', 'r20yr'])
fname = 'catalog_'+name+'.txt'
catalog.write(fname, format='ascii', overwrite=True)

print('Saved the qso light curve catalog as %s'%fname)

