#! /usr/bin/env python3

import numpy as np
import h5py
import note_decompose as nde
import scipy.signal as signal
import matplotlib.pyplot as plt
import tftb.processing as tftb
sudo pip install adaptfilt
import adaptfilt as adpt


def wave2vol(wave, spread=10, detect_type='peak'):
	
	oldlen = len(wave)
	newlen = int(np.ceil(len(wave) / spread)) * spread
	
	wave = np.append(wave, np.zeros(newlen - oldlen))
	
	wave = wave.reshape(int(newlen / spread), spread)
	
	if detect_type == 'peak':
		volume = np.max(np.abs(wave), axis=1)
	elif detect_type == 'rms':
		volume = np.sqrt(np.sum(wave**2, axis=1) / wave.shape[1])
	else:
		raise Exception("only peak and rms detection types are defined so far")
	
	volume = np.repeat(volume, spread)
	
	volume = volume[0:oldlen]
	
	return volume


# -------------------------------------------------------------------------------------------- #


class noise_gate_sigma:
	def __init__(self, bg_file, octave):
		nde_class = nde.decompose(bg_file)
		nde_class.octaves = octave
		nde_class.decompose('ns~background')
		
		fp = h5py.File('ns~background.hdf5', 'r', libver='latest')
		
		self.bg_sigma = {}
		self.bg_mean = {}
		
		for key in list(fp.keys()):
			
			if key == 'meta':
				continue
			
			self.bg_sigma[key] = np.std(wave2vol(fp[key], spread=1000))
			self.bg_mean[key] = np.mean(wave2vol(fp[key], spread=1000))
		
		fp.close()
		return
	
	def noise_gate_sigma(self, data, key, sigma, spread=1000):
		""" noise gate: let noise through once it reaches a certain level
		level => abs level at witch the noise gate activates
		data => the data we want to operate on
		mult => scalar multiplier for the data
		spread => how long after the last activation do we shut off
		
		return: gated data
		side effects: None
		"""
		
		level = self.bg_mean[key] + (self.bg_sigma[key] * sigma)
		
		volume = wave2vol(data, spread=spread)
		
		vol_mask = np.logical_not(np.greater(volume, level))
		
		new_data = np.copy(data)
		new_data[vol_mask] = 0
		
		return new_data


# -------------------------------------------------------------------------------------------- #


def noise_gate_PWVD(data, spread=1000):
	
	# TODO something smoothed_pseudo_wigner_ville
	
	data_volume = wave2vol(data, spread=spread)
	
	# ------- PWVD ---------------------------- #
	fwindow = signal.hamming(1)
	twindow = signal.hamming(1)
	
	de_dup_vol = data_volume[0::spread]
	
	spec = tftb.smoothed_pseudo_wigner_ville(de_dup_vol, fwindow=fwindow, twindow=twindow)
	m = np.max(spec, axis=0)
	
	ada_volume = m * np.max(de_dup_vol)
	
	ada_volume = np.repeat(ada_volume, spread)
	
	ada_volume = ada_volume[0:len(volume)]
	# -----------------------------------------#
	
	volume_scale = ada_volume / data_volume
	
	new_data = np.copy(data)
	new_data = new_data * volume_scale
	
	return new_data
# -------------------------------------------------------------------------------------------#
#y, e, w = nlms(u, d, M, step, eps=0.001, leak=0, initCoeffs=None, N=None, returnCoeffs=False)**
#this function used the normalsied least mean square adaptive filtering methodon u minimising error on e=d-y where d is the desired signal
# some of this code i undertand and other parts i dont but if we get this working/ adapted to our code i think it would then track and remove the unwanted signal, ive tried to adapt the start of it but have alot of question i need to ask about code first to do this :) 



    u = oldlen  
    d = newlen
    M = 20  # No. of taps
    step = 1  # Step size
    y, e, w = nlms(u, d, M, step, N=N, returnCoeffs=True)
    >>> y.shape == (N,)
    True
    >>> e.shape == (N,)
    True
    >>> w.shape == (N, M)
    True
    >>> # Calculate mean square weight error
    >>> mswe = np.mean((w - coeffs)**2, axis=1)
    >>> # Should never increase so diff should above be > 0
    >>> diff = np.diff(mswe)
    >>> (diff <= 1e-10).all()
    True
    
   
    # Max iteration check
    if N is None:
        N = len(u)-M+1
    _pchk.checkIter(N, len(u)-M+1)

    # Check len(d)
    _pchk.checkDesiredSignal(d, N, M)

    # Step check
    _pchk.checkStep(step)

    # Leakage check
    _pchk.checkLeakage(leak)

    # Init. coeffs check
    if initCoeffs is None:
        initCoeffs = np.zeros(M)
    else:
        _pchk.checkInitCoeffs(initCoeffs, M)

    # Initialization
    y = np.zeros(N)  # Filter output
    e = np.zeros(N)  # Error signal
    w = initCoeffs  # Initial filter coeffs
    leakstep = (1 - step*leak)
    if returnCoeffs:
        W = np.zeros((N, M))  # Matrix to hold coeffs for each iteration

    # Perform filtering
    for n in xrange(N):
        x = np.flipud(u[n:n+M])  # Slice to get view of M latest datapoints
        y[n] = np.dot(x, w)
        e[n] = d[n+M-1] - y[n]

        normFactor = 1./(np.dot(x, x) + eps)
        w = leakstep * w + step * normFactor * x * e[n]
        y[n] = np.dot(x, w)
        if returnCoeffs:
            W[n] = w

    if returnCoeffs:
        w = W

    return y, e, w




