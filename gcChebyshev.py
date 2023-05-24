'''
This file contains the graph convolution layer.

The graph convolution is performed in the spectrum of the signal, while 
only considering the low frequencies. The filters are defined by design to 
converge to zero at the end of the low-pass frequency to ensure a smooth 
spectrum in the filtered signal. This in return allows for localized 
filters.

Notes
-----
The mathematical operations to filter a signal :math:`s` with the 
filter :math:`g_{\theta}(\Lambda)` are as follows:
.. math::
    \hat{s} = U g_{\theta}(\Lambda) U^T s.
The Fourier basis :math:`U` of the graph can be obtained by the eigenvalue 
decomposition of symmetric normalized the Laplacian matrix :math:`L`:
.. math::
    L = U \Lambda U^T.

The filter :math:`g_{\theta}(\Lambda)` are defined using Chebyshev
polynomials of the first kind :math:`T_n(\Lambda)`.
'''

from os import path
import tensorflow as tf
import numpy as np
from scipy import sparse

class gcChebyshev(tf.keras.layers.Layer):
    '''
    A layer for graph convolution.
    '''
    
    def __init__(self, num_filters, filter_size, smoothing_cut, smoothing_width, max_frequency=100, A=None, spectrumFilePath='U-{:d}.csv', **kwargs):
        '''
        A layer for graph convolution.

        Parameters
        ----------
        num_filters : int
            The number of filters to use for the convolution.
        filter_size : int
            Defines till which degree the Chebyshev polynomials should be used.
        smoothing_cut : int
            At which index of the frequency the filter should converge to 
            zero. Note, values after this index are guarantied to be zero 
            after half the `smoothing_width` is added.
        smoothing_width : int
            This adjusts the width of the low-pass filter to ensure a 
            converging filter to zero.
        max_frequency : int, default=100
            How many low frequencies should the convolution be applied. The 
            maximum is the whole spectrum, which is equal to the number of 
            vertices of the graph.
        A : scipy.sparse.lil_matrix, optional
            A sparse matrix to represent the symmetric adjacency matrix. This 
            matrix is used to compute the Fourier basis, which is cached in a 
            filed defined by `spectrumFilePath`. Omitting this parameter 
            results in using the stored Fourier basis defined in 
            `spectrumFilePath`.
        spectrumFilePath : str, default='U-{d}.csv'
            The path to the Fourier basis cache. Please note, if `A` is set, 
            this file is getting overwritten.
        '''

        super(gcChebyshev, self).__init__(**kwargs)
        
        self._A = A
        self._num_filters = num_filters
        self._filter_size = filter_size
        self._smoothing_cut = smoothing_cut
        self._smoothing_width = smoothing_width
        self._max_frequency = max_frequency
        self._spectrumFilePath = spectrumFilePath
        
    def get_config(self):
        '''
        This method is used by the tensorflow framework to load this layer.
        '''
        config = super(gcChebyshev, self).get_config()
        config.update({
            'num_filters': self._num_filters,
            'filter_size': self._filter_size,
            'smoothing_cut': self._smoothing_cut,
            'smoothing_width': self._smoothing_width,
            'max_frequency': self._max_frequency
        })
        return config
        
    def build(self, input_shape):
        '''
        Builds the layer based on the input shape.

        The build involves computing or loading the Fourier basis and 
        building up the Chebyshev polynomials.

        Parameters
        ----------
        input_shape : array_like
            The shape of the data set.
        '''
        assert (len(input_shape) == 3), 'expected 3 dimensional input, got ' + str(input_shape)
        
        if self._A != None:
            # compute the symmetric normalized the Laplacian matrix
            d = self._A.sum(axis=0)
            d += np.spacing(np.array(0, self._A.dtype))
            d = 1 / np.sqrt(d)
            D = sparse.diags(d.A.squeeze(), 0)
            I = sparse.identity(d.size, dtype=self._A.dtype)
            L = I - D * self._A * D

            # compute the Fourier basis
            U = self._fourier(L)
            self.U_real = tf.constant(tf.cast(tf.math.real(U), dtype=tf.float32))
        else:
            # load the Fourier basis
            U = self._fourier(None)
            self.U_real = tf.constant(tf.cast(tf.math.real(U), dtype=tf.float32))
        
        self.Filters_real = self.add_weight(
            'filters',
            shape=(self._num_filters, self._filter_size, input_shape[2]),
            initializer=tf.keras.initializers.RandomNormal(),
            trainable=True)
        
        # build up the Chebychev polynomials of the first kind.
        T = np.empty((self._filter_size, self._max_frequency))
        T[0,...] = self._smooth(np.ones((self._max_frequency,)), self._smoothing_cut, self._smoothing_width)
        T[1,...] = self._smooth(np.linspace(-1, 1, self._max_frequency), self._smoothing_cut, self._smoothing_width)
        for k in range(2, self._filter_size):
            T[k,...] = 2 * T[1,...] * T[k-1,...] - T[k-2,...]
            
        self.T = tf.constant(T, dtype=tf.float32)
        
    def _smooth(self, x, cut, width):
        '''
        Applies the low-pass filtering to the Chebyshev polynomials.

        Parameters
        ----------
        x : array_like
            A polynomial to converge to zero at the `cut` frequency.
        cut : int
            The index of the frequency to cut the polynomial at.
        width : int
            The width to allow the cut to converge around the `cut` 
            frequency.

        Returns
        -------
        array_like
            The same as the input `x`, but cut at the `cut` frequency with
            the `width` to converge to zero around the cut-frequency.
        '''
        idx_start = int(cut-width/2)
        idx_end = int(cut+width/2)
        g = 1 / np.sqrt( 1 + (np.arange(width)/width*2*np.pi)**4)
        x[idx_start:idx_end] = x[idx_start:idx_end] * g
        x[idx_end:] = 0
        return x
        
    def _fourier(self, L):
        '''
        Computes the Fourier basis and stores it in an cache file or loads 
        from this file, without computation.

        Parameters
        ----------
        L : scipy.sparse.lil_matrix
            The symmetric normalized Laplacian matrix.

        Returns
        -------
        array_like
            The Fourier basis as a matrix.
        '''
        filePath = self._spectrumFilePath.format(self._max_frequency)
        if path.exists(filePath):
            U = np.loadtxt(filePath, dtype=np.complexfloating)
        else:
            lamb, U = sparse.linalg.eigsh(
                L, 
                k=self._max_frequency, 
                sigma=0,
                v0=np.ones((L.shape[0],)) * 0.00230709,
                which='LM',
                ncv=300,
            )
            np.savetxt(filePath, U)
        return U
            
    def call(self, inputs):
        '''
        Computes the actual convolution on the input without the inverse 
        Fourier transformation.

        Parameters
        ----------
        inputs : array_like
            The batch input to convolve with the filters.

        Returns
        -------
        array_like
            The convolved signals of this batch.
        '''
        with tf.name_scope('gcnn'):
            s_out = []
            for ch in range(self.Filters_real.shape[2]):
                s = tf.transpose(inputs[:,:,ch])
                S_real = tf.matmul(tf.transpose(self.U_real), s)
                s_ch_out = []
                for i in range(self._num_filters):
                    F_real = tf.matmul(tf.transpose(self.T), self.Filters_real[i,:,ch:ch+1])
                    S_ch_tap_real = tf.math.multiply(F_real, S_real)
                    
                    s_ch_out.append(S_ch_tap_real)
                
                s_out.append(s_ch_out)
            s_out = tf.reduce_sum(s_out, axis=0)
            
            s_out = tf.transpose(s_out)
            
        return s_out