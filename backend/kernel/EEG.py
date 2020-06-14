import numpy as np
import entropy
import kernel


# Delta 0 - 4
# Theta 4 - 7
# Alpha 8 - 12
# Beta 12 - 30


class EEG:
    """
    This class encapsulates all information related to this EEG, 17 different data are extracted
    This class is implemented like a collection, where each bucket of data represents an iteration.
    The block size depends on window_size parameter given.
    The loop count depends on numer_blocks parameter.
    """

    def __init__(self):
        """
        Constructor that initializes an entity.
        """
        self.__sampleRate__ = kernel.get_sample_rate() or 256
        self.__channels__ = kernel.get_channels() or 14
        self.__window_size__ = kernel.get_window_size() or 256
        self.__number_blocks__ = kernel.get_number_blocks() or -1
        self.__bands__ = kernel.get_bands() or {'delta': [0, 4], 'theta': [4, 7], 'alpha': [8, 12], 'beta': [12, 30],
                                                'gamma': [30, 40]}
        self.__total__: np.array = None
        self.__data__: np.array = None
        self.__fourier__ = list()
        self.__is_full__ = False
        self.__counter__ = 0
        self.__actual_blocks__ = 0
        self.__delta_avg__ = 0
        self.__theta_avg__ = 0
        self.__alpha_avg__ = 0
        self.__beta_avg__ = 0
        self.__gamma_avg__ = 0
        self.__delta__ = 0
        self.__theta__ = 0
        self.__alpha__ = 0
        self.__beta__ = 0
        self.__gamma__ = 0
        self.__hjorth_complexity__ = 0
        self.__hjorth_mobility__ = 0
        self.__hjorth_activity__ = 0
        self.__pfd__ = 0
        self.__kfd__ = 0
        self.__hfd__ = 0
        self.__engagement__ = 0

    def count(self):
        if self.__total__ is None:
            return 0
        return len(self.__total__)

    def add(self, data):
        """
        Adds a block of data, the size must be of specified size in configuration,
        if the size is wrong or the collection is full an exception will be raised
        :param data: Amount of data to be processed.
        """
        if len(data) > 0:
            if self.__is_full__ and self.__number_blocks__ != -1:
                raise Exception("Full ")
            else:
                if len(data) % self.__window_size__ != 0:
                    raise Exception("Not divisible " + str(self.__window_size__))
                if self.__total__ is None:
                    self.__total__ = data
                else:
                    self.__total__ = np.concatenate((self.__total__, data))

                self.__actual_blocks__, mod = divmod(len(self.__total__), self.__window_size__)
                if self.__number_blocks__ != -1:  # Create as many as possible
                    self.__actual_blocks__ = self.__number_blocks__  # As many as possible based on __windowSize__

            self.__is_full__ = self.__number_blocks__ == -1 or self.__number_blocks__ == self.__actual_blocks__

    def is_full(self):
        """
        Tells if the collection is full.
        :return: True if full, false otherwise.
        """
        return self.__is_full__

    def reset(self):
        """
        Resets the collection, all data will be cleared and indexes will be reset.
        """
        self.__fourier__ = list()
        self.__counter__ = 0
        self.__number_blocks__ = -1
        self.__is_full__ = False
        self.__total__ = None
        self.__data__ = None

    def __iter__(self):
        """
        Internal Python method to iterate, while there be blocks to return the iterate method will return them.
        :return: A block of data till be empty.
        """
        while self.__counter__ < self.__actual_blocks__:
            self.__fourier__.clear()
            self.__data__ = self.__total__[
                            (self.__window_size__ * self.__counter__):(self.__window_size__ * (self.__counter__ + 1))]
            self.__means__ = np.mean(self.__data__,
                                     axis=0)  # la media de los canales, cada elemento tiene la media de una canal
            try:
                for i in range(self.__channels__):
                    self.__fourier__.append(self.set_fourier_transform_at(i))
                self.extract()
            except Exception as e:
                print(e)
            yield self
            self.__counter__ += 1

    def extract(self):
        """
        Processes all data and set all internal values.
        """
        bandValues = {b: 0 for b in self.__bands__.keys()}
        for r in self.get_average_band_values():
            for bandName in bandValues:
                bandValues[bandName] += r[bandName]
        self.__delta_avg__ = bandValues["delta"] / self.__channels__
        self.__theta_avg__ = bandValues["theta"] / self.__channels__
        self.__alpha_avg__ = bandValues["alpha"] / self.__channels__
        self.__beta_avg__ = bandValues["beta"] / self.__channels__
        self.__gamma_avg__ = bandValues["gamma"] / self.__channels__
        alpha, beta, theta = np.mean(self.__alpha_avg__), np.mean(self.__beta_avg__), np.mean(self.__theta_avg__)
        self.__engagement__ = beta / (alpha + theta)

        bandValues = {b: 0 for b in self.__bands__.keys()}
        for r in self.get_band_signals():
            for bandName in bandValues:
                bandValues[bandName] += r[bandName]

        for bandName in bandValues:
            bandValues[bandName] = np.abs(np.sum(bandValues[bandName]))

        self.__delta__ = bandValues["delta"] / self.__channels__
        self.__theta__ = bandValues["theta"] / self.__channels__
        self.__alpha__ = bandValues["alpha"] / self.__channels__
        self.__beta__ = bandValues["beta"] / self.__channels__
        self.__gamma__ = bandValues["gamma"] / self.__channels__

        self.set_hjorth_complexity()
        self.set_hjorth_mobility()
        self.set_hjorth_activity()

        self.set_pfd()
        self.set_kfd()
        self.set_hfd()

    def get_engagement(self):
        """
        Get The engagement value
        :return: The engagement value.
        """
        return self.__engagement__

    def get_delta_avg(self):
        """
        Get delta average value.
        :return: Delta average value.
        """
        return self.__delta_avg__

    def get_theta_avg(self):
        """
        Get theta average value.
        :return: Theta average value.
        """
        return self.__theta_avg__

    def get_alpha_avg(self):
        """
        Get alpha average value.
        :return: Alpha average value.
        """
        return self.__alpha_avg__

    def get_beta_avg(self):
        """
        Get beta average value.
        :return: Beta average value.
        """
        return self.__beta_avg__

    def get_gamma_avg(self):
        """
        Get gamma average value.
        :return: Gamma average value.
        """
        return self.__gamma_avg__

    def get_delta(self):
        """
        Get delta total value.
        :return: Delta value.
        """
        return self.__delta__

    def get_theta(self):
        """
        Get theta total value.
        :return: theta value.
        """
        return self.__theta__

    def get_alpha(self):
        """
        Get theta total value.
        :return: theta value.
        """
        return self.__alpha__

    def get_beta(self):
        """
        Get beta total value.
        :return: beta value.
        """
        return self.__beta__

    def get_gamma(self):
        """
        Get gamma total value.
        :return: gamma value.
        """
        return self.__gamma__

    def get_hjorth_activity(self):
        """
        Get Hjorth activity.
        :return: Hjorth activity value.
        """
        return self.__hjorth_activity__

    def get_hjorth_mobility(self):
        """
        Get Hjorth mobility.
        :return: Hjorth mobility value.
        """
        return self.__hjorth_mobility__

    def get_hjorth_complexity(self):
        """
        Get Hjorth complexity.
        :return: Hjorth complexity value.
        """
        return self.__hjorth_complexity__

    def get_pfd(self):
        """
        Get Petrosian fractal dimension.
        :return: Petrosian fractal dimension value.
        """
        return self.__pfd__

    def get_kfd(self):
        """
        Get Katz fractal dimension.
        :return: Katz fractal dimension value.
        """
        return self.__kfd__

    def get_hfd(self):
        """
        Get Highuchi fractal dimension.
        :return: Higuchi fractal dimension value.
        """
        return self.__hfd__

    def set_fourier_transform_at(self, channel):
        """
        Return the normalized componente to window i-electrode,
        normalized means the value per each minus the average value
        :param channel: Channel to compute value.
        """
        norm = list(map(lambda x: x - self.__means__[channel], self.__get_raw_data_at__(channel)))
        function = np.hanning(len(self.__data__))
        fft = np.fft.fft(norm * function)
        return fft

    def get_fourier_transform_at(self, channel):
        """
        Get the fourier transform for this channel.
        :param channel: Channel to get value.
        :return: Fourier value for this channel
        """
        return self.__fourier__[channel]

    """
    Devuelve unos límites inferior y superior que corresponden a los límites de frecuencia de datos
    que se pasan como parámetro pero sobre la transformada de Fourier.
    """

    def get_bounds_for_band(self, band_bounds):
        """
        Return lower and upper bounds that correspond to data frequency limits
        passed as parameter over Fourier transform.
        :param band_bounds: Band bounds used to compute.
        :return: A tuple with band bounds frequency limits.
        """
        return tuple(map(lambda val: int(val * self.__window_size__ / self.__sampleRate__), band_bounds))

    def get_average_band_values(self):
        """
        Calls get_average_band_values_at per each channel.
        :return: All average band values per each channel.
        """
        total = []
        for i in range(self.__channels__):
            total.append(self.get_average_band_values_at(i))
        return total

    def get_average_band_values_at(self, i):
        """
        This method uses get_magnitudes_at results for channel i making an average per each
        frequency band amplitude per each channel.
        :param i: Channel to compute.
        :return: Channel average.
        """
        magnitudes = abs(self.get_fourier_transform_at(i))
        bands_values = {}
        for key in self.__bands__:
            bounds = self.get_bounds_for_band(self.__bands__[key])
            bands_values[key] = np.mean(magnitudes[bounds[0]:bounds[1]] / self.__window_size__)
        return bands_values

    def get_band_signals(self):
        """
        Calls get_band_signals_at per each channel.
        :return: Get all band signals.
        """
        total = []
        for i in range(self.__channels__):
            value = self.get_band_signals_at(i, self.__bands__)
            total.append(value)
        return total

    def get_band_signals_at(self, i, bands):
        """
        Recreates signal related to bands for i channel.
        :param i: Channel to compute.
        :param bands: Band to compute.
        :return: Numpy array with recreated signal.
        """
        fft = self.get_fourier_transform_at(i)
        band_signals = {}
        for key in self.__bands__:
            bounds = self.get_bounds_for_band(bands[key])
            band_signals[key] = self.rebuild_signal_from_DFT(fft, bounds)
        return band_signals

    def rebuild_signal_from_DFT(self, dft, bounds=None):
        """
        Rebuild Fourier inverse transform
        :param dft: Fourier data to rebuild.
        :param bounds: Bound limit to filter, all values below or upper will be set as zero.
        :return: Inverse Fourier Transform.
        """
        data = list()
        for i, value in enumerate(dft):
            if value < bounds[0] or value > bounds[1]:
                data.append(0)
            else:
                data.append(value)

        ifourier = np.fft.ifft(data)
        return ifourier

    def __get_raw_data_at__(self, i):
        """
        Get the data stored at i-channel in raw format, it means before any process.
        :param i: Channel to return data.
        :return: Data in original format.
        """
        return self.__data__[:, i]

    def get_PFD_at(self, channel):
        """
        Get Petrosial Fractal Dimension.
        :param channel: Channel to compute Petrosian fractal.
        :return: Petrosian fractal dimension.
        """
        channel_data = self.__get_raw_data_at__(channel)
        pfd = entropy.petrosian_fd(channel_data)
        return pfd

    def set_pfd(self):
        """
        Computes Petrosian fractal dimension.
        """
        self.__pfd__ = np.nan_to_num(np.mean([self.get_PFD_at(i) for i in range(self.__channels__)]), nan=self.__pfd__)

    def get_pfd(self):
        """
        Get Petrosian fractal dimension for all channels.
        :return: Petrosian fractal dimension.
        """
        return self.__pfd__

    def get_HFD_at(self, channel, kMax=None):
        """
        Computes Higuchi Fractal dimension per channel.
        :param channel: Channel to compute Higuchi Fractal dimension.
        :param kMax: kmax required for Higuchi.
        :return: Higuchi fractal dimension.
        """
        channel_data = self.__get_raw_data_at__(channel)
        n = len(channel_data)
        kMax = (n // 2) - 1 if kMax is None else kMax
        data = entropy.higuchi_fd(channel_data, kMax)
        return data

    def set_hfd(self):
        """
        Computes Higuchi fractal dimension.
        """
        hfd = np.mean([self.get_HFD_at(i) for i in range(self.__channels__)])
        self.__hfd__ = np.nan_to_num(hfd, self.__hfd__, nan=self.__hfd__)

    def get_hfd(self):
        """
        Get Higuchi fractal dimension for all channels.
        :return: Higuchi fractal dimension.
        """
        return self.__hfd__

    """
    Katz Fractal Dimension per channel
    """

    def get_KFD_at(self, channel):
        """
        Get Katz fractal dimension per channel.
        :param channel: Channel to compute Katz Fractal dimension.
        :return: Katz fractal dimension.
        """
        channel_data = self.__get_raw_data_at__(channel)
        data = entropy.katz_fd(channel_data)
        return data

    def set_kfd(self):
        """
        Computes Katz fractal dimension.
        """
        self.__kfd__ = np.nan_to_num(np.mean([self.get_KFD_at(i) for i in range(self.__channels__)]), nan=self.__kfd__)

    def get_kfd(self):
        """
        Get Katz fractal dimension for all channels.
        :return: Katz fractal dimension.
        """
        return self.__kfd__

    def set_hjorth_complexity(self):
        """
        Computes Hjorth complexity.
        """
        hjorth = Hjorth()
        data = np.mean(
            [hjorth.hjorth_complexity(self.__get_raw_data_at__(channel)) for channel in range(self.__channels__)])
        self.__hjorth_complexity__ = np.nan_to_num(data, nan=self.__hjorth_complexity__)

    def set_hjorth_mobility(self):
        """
        Computes Hjorth mobility.
        """
        hjorth = Hjorth()
        data = np.mean(
            [hjorth.hjorth_mobility(self.__get_raw_data_at__(channel)) for channel in range(self.__channels__)])
        self.__hjorth_mobility__ = np.nan_to_num(data, nan=self.__hjorth_mobility__)

    def set_hjorth_activity(self):
        """
        Computes Hjorth activity.
        """
        hjorth = Hjorth()
        data = np.mean(
            [hjorth.hjorth_activity(self.__get_raw_data_at__(channel)) for channel in range(self.__channels__)])
        self.__hjorth_activity__ = np.nan_to_num(data, nan=self.__hjorth_activity__)


class Hjorth:

    @staticmethod
    def hjorth_activity(data):
        """
        Computes Hjorth activity on selected data.
        :param data: Data to compute.
        :return: Hjorth activity on selected data.
        """
        return np.var(data)

    @staticmethod
    def hjorth_mobility(data):
        """
        Computes Hjorth mobility on selected data.
        :param data: Data to compute.
        :return: Hjorth mobility on selected data.
        """
        var = np.var(data)
        if var == 0:
            return 0
        else:
            return np.sqrt(np.var(np.gradient(data)) / var)

    def hjorth_complexity(self, data):
        gradient = np.gradient(data)
        hjorth = self.hjorth_mobility(data)
        if hjorth == 0:
            return 0
        else:
            return self.hjorth_mobility(gradient / hjorth)
