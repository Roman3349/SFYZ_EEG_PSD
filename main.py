import random

import numpy
import pandas
import matplotlib.pyplot as plt
import scipy

# Sampling frequency
sampling_frequency: int = 200


def read_eeg_data(file_name: str) -> numpy.ndarray:
    """
    Reads EEG data from CSV file.

    Parameters
    ----------
    file_name: str
        File name

    Returns
    -------
    numpy.ndarray
        EEG data
    """
    data = pandas.read_csv(file_name)['EEG'].to_numpy()
    index = random.randint(0, len(data) - sampling_frequency)
    return data[index:index + sampling_frequency]


def plot_psd(data: numpy.ndarray, label: str):
    """
    Plots power spectral density of EEG signal.

    Parameters
    ----------
    data: numpy.ndarray
        EEG data
    label: str
        Label
    """
    freqs, psd = scipy.signal.welch(data, sampling_frequency, nperseg=sampling_frequency)
    plt.plot(freqs, psd, label=label)


# Opened eyes
open_data = read_eeg_data('eyes-opened.csv')
# Closed eyes
closed_data = read_eeg_data('eyes-closed.csv')

plt.title('Výkonová spektrální hustota EEG signálu')
plot_psd(open_data, 'Otevřené oči')
plot_psd(closed_data, 'Zavřené oči')
plt.legend()
plt.xlabel('Frekvence [Hz]')
plt.ylabel('Výkonová spektrální hustota [μV²/Hz]')
plt.xlim([0, 50])
plt.show()
