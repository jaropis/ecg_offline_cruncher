import numpy as np

import plotly.graph_objs as go
import peakutils

import settings


def return_current_ECG(ECG_data, start, window):
    '''
    function takse a pandas dataframe with first column ECG, second column time track
    returns two wectors - ECG and time_track
    '''
    # using global variable for now - this will be rewritten
    return ECG_data.iloc[start:start + window, 0], ECG_data.iloc[start:start + window, 1]


def detect_r_waves(ECG_data):
    time_track = ECG_data.iloc[:, 0]
    voltage = ECG_data.iloc[:, 1]
    peak_indices = peakutils.indexes(voltage, thres=1 / 3., min_dist=200)
    # adding 'start' here to keep track of the time
    peaks_values = [voltage[i] + 0.1 for i in peak_indices]
    peaks_positions = [time_track[i] for i in peak_indices]
    return np.transpose(np.array([np.array(peaks_positions), np.array(peaks_values)]))


def detect_artifacts(r_waves):
    '''this function takes the detected R waves: both voltage and time and
    checks whether or not a peak is an artifact. For now an artifact will be
    anything with voltage over 2
    '''
    artifact_positions = np.where(R_waves[:, 1] > 2)[0]
    return R_waves[artifact_positions, :]


def return_RRs(R_waves):
    return np.diff(R_waves[:, 0])


def return_current_peaks(R_waves, ECG_data, start, window=3000):
    '''
    this function returns peak positions in the viewing window, peak values, and
    the position of the first peak in the R_waves trace (i.e. number of peak,
    rather than sample number)
    '''
    # adding 'start' here to keep track of the time
    time_of_start = ECG_data.iloc[:, 0][start]
    delta_t = np.diff(ECG_data.iloc[:, 0])[0]
    time = R_waves[:, 0]
    peaks = R_waves[:, 1]
    peaks_positions = time[np.logical_and(
        time >= time_of_start, time_of_start < time_of_start + window * delta_t)]
    peaks_values = peaks[np.logical_and(
        time >= time_of_start, time < time_of_start + window * delta_t)]
    index_of_first_peak = np.where(time == peaks_positions[0])  # this will come
    # in handy for establishing which peak was clicked in the window
    return peaks_positions, peaks_values, index_of_first_peak


def return_ECG_and_peak_traces(ECG_data, R_waves, artifacts, start, window=3000):
    '''
    ths function takes data as a pandas dataframe, starting position, window length
    and extracts the time_track, voltage, the time-positions of the peaks and the peak values.
    It returns two Dash tracks which are then used for the figure creation
    '''

    time_track, voltage = return_current_ECG(ECG_data, start, window)
    peak_positions, peak_values, settings.signal_data['ECG_I.csv']['app_state']['first_peak_position'] = return_current_peaks(
        R_waves, ECG_data, start, window)


    artifact_positions, artifact_values, settings.signal_data['ECG_I.csv']['app_state']['first_artifact_position'] = return_current_peaks(
        artifacts, ECG_data, start, window)

    return {'data': [go.Scatter(
        x=time_track, y=voltage,
        mode='lines',
        name="ECG trace"
    ),
        go.Scatter(
        x=peak_positions,
        y=peak_values,
        mode='markers',
        marker=dict(size=12, color='red'),
        name='Detected R-peaks'
    ),
        go.Scatter(
        x=artifact_positions,
        y=artifact_values,
        mode='markers',
        marker=dict(size=12, color='blue'),
        name='Detected artifacts'
    )],
        'layout': go.Layout(
        title="Example R-wave detection for the data I got from Jyotpal",
        hovermode='closest'
    )
    }

def return_poincare_plot(RR):
    N = len(RR)
    return {'data': [go.Scatter(
        x=RR[0:N - 1],
        y=RR[1:N],
        mode='markers',
        marker=dict(size=6, color='orange', opacity='0.2'),
        name='Poincare plot'
    )],
        'layout': go.Layout(
        title="Poincare plot<br>(<i>toggle off for faster scrolling</i>)",
        hovermode='closest'
    )
    }


def return_histogram(RR):
    return {'data': [go.Histogram(
        x=RR,
        histnorm='probability',
        name='Histogram of RR intervals'
    )],
        'layout': go.Layout(
        title="Histogram of RR intervals <br>(<i>toggle off for faster scrolling</i>)",
        hovermode='closest'
    )
    }
