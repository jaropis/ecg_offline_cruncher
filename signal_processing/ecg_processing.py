# import peakutils- t be reused
import warnings
import numpy as np
from mne.preprocessing import ecg

old_err = np.seterr(divide='raise')


def find_max_qrs(qrs_pos, ecg_line, dist = 100):
    if qrs_pos - dist > 0:
        return np.argmax(ecg_line[qrs_pos - dist: qrs_pos + dist]) + (qrs_pos - dist)
    else:
        return np.argmax(ecg_line[: qrs_pos + dist])


def clean_peaks(qrs_complexes, dist=20):
    # resolving the quirks of the qrs detector used - it tends to put qrs complexes on top of one another, i.e. detect
    # a few complexes at the same place - what I want to do is replace a "run" of similar qrs's by a single, median v.
    # of them
    new_peaks = []  # this list will hold the filtered qrs complexes
    accumulator = []  # this list will hold peaks which are close to one another
    push = True
    for idx in range(0, len(qrs_complexes)-1):
        if qrs_complexes[idx+1] - qrs_complexes[idx] < dist:
            accumulator.append(qrs_complexes[idx])
            push = False
        elif not push:  # this will be the last of close-by qrs complexes
            accumulator.append(qrs_complexes[idx])
            new_peaks.append(int(np.median(accumulator)))
            push = True
            accumulator = []
        else:
            assert (len(accumulator) == 0), "the accumulator length is not 0!"
            assert push, "pop is not True!"
            new_peaks.append(qrs_complexes[idx])

    # now for the last qrs
    if len(accumulator) == 0:
        new_peaks.append(qrs_complexes[idx+1])
    else:
        accumulator.append(qrs_complexes[idx+1])
        new_peaks.append(int(np.median(accumulator)))
    return np.array(new_peaks)


def get_template(qrs_voltage, qrs_position, template_length=101, n_sample=8, hi_corr=0.8):
    """
    this function calculates the template qrs which later on will be convolved with the voltage to identify outstanding
    qrs complexes
    :param qrs_voltage: the ECG
    :param qrs_position: found positions of QRSs
    :param template_length: an odd number which indicates the length of the template - odd so that the R is in the middle
    :param n_sample: how many qrs's will be drawn from all to form a template
    :param hi_corr the cut-off for correlation to be considered high
    :return:
    """
    np.random.seed(777)
    try:
        random_qrs = np.random.choice(qrs_position[2:], n_sample, replace=False)
    except ValueError:
        return np.array([-1])
    # this dictionary will hold the correlations between the templates
    correlations = {}
    for idx in range(n_sample):
        for idy in range(idx, n_sample):
            if idx != idy:
                correlations[(idx, idy)] = np.corrcoef(
                    qrs_voltage[random_qrs[idx] - int((template_length - 1) / 2):random_qrs[idx] + int((template_length - 1) / 2) + 1],
                    qrs_voltage[random_qrs[idy] - int((template_length - 1) / 2):random_qrs[idy] + int((template_length - 1) / 2) + 1]
                )[0, 1]

    # now selecting the template - dropping the QRSs whose correlation is too small
    high_correlations = []  # this list holds correlations which are "high"
    for corre in correlations:
        if correlations[corre] > hi_corr:
            high_correlations.extend(corre)
    high_correlations = list(set(high_correlations))

    # calculating template
    template = np.zeros(template_length)
    for idx in random_qrs[high_correlations]:
        template += qrs_voltage[idx - int((template_length - 1) / 2):idx + int((template_length - 1) / 2) + 1]
    template = template / n_sample

    # for idx in range(n_sample):
    #     plt.subplot(3, 3, idx+1)
    #     plt.plot(qrs_voltage[random_qrs[idx] - int((template_length-1)/2):random_qrs[idx] + int((template_length-1)/2) + 1])
    # plt.subplot(3, 3, 9)
    # plt.plot(template, color='red')
    # plt.show()
    return template


def correlation_machine(vector, template):
    extended_vector = np.concatenate([vector, template * 0])
    rolling_correlations = []
    for idx in range(0, len(vector)):
        # h..jowo zaimplementowana korelacja w corrcoef
        try:
            rolling_correlations.append(np.corrcoef(template, extended_vector[idx:idx+len(template)])[0, 1])
        except FloatingPointError as e:
            rolling_correlations.append(0)
    return np.array(rolling_correlations)


def find_correlated_peaks(correlations_vector, voltage, template, threshold=0.75, search_width=10):
    """
    function to actually look for peaks in the result of correlation machine - maxima ideally correspond to
    QRSs
    :param correlations_vector: the vector resulting from correlation machine between template and ECG
    :param voltage: actual ECG
    :param template: self explanatory
    :param threshold: what "similar" means - 0.85 means that correlation betwwen template and ECG segment is high enough
    for QRS to be identified
    :param search_width: how wide should the search be around the correlated maximum - the values in
    correlation_vectors drop rapidly at QRS's, within 2-3 samples, so the search within actual ECG is widened
    :return: the peaks identified by this correlate - locate - maximum procedure
    """
    template_length = len(template)
    template_ptp = np.ptp(template)
    boo = np.array(correlations_vector >= threshold)  # finding indices which are over the threshold
    indices = np.nonzero(boo[1:] != boo[:-1])[0] + 1  # finding indices of segments with only True or only False
    index_vector = np.arange(len(correlations_vector))
    regions = np.split(index_vector, indices)  # splitting correlations into true/false regions on threshold condition
    regions_over_thr = np.array(regions[0::2] if boo[0] else regions[1::2])  # selecting only "true" regions
    regions_over_thr += int(np.floor(template_length/2) + 1)  # moving half a template
    peaks = []
    # segment holds actual indices, not values - see above
    for segment in regions_over_thr:
        extended_segment = np.concatenate(
            [np.arange(segment[0] - search_width, segment[0]), segment, np.arange(segment[-1] + 1, segment[-1] + search_width + 1)]
        )
        peak_position = np.argmax(voltage[extended_segment]) + segment[0] - search_width
        # now I check whether the shape is not to low or too large
        if np.isnan(np.ptp(voltage[extended_segment])):
            print(voltage[extended_segment], segment)
        if 1/2*template_ptp < np.ptp(voltage[extended_segment]) < 2 * template_ptp:
            peaks.append(peak_position)
    return np.array(peaks)  # + int(np.floor(template_length/2) + 1)


def detect_r_waves(time_track, voltage, frequency=200):
    # current_position = 0
    # step = 1000
    # global_peaks = []
    # while current_position <= len(voltage):
    #     current_voltage_window = voltage[current_position:current_position + step]
    #     current_time_track_window = time_track[current_position:current_position + step]
    #     current_peak_indices = peakutils.indexes(current_voltage_window, thres=1 / 3., min_dist=step/10)
    #
    #     if 0 in current_peak_indices:
    #         current_peak_indices = np.delete(current_peak_indices, np.where(current_peak_indices==0)) # can't have a max at the beginning
    #     if step-1 in current_peak_indices:
    #         current_peak_indices = np.delete(current_peak_indices, np.where(current_peak_indices==step-1)) # or at the end
    #
    #     if current_peak_indices.size != 0:
    #         too_small_values = np.where(current_voltage_window[current_peak_indices] - np.min(current_voltage_window) < np.median(current_voltage_window[current_peak_indices]-np.min(current_voltage_window)) * 3/4)[0] # removing values which are clearly too small
    #         current_peak_indices = np.delete(current_peak_indices, too_small_values)
    #
    #         current_peak_indices = current_peak_indices + current_position
    #         global_peaks.extend(current_peak_indices)
    #
    #     current_position = current_position + step
    # adding 'start' here to keep track of the time
    global_peaks = ecg.qrs_detector(frequency, ecg=voltage, thresh_value=0.3,
                                    h_freq=99, l_freq=1, filter_length=200 * 3)
    global_peaks = np.array([find_max_qrs(_, voltage) for _ in global_peaks])
    global_peaks = clean_peaks(global_peaks)
    template = get_template(voltage, global_peaks, template_length=int(frequency / 4 + 1))
    if template == np.array([-1]):
        return np.transpose(np.array([np.array([]), np.array([])]))
    correlations = correlation_machine(voltage, template)
    global_peaks = find_correlated_peaks(correlations, voltage=voltage, template=template)
    peaks_values = [voltage[i] for i in global_peaks]
    peaks_positions = [time_track[i] for i in global_peaks]
    return np.transpose(np.array([np.array(peaks_positions), np.array(peaks_values)]))


def detect_ventriculars(r_waves_positions, r_waves_values):
    '''this function takes the detected R waves: both voltage and time and
    checks whether or not a peak is an artifact. For now an artifact will be
    anything with voltage over 2
    '''
    ventriculars_positions = np.where(np.logical_and(r_waves_values > 200, r_waves_values < 500))[0]
    return np.array([])


def detect_supraventriculars(r_waves_positions, r_waves_values):
    '''this function takes the detected R waves: both voltage and time and
    checks whether or not a peak is an artifact. For now an artifact will be
    anything with voltage over 2
    '''
    supraventriculars_positions = np.where(np.logical_and(r_waves_values > 500, r_waves_values < 1000))[0]
    return np.array([])# supraventriculars_positions

# this part (from now on) is for detecting artifact


def noise_profile(r_waves, time, signal, reasonable_rr, n_sample=100, half_template_length=50):
    '''
    this function collects the noise profile - it mirrors the QRS template function - first it randomly draws R-waves,
    then, calculates RR intervals for them, then leaves only those which do not exceed some reasonable value (calculated
    on the basis of the RR-filter, finds the average variance of the signal between R-waves and returns it as the
    profile
    :param r_waves:
    :param time:
    :param signal:
    :param reasonable_rr:
    :param n_sample: number of R-waves drawn for noise profile calculation
    :param half_template_length: the length of qrs template detection window
    :return:
    '''
    if r_waves.size == 0 or len(r_waves) < 6:
        return -1.0
    np.random.seed(777)
    if n_sample < len(r_waves):
        n_sample = len(r_waves) - 3
    random_r_pos = np.random.choice(range(2, len(r_waves)), n_sample, replace=False)
    random_rr = r_waves[random_r_pos] - r_waves[random_r_pos - 1]

    reasonable_r_pos_log = np.logical_and(random_rr > reasonable_rr[0], random_rr < reasonable_rr[1])
    reasonable_r_pos = random_r_pos[reasonable_r_pos_log]

    noise_between_rr = []
    for r in reasonable_r_pos:
        peak_position = np.where(time >= r_waves[r])[0][0]
        prev_peak_position = np.where(time >= r_waves[r-1])[0][0]
        # get segment and reject the beginning and end of the segment (it will comprise the QRS, P and Q
        sig_segment = signal[prev_peak_position + half_template_length:peak_position - half_template_length]
        noise_level = np.std(sig_segment)
        noise_between_rr.append(noise_level)

    # reject the extreme values of the noise using iqr and also widening it by 30%
    noise_range = np.percentile(noise_between_rr, [25, 75]) * np.array((0.7, 1.3))
    return noise_range[1]


def noise(r_waves, time_track, signal, half_template_length=50):
    """
    function for calculating noise between r-peaks
    :param r_waves:
    :param time_track:
    :param signal:
    :param half_template_length:
    :return:
    """
    rr_noise = [] # this list holds the resulting noise
    for idx in range(1, len(r_waves)):
        peak_position = np.where(time_track >= r_waves[idx])[0][0]
        prev_peak_position = np.where(time_track >= r_waves[idx - 1])[0][0]
        sig_segment = signal[prev_peak_position + half_template_length:peak_position - half_template_length]
        noise_level = np.std(sig_segment)
        rr_noise.append(noise_level)
    return rr_noise


def detect_artifacts(r_waves_positions, time_track, signal, rr_filter=(0.3, 1.75)):
    '''
    this function takes the detected R waves: this is based on the length of the interval and, if the length is a bit
    too big, the noise profile between the R-waves is checked
    '''

    reasonable_rr = np.array(rr_filter) * np.array((2, 0.75))
    current_noise_profile = noise_profile(r_waves_positions, time_track, signal, reasonable_rr)
    noise_for_all = noise(r_waves_positions, time_track, signal)
    artifact_positions = []

    for idx in range(1, len(r_waves_positions)):
        current_rr = r_waves_positions[idx] - r_waves_positions[idx - 1]
        if np.logical_or(current_rr < rr_filter[0], current_rr > rr_filter[1]):
            artifact_positions.append(idx)
        elif np.logical_or(current_rr < reasonable_rr[0], current_rr > reasonable_rr[1]) and noise_for_all[
            idx - 1] > current_noise_profile * 2:
            artifact_positions.append(idx)
        elif noise_for_all[idx - 1] > 4 * current_noise_profile:
            artifact_positions.append(idx)
    return artifact_positions