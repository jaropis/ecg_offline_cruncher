# import peakutils- t be reused
import warnings
import numpy as np
import numba as nb
from mne.preprocessing import ecg
from time import time
old_err = np.seterr(divide='raise')


@nb.jit(nopython=True)
def correlation_numba(x, y):
    """
    Manual correlation implementation for Numba compatibility
    Equivalent to np.corrcoef(x, y)[0, 1] but faster and Numba-compatible
    """
    n = len(x)
    if n == 0 or len(y) == 0:
        return 0.0
    
    # Check for NaN or infinite values
    has_nan_x = False
    has_nan_y = False
    for i in range(n):
        if not np.isfinite(x[i]):
            has_nan_x = True
            break
    for i in range(len(y)):
        if not np.isfinite(y[i]):
            has_nan_y = True
            break
    
    if has_nan_x or has_nan_y:
        return 0.0
    
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    
    # Calculate numerator and denominator
    num = 0.0
    sum_sq_x = 0.0
    sum_sq_y = 0.0
    
    for i in range(n):
        dx = x[i] - mean_x
        dy = y[i] - mean_y
        num += dx * dy
        sum_sq_x += dx * dx
        sum_sq_y += dy * dy
    
    den = np.sqrt(sum_sq_x * sum_sq_y)
    
    if den == 0.0 or not np.isfinite(den):
        return 0.0
    
    result = num / den
    if not np.isfinite(result):
        return 0.0
        
    return result


@nb.jit(nopython=True)
def find_max_qrs_numba(qrs_pos, ecg_line, dist=100):
    """Numba-optimized version of find_max_qrs"""
    if qrs_pos - dist > 0:
        start_idx = qrs_pos - dist
        end_idx = qrs_pos + dist
        segment = ecg_line[start_idx:end_idx]
        return np.argmax(segment) + start_idx
    else:
        segment = ecg_line[:qrs_pos + dist]
        return np.argmax(segment)


def find_max_qrs(qrs_pos, ecg_line, dist = 100):
    if qrs_pos - dist > 0:
        return np.argmax(ecg_line[qrs_pos - dist: qrs_pos + dist]) + (qrs_pos - dist)
    else:
        return np.argmax(ecg_line[: qrs_pos + dist])


@nb.jit(nopython=True)
def clean_peaks_numba(qrs_complexes, dist=20):
    """
    Numba-optimized version of clean_peaks
    Note: Uses pre-allocated arrays instead of dynamic lists
    """
    n = len(qrs_complexes)
    if n <= 1:
        return qrs_complexes.copy()
    
    # Pre-allocate arrays (worst case: no peaks are merged)
    new_peaks = np.zeros(n, dtype=qrs_complexes.dtype)
    accumulator = np.zeros(n, dtype=qrs_complexes.dtype)
    new_peaks_idx = 0
    accumulator_size = 0
    push = True
    
    for idx in range(n - 1):
        if qrs_complexes[idx + 1] - qrs_complexes[idx] < dist:
            accumulator[accumulator_size] = qrs_complexes[idx]
            accumulator_size += 1
            push = False
        elif not push:  # this will be the last of close-by qrs complexes
            accumulator[accumulator_size] = qrs_complexes[idx]
            accumulator_size += 1
            # Calculate median manually for Numba compatibility
            if accumulator_size > 0:
                acc_segment = accumulator[:accumulator_size]
                acc_sorted = np.sort(acc_segment)
                if accumulator_size % 2 == 1:
                    median_val = acc_sorted[accumulator_size // 2]
                else:
                    median_val = (acc_sorted[accumulator_size // 2 - 1] + acc_sorted[accumulator_size // 2]) / 2
                new_peaks[new_peaks_idx] = int(median_val)
                new_peaks_idx += 1
            push = True
            accumulator_size = 0
        else:
            new_peaks[new_peaks_idx] = qrs_complexes[idx]
            new_peaks_idx += 1
    
    # Handle the last qrs
    if accumulator_size == 0:
        new_peaks[new_peaks_idx] = qrs_complexes[n - 1]
        new_peaks_idx += 1
    else:
        accumulator[accumulator_size] = qrs_complexes[n - 1]
        accumulator_size += 1
        acc_segment = accumulator[:accumulator_size]
        acc_sorted = np.sort(acc_segment)
        if accumulator_size % 2 == 1:
            median_val = acc_sorted[accumulator_size // 2]
        else:
            median_val = (acc_sorted[accumulator_size // 2 - 1] + acc_sorted[accumulator_size // 2]) / 2
        new_peaks[new_peaks_idx] = int(median_val)
        new_peaks_idx += 1
    
    return new_peaks[:new_peaks_idx]


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

@nb.jit(nopython=True)
def correlation_machine_numba(vector, template):
    """
    Numba-optimized version of correlation_machine
    This should be MUCH faster than the original version
    """
    extended_vector = np.zeros(len(vector) + len(template))
    extended_vector[:len(vector)] = vector
    rolling_correlations = np.zeros(len(vector))
    
    for idx in range(len(vector)):
        segment = extended_vector[idx:idx+len(template)]
        rolling_correlations[idx] = correlation_numba(template, segment)
    
    return rolling_correlations

def correlation_machine(vector, template):
    extended_vector = np.zeros(len(vector) + len(template))
    extended_vector[:len(vector)] = vector
    rolling_correlations = []
    for idx in range(0, len(vector)):
        # h..jowo zaimplementowana korelacja w corrcoef
        # try:
        #     rolling_correlations.append(np.corrcoef(template, extended_vector[idx:idx+len(template)])[0, 1])
        # except FloatingPointError as e:
        #     rolling_correlations.append(0)
        rolling_correlations.append(np.corrcoef(template, extended_vector[idx:idx+len(template)])[0, 1])
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
    
    # Fix: Handle regions as a list instead of trying to create a homogeneous numpy array
    if len(regions) > 0:
        regions_over_thr = regions[0::2] if boo[0] else regions[1::2]  # selecting only "true" regions
    else:
        regions_over_thr = []
    
    peaks = []
    # segment holds actual indices, not values - see above
    for segment in regions_over_thr:
        if len(segment) == 0:
            continue
        segment_shifted = segment + int(np.floor(template_length/2) + 1)  # moving half a template
        extended_segment = np.concatenate(
            [np.arange(segment_shifted[0] - search_width, segment_shifted[0]), 
             segment_shifted, 
             np.arange(segment_shifted[-1] + 1, segment_shifted[-1] + search_width + 1)]
        )
        # Ensure indices are within bounds
        extended_segment = extended_segment[(extended_segment >= 0) & (extended_segment < len(voltage))]
        if len(extended_segment) == 0:
            continue
            
        peak_position = np.argmax(voltage[extended_segment]) + extended_segment[0]
        # now I check whether the shape is not to low or too large
        if np.isnan(np.ptp(voltage[extended_segment])):
            print(voltage[extended_segment], segment)
        if 1/2*template_ptp < np.ptp(voltage[extended_segment]) < 2 * template_ptp:
            peaks.append(peak_position)
    return np.array(peaks)  # + int(np.floor(template_length/2) + 1)


@nb.jit(nopython=True)
def find_correlated_peaks_numba(correlations_vector, voltage, template, threshold=0.75, search_width=10):
    """
    Numba-optimized version of find_correlated_peaks
    This version manually implements the region detection and peak finding logic
    """
    template_length = len(template)
    template_ptp = np.ptp(template)
    
    # Find regions above threshold
    above_threshold = correlations_vector >= threshold
    peaks = []
    
    i = 0
    while i < len(above_threshold):
        if above_threshold[i]:
            # Found start of a region above threshold
            region_start = i
            # Find end of region
            while i < len(above_threshold) and above_threshold[i]:
                i += 1
            region_end = i - 1
            
            # Process this region
            if region_end >= region_start:
                # Shift by half template length
                shift = int(np.floor(template_length/2) + 1)
                segment_start = region_start + shift
                segment_end = region_end + shift
                
                # Create extended search region
                search_start = max(0, segment_start - search_width)
                search_end = min(len(voltage) - 1, segment_end + search_width)
                
                if search_end > search_start:
                    # Find maximum in the extended region
                    max_val = voltage[search_start]
                    peak_position = search_start
                    
                    for j in range(search_start + 1, search_end + 1):
                        if voltage[j] > max_val:
                            max_val = voltage[j]
                            peak_position = j
                    
                    # Check if peak amplitude is reasonable
                    region_min = voltage[search_start]
                    region_max = voltage[search_start]
                    for j in range(search_start, search_end + 1):
                        if voltage[j] < region_min:
                            region_min = voltage[j]
                        if voltage[j] > region_max:
                            region_max = voltage[j]
                    
                    region_ptp = region_max - region_min
                    
                    if not np.isnan(region_ptp) and template_ptp/2 < region_ptp < 2 * template_ptp:
                        peaks.append(peak_position)
        else:
            i += 1
    
    return np.array(peaks)


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
    
    # Fix: Handle regions as a list instead of trying to create a homogeneous numpy array
    if len(regions) > 0:
        regions_over_thr = regions[0::2] if boo[0] else regions[1::2]  # selecting only "true" regions
    else:
        regions_over_thr = []
    
    peaks = []
    # segment holds actual indices, not values - see above
    for segment in regions_over_thr:
        if len(segment) == 0:
            continue
        segment_shifted = segment + int(np.floor(template_length/2) + 1)  # moving half a template
        extended_segment = np.concatenate(
            [np.arange(segment_shifted[0] - search_width, segment_shifted[0]), 
             segment_shifted, 
             np.arange(segment_shifted[-1] + 1, segment_shifted[-1] + search_width + 1)]
        )
        # Ensure indices are within bounds
        extended_segment = extended_segment[(extended_segment >= 0) & (extended_segment < len(voltage))]
        if len(extended_segment) == 0:
            continue
            
        peak_position = np.argmax(voltage[extended_segment]) + extended_segment[0]
        # now I check whether the shape is not to low or too large
        if np.isnan(np.ptp(voltage[extended_segment])):
            print(voltage[extended_segment], segment)
        if 1/2*template_ptp < np.ptp(voltage[extended_segment]) < 2 * template_ptp:
            peaks.append(peak_position)
    return np.array(peaks)  # + int(np.floor(template_length/2) + 1)


def find_correlated_peaks_wrapper(correlations_vector, voltage, template, threshold=0.75, search_width=10, use_numba=True):
    """
    Wrapper function that can switch between original and Numba implementations
    """
    if use_numba:
        return find_correlated_peaks_numba(correlations_vector, voltage, template, threshold, search_width)
    else:
        return find_correlated_peaks(correlations_vector, voltage, template, threshold, search_width)


@nb.jit(nopython=True)
def noise_numba(r_waves, time_track, signal, half_template_length=50):
    """
    Numba-optimized version of noise calculation
    :param r_waves: R-wave positions in time
    :param time_track: time array
    :param signal: ECG signal
    :param half_template_length: half template length
    :return: noise levels between R-waves
    """
    rr_noise = np.zeros(len(r_waves) - 1)
    
    # Optimization: Start search from previous position instead of searching entire array
    last_search_idx = 0
    
    for idx in range(1, len(r_waves)):
        # Find peak positions in the signal array - optimized search
        peak_position = -1
        prev_peak_position = -1
        
        # Search for previous peak position starting from last known position
        for i in range(last_search_idx, len(time_track)):
            if time_track[i] >= r_waves[idx - 1]:
                prev_peak_position = i
                break
        
        # Search for current peak position starting from previous peak
        if prev_peak_position != -1:
            for i in range(prev_peak_position, len(time_track)):
                if time_track[i] >= r_waves[idx]:
                    peak_position = i
                    last_search_idx = prev_peak_position  # Update starting position for next iteration
                    break
        
        # If we found both positions and there's enough space between them
        if (peak_position != -1 and prev_peak_position != -1 and 
            peak_position > prev_peak_position + 2 * half_template_length):
            start_idx = prev_peak_position + half_template_length
            end_idx = peak_position - half_template_length
            if end_idx > start_idx:
                sig_segment = signal[start_idx:end_idx]
                rr_noise[idx - 1] = np.std(sig_segment)
            else:
                rr_noise[idx - 1] = 0.0
        else:
            rr_noise[idx - 1] = 0.0
    
    return rr_noise


@nb.jit(nopython=True)
def noise_numba_fast(r_waves, time_track, signal, half_template_length=50):
    """
    Ultra-fast version assuming time_track is regularly sampled
    This version assumes time_track[i] corresponds to signal[i] with regular sampling
    """
    rr_noise = np.zeros(len(r_waves) - 1)
    
    # Calculate sampling rate (time per sample)
    if len(time_track) > 1:
        dt = (time_track[-1] - time_track[0]) / (len(time_track) - 1)
        time_offset = time_track[0]
        
        for idx in range(1, len(r_waves)):
            # Convert time positions to array indices directly
            prev_idx = int((r_waves[idx - 1] - time_offset) / dt)
            curr_idx = int((r_waves[idx] - time_offset) / dt)
            
            # Ensure indices are within bounds
            prev_idx = max(0, min(prev_idx, len(signal) - 1))
            curr_idx = max(0, min(curr_idx, len(signal) - 1))
            
            # Calculate noise if there's enough space
            if curr_idx > prev_idx + 2 * half_template_length:
                start_idx = prev_idx + half_template_length
                end_idx = curr_idx - half_template_length
                if end_idx > start_idx and end_idx <= len(signal):
                    sig_segment = signal[start_idx:end_idx]
                    rr_noise[idx - 1] = np.std(sig_segment)
                else:
                    rr_noise[idx - 1] = 0.0
            else:
                rr_noise[idx - 1] = 0.0
    
    return rr_noise


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
    global_peaks = np.array([find_max_qrs_numba(_, voltage) for _ in global_peaks])
    global_peaks = clean_peaks_numba(global_peaks)
    template = get_template(voltage, global_peaks, template_length=int(frequency / 4 + 1))
    if len(template) == 1 and template[0] == -1:
        return np.transpose(np.array([np.array([]), np.array([])]))
    start_time = time()    
    correlations = correlation_machine_numba(voltage, template)
    correlation_time = time() - start_time
    print(f"Numba correlation time: {correlation_time:.4f} seconds")
    
    # Test both implementations for comparison
    peak_start_time = time()
    global_peaks = find_correlated_peaks_numba(correlations, voltage=voltage, template=template)
    numba_peak_time = time() - peak_start_time
    
    print(f"Numba peak detection time: {numba_peak_time:.4f} seconds")
    print(f"Total Numba processing time: {correlation_time + numba_peak_time:.4f} seconds")
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


def noise(r_waves, time_track, signal, half_template_length=50, use_fast=True):
    """
    function for calculating noise between r-peaks
    :param r_waves:
    :param time_track:
    :param signal:
    :param half_template_length:
    :param use_fast: if True, use ultra-fast version assuming regular sampling
    :return:
    """
    # Use fast version for better performance if time_track is regularly sampled
    if use_fast:
        return noise_numba_fast(r_waves, time_track, signal, half_template_length).tolist()
    else:
        return noise_numba(r_waves, time_track, signal, half_template_length).tolist()


    return artifact_positions


@nb.jit(nopython=True)
def detect_artifacts_numba(r_waves_positions, time_track, signal, rr_filter_min=0.3, rr_filter_max=1.75, 
                          current_noise_profile=10.0):
    """
    Numba-optimized version of detect_artifacts
    Note: This version requires pre-computed noise profile since noise_profile() 
    uses functions not supported in nopython mode
    """
    reasonable_rr_min = rr_filter_min * 2.0
    reasonable_rr_max = rr_filter_max * 0.75
    
    # Calculate noise using Numba version
    print("Calculating noise profile with Numba")
    noise_for_all = noise_numba_fast(r_waves_positions, time_track, signal)
    print("noise detected")
    artifact_positions = []
    
    for idx in range(1, len(r_waves_positions)):
        current_rr = r_waves_positions[idx] - r_waves_positions[idx - 1]
        
        # Check basic RR interval limits
        if current_rr < rr_filter_min or current_rr > rr_filter_max:
            artifact_positions.append(idx)
        # Check extended RR limits with noise condition
        elif ((current_rr < reasonable_rr_min or current_rr > reasonable_rr_max) and 
              noise_for_all[idx - 1] > current_noise_profile * 2):
            artifact_positions.append(idx)
        # Check extreme noise condition
        elif noise_for_all[idx - 1] > 4 * current_noise_profile:
            artifact_positions.append(idx)
    
    return np.array(artifact_positions)


@nb.jit(nopython=True)
def noise_profile_numba(r_waves, time_track, signal, reasonable_rr_min, reasonable_rr_max, 
                       n_sample=50, half_template_length=50, seed=777):
    """
    Numba-optimized version of noise_profile (simplified)
    This version uses a simpler sampling strategy since np.random.choice isn't fully supported
    """
    if len(r_waves) < 6:
        return -1.0
    
    # Simple sampling instead of np.random.choice
    # Take every nth R-wave to get approximately n_sample points
    step = max(1, len(r_waves) // n_sample)
    
    noise_values = []
    
    for i in range(2, len(r_waves), step):
        if len(noise_values) >= n_sample:
            break
            
        current_rr = r_waves[i] - r_waves[i - 1]
        
        # Check if RR interval is reasonable
        if reasonable_rr_min < current_rr < reasonable_rr_max:
            # Find positions in signal array
            peak_position = -1
            prev_peak_position = -1
            
            for j in range(len(time_track)):
                if time_track[j] >= r_waves[i] and peak_position == -1:
                    peak_position = j
                if time_track[j] >= r_waves[i-1] and prev_peak_position == -1:
                    prev_peak_position = j
            
            if (peak_position > prev_peak_position + 2 * half_template_length and 
                peak_position != -1 and prev_peak_position != -1):
                start_idx = prev_peak_position + half_template_length
                end_idx = peak_position - half_template_length
                if end_idx > start_idx:
                    sig_segment = signal[start_idx:end_idx]
                    noise_level = np.std(sig_segment)
                    noise_values.append(noise_level)
    
    if len(noise_values) == 0:
        return 10.0  # Default fallback value
    
    # Simple approximation of 75th percentile * 1.3
    # Since np.percentile isn't available, use a simpler approach
    noise_array = np.array(noise_values)
    sorted_noise = np.sort(noise_array)
    p75_idx = int(len(sorted_noise) * 0.75)
    if p75_idx >= len(sorted_noise):
        p75_idx = len(sorted_noise) - 1
    
    return sorted_noise[p75_idx] * 1.3


def detect_artifacts(r_waves_positions, time_track, signal, rr_filter=(0.3, 1.75), use_numba=True):
    '''
    this function takes the detected R waves: this is based on the length of the interval and, if the length is a bit
    too big, the noise profile between the R-waves is checked
    '''
    if use_numba:
        # Use Numba-optimized version
        print("Using Numba for artifact detection")
        reasonable_rr = np.array(rr_filter) * np.array((2, 0.75))
        print("noise profile calculation")
        current_noise_profile = noise_profile_numba(r_waves_positions, time_track, signal, 
                                                   reasonable_rr[0], reasonable_rr[1])
        if current_noise_profile == -1.0:
            current_noise_profile = 10.0  # Fallback value
        print("Detecting artifacts with Numba")
        return detect_artifacts_numba(r_waves_positions, time_track, signal, 
                                    rr_filter[0], rr_filter[1], current_noise_profile)
    else:
        # Original implementation
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


def detect_artifacts_wrapper(r_waves_positions, time_track, signal, rr_filter=(0.3, 1.75), use_numba=True):
    """
    Wrapper function that can switch between original and Numba implementations
    """
    return detect_artifacts(r_waves_positions, time_track, signal, rr_filter, use_numba)