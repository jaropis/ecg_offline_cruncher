import numpy as np
import mne
import glob


edf_file_list = glob.glob("*.edf")

for edf_file in edf_file_list:
    raw = mne.io.read_raw_edf(edf_file, preload=True, stim_channel=None)
    polisomno_panda = raw.to_data_frame()
    ecg = polisomno_panda.ECG
    time_track = np.cumsum(ecg * 0 + 1/200)
    csv_name = edf_file[0:-4] + ".csv"
    csv_file = open(csv_name, 'w')
    csv_file.write("time,voltage\n")
    for time, sample in zip(time_track, ecg):
        csv_file.write("%f,%f\n" % (time, sample))
    csv_file.close()