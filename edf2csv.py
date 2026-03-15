import numpy as np
import mne
import glob
import argparse
import warnings

edf_file_list = glob.glob("*.edf")

parser = argparse.ArgumentParser()
parser.add_argument("--signal")
parser.add_argument("--skipfileswith")
parser.add_argument("--justread", action="store_true")
parser.add_argument("--supresswarnings", action="store_true")
args = parser.parse_args()

if args.skipfileswith:
    edf_file_list = [x for x in edf_file_list if not (args.skipfileswith in x)]

for edf_file in edf_file_list:
    print(f"analyzing: {edf_file}")
    with warnings.catch_warnings():
        if args.supresswarnings:
            warnings.simplefilter("ignore", RuntimeWarning)
        raw = mne.io.read_raw_edf(edf_file, preload=True, stim_channel=None, verbose=False)
    polisomno_panda = raw.to_data_frame()
    if args.justread:
        print(list(polisomno_panda))
        break
    else:
        signal = args.signal if args.signal else "ECG"
        ecg = polisomno_panda[signal]
        time_track = np.cumsum(ecg * 0 + 1/200)
        csv_name = edf_file[0:-4] + ".csv"
        csv_file = open(csv_name, 'w')
        csv_file.write("time,voltage\n")
        for time, sample in zip(time_track, ecg):
            csv_file.write("%f,%f\n" % (time, sample))
        csv_file.close()
