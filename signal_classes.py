import numpy as np
import pandas as pd
import json
from datetime import datetime
from . signal_processing.ecg_processing import detect_r_waves, detect_artifacts, detect_supraventriculars, \
                                               detect_ventriculars
# Now loading the HRA modules
from . HRAExplorer.signal_properties.RRclasses import RRSignal


class Signal:

    def __init__(self, file_path):
        self.file_path = file_path
        self.time_track, self.signal_values = self.get_data()
        self.sampling_period = self.get_sampling_period()
        self.sampling_rate = int(1/self.sampling_period)
        self.name = "default_file"

    def get_data(self):
        local_data = pd.read_csv(self.file_path)
        # assume the time is in the second / milisecond format
        time_track = local_data.iloc[:, 0]
        if " " in str(time_track[0]):
            # this is the OTHER format I got from Jyotpal - a possible bottleneck
            time_track = [datetime.strptime(_, "%d/%m/%Y %H:%M:%S.%f") for _ in local_data.iloc[:, 0]]
            time_track = np.array([_.hour*60*60 + _.minute * 60 + _.second + _.microsecond / 1000000 for _ in time_track])
            time_track = time_track - time_track[0] #TUTU
            signal_values = np.array(local_data.iloc[:, 1]) / (2 ** 8) # 8-bit ADAC?
        else:
            time_track = np.array(time_track)
            signal_values = np.array(local_data.iloc[:, 1])
        return time_track, signal_values

    def get_sampling_period(self):
        return np.diff(self.time_track)[0]


class ECG(Signal):

    def __init__(self, file_path, inverted=False):
        super().__init__(file_path)
        # constructing json filename
        self.loaded = False  # were the results loaded or calculated
        self.inverted = inverted
        self.r_waves_all_pos, self.r_waves_all_vals = None, None
        self.annotations = None
        self.ventriculars_pos, self.ventriculars_vals = None, None
        self.supraventriculars_pos, self.supraventriculars_vals = None, None
        self.artifacts_pos, self.artifacts_vals = None, None
        self.rr_filter = (0.3, 1.75)
        self.rr_intervals, self.rr_annotations = None, None
        self.RRSignal = None
        self.full_data = [dict(), dict()]  # this list holds results for normal [0] and inverted [1] ECG

        json_file = self.jsonify()
        try:
            data_file = open(json_file, 'r')
            self.get_preprocessed_data(data_file)
            self.loaded = True

        except FileNotFoundError:
            self.calculate_results()

    def jsonify(self):
        json_file = self.file_path.split('.')
        json_file[-1] = '.json'
        json_file = ''.join(json_file)
        return(json_file)

    def invert_ecg(self):
        self.inverted = np.invert(self.inverted)
        mean_ecg_value = np.mean(self.signal_values)
        self.signal_values = -1 * (self.signal_values - mean_ecg_value) + mean_ecg_value
        # and now repeating the constructor with the inverted ECG
        idx = 1 if self.inverted else 0
        if any(_ is None for _ in self.full_data[idx].values()) or not self.full_data[idx]:
            self.calculate_results()
            self.update_results_dict()
        else:
            self.get_vals_from_results_dict(idx=idx)

    def calculate_results(self):
        # unpacking a numpy array below
        self.r_waves_all_pos, self.r_waves_all_vals = detect_r_waves(self.time_track, self.signal_values).T
        self.annotations = self.annotate_r_waves()
        self.ventriculars_pos, self.ventriculars_vals = self.get_ventriculars()
        self.supraventriculars_pos, self.supraventriculars_vals = self.get_supraventriculars()
        self.artifacts_pos, self.artifacts_vals = self.get_artifacts()
        self.rr_intervals, self.rr_annotations = self.get_rrs()
        self.update_poincare()
        self.update_results_dict()

    def get_preprocessed_data(self, data_file):
        """
        This function gets data from .json file, loads it to a dictionary and then fills the various results
        :param data_file: the file opened by the constructor
        :return: None
        """
        self.full_data = json.load(data_file)

        for idx in (0, 1):  # get both, regular and inverted
            for data_key in self.full_data[idx]:
                self.full_data[idx][data_key] = np.array(self.full_data[idx].get(data_key))
        # you need to start with something, so start with the non-inverted ecg - either load the results, or calculate
        # them
        idx = 1 if self.inverted else 0
        self.get_vals_from_results_dict(idx)
        # with the inverted ecg, if it is on disk, load, if not, don't do anything

    def get_vals_from_results_dict(self, idx):
        """
        This function extracts the actual values from the loaded dictionary
        :param idx: whether the ecg should be inverted or not
        :return:
        """
        self.r_waves_all_pos, self.r_waves_all_vals = self.full_data[idx].get('r_waves_all_pos'), self.full_data[
            idx].get('r_waves_all_vals')
        self.annotations = self.full_data[idx].get('annotations')
        self.ventriculars_pos, self.ventriculars_vals = self.full_data[idx].get('ventriculars_pos'), self.full_data[
            idx].get('ventriculars_vals')
        self.supraventriculars_pos, self.supraventriculars_vals = self.full_data[idx].get('supraventriculars_pos'), \
                                                                  self.full_data[idx].get('supraventriculars_vals')
        self.artifacts_pos, self.artifacts_vals = self.full_data[idx].get('artifacts_pos'), self.full_data[idx].get(
            'artifacts_vals')
        self.rr_intervals, self.rr_annotations = self.full_data[idx].get('rr_intervals'), self.full_data[idx].get(
            'rr_annotations')
        self.update_poincare()

    def update_poincare(self):
        """
        this method updates the Poincare plot - it is used in two places, so it has been abstracted out
        :return:
        """
        if self.rr_intervals is not None:
            self.RRSignal = RRSignal([self.rr_intervals, self.rr_annotations], annotation_filter=(1, 2, 3))
            self.RRSignal.set_poincare()

    def update_results_dict(self):
        """
        This method puts the results in memory to results dictionary - it can e.g. update the results which are already
        there with the results for inverted ECG or can
        :return:
        """
        data_keys = ['r_waves_all_pos', 'r_waves_all_vals', 'annotations', 'ventriculars_pos', 'ventriculars_vals',
                     'supraventriculars_pos', 'supraventriculars_vals', 'artifacts_pos', 'artifacts_vals',
                     'rr_intervals', 'rr_annotations']

        idx = 1 if self.inverted else 0

        data_items = [self.r_waves_all_pos, self.r_waves_all_vals, self.annotations, self.ventriculars_pos,
                      self.ventriculars_vals, self.supraventriculars_pos, self.supraventriculars_vals,
                      self.artifacts_pos, self.artifacts_vals, self.rr_intervals, self.rr_annotations]

        for data_key, data_item in zip(data_keys, data_items):
            self.full_data[idx][data_key] = data_item

    def save_processed_data(self):
        json_file = self.jsonify()

        results = [dict(), dict()]
        for idx in (0, 1):
            for data_key in self.full_data[idx]:
                results[idx][data_key] = self.full_data[idx][data_key].tolist()

        with open(json_file, 'w') as data_file:
            json.dump(results, data_file)

    def detect_r_waves(self):
        return detect_r_waves(self.time_track, self.signal_values)

    def annotate_r_waves(self):
        '''
        input: self
        output: annotates the RR intervals
        '''
        annotations = np.zeros_like(self.r_waves_all_pos)
        artifacts = self.detect_artifacts()
        supraventriculars = self.detect_supraventriculars()
        ventriculars = self.detect_ventriculars()
        if len(supraventriculars) > 0:
            annotations[supraventriculars] = 2
        if len(ventriculars) > 0:
            annotations[ventriculars] = 1
        if len(artifacts) > 0:
            annotations[artifacts] = 3
        return annotations

    def detect_artifacts(self):
        if self.r_waves_all_pos:
            return detect_artifacts(self.r_waves_all_pos, self.time_track, self.signal_values)
        else:
            return np.array([])

    def detect_supraventriculars(self):
        if self.r_waves_all_pos:
            return detect_supraventriculars(self.r_waves_all_pos, self.r_waves_all_vals)
        else:
            return np.array([])

    def detect_ventriculars(self):
        if self.r_waves_all_pos:
            return detect_ventriculars(self.r_waves_all_pos, self.r_waves_all_vals)
        else:
            return np.array([])

    def get_rrs(self):
        """
        the function to get rr intervals from the r-waves annotations - the philosophy of this is that
        1) annotations of supraventricular and ventricular beats correspond to actual beats, whereas artifact beats are
        those which happen AFTER a noise sequence
        2) thus supra and ventricular beats yield pairs of RR intervals of non-sinus origin, whereas an artifact results
         in a single removed beat
        :return: rr values and rr annotations
        """

        rr_intervals = np.diff(self.r_waves_all_pos)
        rr_annotations = np.copy(self.annotations[1:])

        # annotating artifacts on the basis of filters

        for idx in (1, 2):
            avs_positions = np.where(rr_annotations == idx)[0]  # 1 is ventricular, 2 supra v -need to add ventriculars
            # and sup AFTER 1 in r-waves now seeing if the last RR is V - should not add anything higher than that
            end_of_rr = np.where(avs_positions == len(rr_annotations))[0]
            avs_positions = np.delete(avs_positions, end_of_rr)
            rr_annotations[avs_positions + 1] = idx
        return rr_intervals, rr_annotations

    def get_rr_annotations(self):
        pass

    def get_ventriculars(self):
        return self.r_waves_all_pos[self.annotations == 1], self.r_waves_all_vals[self.annotations == 1]

    def get_supraventriculars(self):
        return self.r_waves_all_pos[self.annotations == 2], self.r_waves_all_vals[self.annotations == 2]

    def get_artifacts(self):
        return self.r_waves_all_pos[self.annotations == 3], self.r_waves_all_vals[self.annotations == 3]
