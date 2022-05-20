from signal_classes import ECG
from glob import glob
import os.path
# list_of_files = glob('../data/*.csv')
# list_of_files = ['0000.csv']
list_of_files = glob('*.csv')
for csv_file in list_of_files:
    json_name = '.'.join(csv_file.split('.')[0:-1]) + '.json'

    if os.path.isfile(json_name):
        print(json_name, " already exists")
        continue
    ecg = ECG(csv_file)
    ecg.invert_ecg()
    ecg.save_processed_data()
