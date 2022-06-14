from threading import Thread
from multiprocessing import Process
import os
from time import perf_counter
from waymo_data_extraction_v2 import call_waymo_data_extraction

start_time = perf_counter()
path = "/data/waymo/validation-small"
obj = os.scandir(path)
print("Directories in '% s':" % path)

files = []  # list of data folders

for entry in obj:
    entry_path=os.path.abspath(entry)
    files.append(entry_path)

file_list = []
for file in files:
    print(file)
    file_list.append(Process(target=call_waymo_data_extraction, args=(file, "/data/waymo/orl_extraction",)))
    if len(file_list) < 2:
        continue
    print('======= list is full', len(file_list))
    for th in file_list:
        th.start()
    
    for th in file_list:
        th.join()
    file_list = []

end_time = perf_counter()
print(f'it took {end_time-start_time: 0.2f} secs to complete.')
