'''
Script to fetch ABIDE pre-processed CPAC Pipeline :-)
'''

from nilearn import datasets
import os
import shutil

save_path = './nofilt_noglobal/' #path to save files
if not os.path.exists(save_path):
    os.makedirs(save_path)

abide = datasets.fetch_abide_pcp(data_dir=save_path, 
                               pipeline='cpac',
                               quality_checked=True)


for file_name in abide.keys():
    shutil.move(abide[file_name], os.path.join(save_path, file_name))

print("Dataset saved to at:", save_path)



