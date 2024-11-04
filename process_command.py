import os
# import argparse


LIST_CMD = [
    "python getNewDicomData.py --data_phase Rest --surg_time Pre_Surgery > log_pre_rest.txt",
    "python getNewDicomData.py --data_phase Struc --surg_time Post_Surgery",
    "python getNewDicomData.py --data_phase Struc --surg_time Pre_Surgery"
]

# BASE_CMD = "python3 test_3d_transfer_fast_v2.py --class_offset {} --cuda_id {} --category {}\
#             --encode_scale 16 64 256 --nepoch 150 --model_suffix transfer_selfrec_SEcls_v4 \
#             --w_chamfer 10.0 --w_emd 50.0 --recon_w 50.0 --local_w 100.0 --cycle_w 50.0 --regular_w 0.05 --cls_w 0.0 --n_critic 1 --lambda_gp 1 \
#             --batch_size 1 --image_save_ep 1 --model_save_ep 5 --log_iter 100 \
#             --cls_category {} --task self --resume"

for cmd in LIST_CMD:
    print('=' * 60)
    print(cmd)
    print('=' * 60)

    os.system(cmd)

    # # Just for debug, comment this line below when running
    # exit(0)
