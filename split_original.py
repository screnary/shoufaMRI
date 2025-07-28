import utils.organize_files_by_batch as org_batch
import utils.dir_utils as find_dir


def main1():
    """
    202505
    """
    src_dir_list = find_dir.find_qualifying_directories(root_dir='/mnt/f/CFH_Original_Data')
    tar_dir_list = [path.replace('/mnt/f/', '/mnt/g/') for path in src_dir_list]

    total_dirs = len(src_dir_list)

    for i,src_dir in enumerate(src_dir_list):
        sub_num = find_dir.extract_subject_number(src_dir)
        tar_dir = tar_dir_list[i]
        if sub_num < 40:
            mode = 'parity'
        else:
            mode = 'half'
        print(f"MODE:'{mode}'. 将从 '{src_dir}' 复制文件到 '{tar_dir}'...")
        org_batch.organize_files_by_batch(directory=src_dir, out_directory=tar_dir, batch_size=40, mode=mode)
        if (i+1) % 10 == 0:
            progress = ((i+1) / total_dirs) * 100
            print(f"进度: {progress:.2f}% ({i+1}/{total_dirs})")


def main2():
    """
    202506, process sub_0038 and after subjects using 'parity' mode
    """
    src_dir_list = find_dir.find_qualifying_directories(root_dir='/mnt/e/data/liuyang/original_202505/CFH_Original_Data')
    tar_dir_list = [path.replace('original_202505', 'processed_202506') for path in src_dir_list]

    total_dirs = len(src_dir_list)

    for i,src_dir in enumerate(src_dir_list):
        sub_num = find_dir.extract_subject_number(src_dir)
        tar_dir = tar_dir_list[i]
        if sub_num >= 38:
            mode = 'parity'
        else:
            continue
        print(f"MODE:'{mode}'. 将从 '{src_dir}' 复制文件到 '{tar_dir}'...")
        org_batch.organize_files_by_batch(directory=src_dir, out_directory=tar_dir, batch_size=40, mode=mode)
        if (i+1) % 10 == 0:
            progress = ((i+1) / total_dirs) * 100
            print(f"进度: {progress:.2f}% ({i+1}/{total_dirs})")


if __name__ == '__main__':
    main2()
