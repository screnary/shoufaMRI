import organize_files_by_batch as org_batch
import find_qualifying_directories as find_dir


if __name__ == '__main__':
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
        print(f"将从 '{src_dir}' 复制文件到 '{tar_dir}'...")
        org_batch.organize_files_by_batch(directory=src_dir, out_directory=tar_dir, batch_size=40, mode=mode)
        if (i+1) % 10 == 0:
            progress = (i+1 / total_dirs) * 100
            print(f"进度: {progress:.2f}% ({i+1}/{total_dirs})")
