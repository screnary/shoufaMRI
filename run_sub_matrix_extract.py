import os

# 按需编辑下面的命令列表。可追加/注释掉任意条。
LIST_CMD = [
    # 批量提取 arrnaged_Whole_Brain 的子矩阵
    "python ./data/extract_matrix.py "
    "--data-flag arrnaged_Whole_Brain "
    "--networks 'CEN,DAN,DMN,FPN,SMN,VAN,VN,LN' "
    "--phase '02_post_surgery/04_MoCA_down_unchanged'"

    # 可多添加一些 "--phase '02_post_surgery/03…'"等
]


def main() -> None:
    for cmd in LIST_CMD:
        print("=" * 60)
        print(cmd)
        print("=" * 60)
        os.system(cmd)
        # # Just for debug
        # break


if __name__ == "__main__":
    main()
