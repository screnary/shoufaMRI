import os
import re
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pydicom
import pyvista as pv
import argparse
from tqdm import tqdm
import pdb

proj_root = os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))
data_root = os.path.join('/mnt/e/data/liuyang/', '2024shoufa')


def default_loader(path):
    v_data = None
    with np.load(path) as vfile:
        vdata = vfile.f.arr_0
        if vdata.shape[0] < 64:
            v_data = vdata.transpose((1, 2, 0))  # shape=(64,64,28)
        else:
            v_data = vdata
    return v_data


class Volume:
    def __init__(self, img_list, name_list, vid, save_dir):
        """
        construct volume instance
        :param img_list: slice data list, list of ndarray
        :param name_list: slice name list, dicom file name
        """
        # super().__init__() #if inherent from parent class
        self._vsize = (len(img_list), img_list[0].shape[0], img_list[0].shape[1])  # protected, volume size, the size of image stack: (img_num, img_h, img_w)
        self._vdata = np.asarray(img_list)
        self._vid = vid  # volume id
        self.save_dir = save_dir
        # self.vdata = self.normalize(np.asarray(img_list))
        self.name_slices = name_list  # slice name list
        self.vdata = self.normalize(mode="volume").astype('float32')

    def normalize(self, mode="volume"):
        """
        normalize the volume slices
        :param imarray:
        :param mode: ['volume', 'slice'], norm across volume or per slice
        :return:
        """
        min_v = np.min(self._vdata.reshape(self._vsize[0], -1), axis=1)
        max_v = np.max(self._vdata.reshape(self._vsize[0], -1), axis=1)
        if mode == "volume":
            min_value = min_v.min()
            max_value = max_v.max()
            return (self._vdata - min_value) / (max_value - min_value)
        elif mode == "slice":
            return (self._vdata - min_v[:,None,None]) / (max_v[:,None,None] - min_v[:,None,None])
        else:
            raise NotImplementedError

    def save2img(self):
        # save volume slices as images
        # create img res folder
        output_dir = os.path.join(self.save_dir, "imgs", "{:03d}".format(self._vid))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for i in range(self._vsize[0]):
            fname = "{:02d}_".format(i+1) + self.name_slices[i] + ".jpg"
            img_dir = os.path.join(output_dir, fname)
            # pdb.set_trace()
            img = np.array(self.vdata[i] * 255).astype('uint8')
            cv2.imwrite(img_dir, img)

    def show(self):
        for i in range(self._vsize[0]):
            cv2.imshow("vis volume slices", self.vdata[i])
            c = cv2.waitKey()
            if c == ord('q'):
                print("Quit by user")
                print("volume size: ", self._vsize)
                cv2.destroyAllWindows()
                quit()


class VolumeDataset:
    def __init__(self, args, loader=default_loader):
        # load volumes from files
        instance = args.instance
        type = args.type
        volume_dir_list = os.listdir(os.path.join(args.save_path, instance))
        volume_dir = [v_dir for v_dir in volume_dir_list if type in v_dir][0]
        self.volume_root = os.path.join(args.save_path, instance, volume_dir)
        self.volume_names = [vn for vn in sorted(os.listdir(self.volume_root)) if vn.startswith('volume_')]
        self.loader = loader
        self.volume_shape = self.load_volume(0)['data'].shape
        self.args = args

    def __len__(self) -> int:
        return len(self.volume_names)

    def load_volume(self, index):
        volume = {}
        v_path = os.path.join(self.volume_root, self.volume_names[index])
        volume['data'] = self.loader(v_path)
        volume['filename'] = '/'.join(v_path.split('/')[-5:])
        return volume

    def __getitem__(self, index: int):
        volume = self.load_volume(index)
        return volume

    def gen_volume_by_blend(self, src_index):
        # src_index: the list of source volume index, the len==28 (slice numbers)
        volume = {}
        volume['data'] = np.zeros(shape=self.volume_shape, dtype='uint8')
        volume['blend_idx'] = src_index
        volume['instance'] = self.args.instance
        volume['experiment_group'] = self.args.time_stamp
        for (i, idx) in enumerate(src_index):
            volume_tmp = self.load_volume(idx)
            volume['data'][..., i] = volume_tmp['data'][..., i]
        return volume


def natural_sort_key(s):
    # usage: sorted(files, key=natural_sort_key)
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]


def get_1ring_subdirs(root_dir=None):
    if root_dir is not None:
        files_tmp = os.listdir(root_dir)
        files = [os.path.join(root_dir, f) for f in files_tmp 
                 if (not f.startswith(".") and not f.startswith("nii_"))]
    else:
        files = os.listdir('.')
    subdirs = []
    for file in files:
        if os.path.isdir(file):
            subdirs.append(file)
    return subdirs


def dcm2nii_sitk(path_read, path_save):
    """_summary_

    Args:
        path_read (str): dicom file path
        path_save (_type_): _description_
    """
    reader = sitk.ImageSeriesReader()
    seriesIDs = reader.GetGDCMSeriesIDs(path_read)
    # pdb.set_trace()
    N = len(seriesIDs)
    lens = np.zeros([N])
    for i in range(N):
        dicom_names = reader.GetGDCMSeriesFileNames(path_read, seriesIDs[i])
        lens[i] = len(dicom_names)
    N_MAX = np.argmax(lens)
    dicom_names = reader.GetGDCMSeriesFileNames(path_read, seriesIDs[N_MAX])
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    sitk.WriteImage(image, path_save)


# TODO: propose all dicom files into images
# TODO: read all bold images and then separate them into volumes [check with Liu Yang]
# TODO: read from volume and random combine to generate training and testing data


def getVolume(fns, dicom_path, save_path, v_id=0):
    """
    construct volume from slices
    :param fns: file name list of dicom files
    :param dicom_path: dir containing dicom files of slices
    :param save_path: output dir to save volume data
    :param v_id: volume id, int
    :return:
    """
    sdata_list = []
    sname_list = []
    for slice_f in fns:
        ds = pydicom.dcmread(os.path.join(dicom_path, slice_f), force=True)
        slice_data = ds.pixel_array
        sdata_list.append(slice_data)
        sname_list.append(slice_f)
    volume = Volume(sdata_list, sname_list, v_id, save_path)
    return volume


def cvDicomData(args):
    # read DICOM files--BOLD files, and process them to Volume data structure
    # setup dirs, can be read from config inputs
    group_root = os.path.join(args.data_root, args.group, args.time_stamp)
    print("******************\n\tProcessing {} {}\n******************".format(args.group, args.time_stamp))
    instance_path_list = sorted(get_1ring_subdirs(group_root))
    for (i, instance_path) in enumerate(instance_path_list):
        instance_name = instance_path.split('/')[-1]
        dicom_phase_list = sorted(get_1ring_subdirs(instance_path))
        dicom_slice_path = dicom_phase_list[1]
        dicom_slice_suffix = dicom_slice_path.split('__')[-1]
        dicom_bold_path = dicom_phase_list[-1]
        dicom_bold_suffix = dicom_bold_path.split('__')[-1]
        slice_file = [f for f in sorted(os.listdir(dicom_slice_path))
                      if not f.startswith(".")]
        bold_file = [f for f in sorted(os.listdir(dicom_bold_path))
                     if f.startswith("MRI")]
        if not (len(slice_file) == args.slice_num and len(bold_file) == args.bold_num):
            print("dicom file num NOT VALID", len(slice_file), len(bold_file))
            pdb.set_trace()

        print("Processing instance {}, [{}/{}]...".format(instance_path.split('/')[-1], i+1, len(instance_path_list)))

        print("Processing Slices High Resolution...")
        save_path = os.path.join(args.save_path, instance_name, 'pose_fix-' + dicom_slice_suffix)
        fns = slice_file
        volume = getVolume(fns, dicom_slice_path, save_path, v_id=0)
        volume.save2img()

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(os.path.join(save_path, 'volume.npz'), 'wb') as f:
            vdata = (volume.vdata * 255).astype('uint8')
            np.savez(f, vdata)

        print("Processing volumes...")
        s_num = args.slice_num
        v_num = args.bold_num // args.slice_num
        save_path = os.path.join(args.save_path, instance_name, 'bold-' + dicom_bold_suffix)
        for j in tqdm(range(v_num)):
            st, ed = j * s_num, (j + 1) * s_num
            fns = bold_file[st:ed]
            try:
                volume = getVolume(fns, dicom_bold_path, save_path, v_id=j+1)
                volume.save2img()
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                with open(os.path.join(save_path, 'volume_{:05d}.npz'.format(j+1)), 'wb') as f:
                    vdata = (volume.vdata * 255).astype('uint8')
                    np.savez(f, vdata)
            except Exception as e:
                print("Error: ", e)
                print("An Error Occurred while processing volume_{}".format(j+1))


def checkDataInfo(args):
    # check the slices number
    # if args.group == "Ngroup" and args.time_stamp != "Baseline":  # temp deal
    #     time_stamp = args.time_stamp.split('_')[0]+'S_'+args.time_stamp.split('_')[1]
    #     args.time_stamp = time_stamp
    group_root = os.path.join(args.data_root, args.group, args.time_stamp)
    instance_path_list = sorted(get_1ring_subdirs(group_root))
    print(group_root)
    for (i, instance_path) in enumerate(instance_path_list):
        instance_name = instance_path.split('/')[-1]
        print("************************Instance: ", instance_name)
        dicom_phase_list = sorted(get_1ring_subdirs(instance_path))
        for phase_dir in dicom_phase_list:
            slice_file = [f for f in sorted(os.listdir(phase_dir)) if f.startswith("MRI")]
            print("phase: {} \n  **file num: {}".format(phase_dir.split('/')[-1], len(slice_file)))


def create_dataset_by_random_blend(args):
    # create new volumes by random sample slices from BOLD data
    # get instance list for this time stamp and Experiment Group, update args
    print("******************\n\tProcessing {}: {}\n******************".format(args.save_path, args.time_stamp))
    instance_list = [instance.split('/')[-1] for instance in sorted(get_1ring_subdirs(args.save_path)) if instance.split('/')[-1].startswith("Rat")]
    instance_num = len(instance_list)
    instance_datasize = args.exp_datasize // instance_num
    for (i, instance) in enumerate(instance_list):
        args.instance = instance
        print("Group {}: Instance {}: {}/{}".format(args.time_stamp, args.instance, i+1, instance_num))
        src_dataset = VolumeDataset(args)
        slice_num = args.slice_num
        src_datasize = len(src_dataset)
        for j in tqdm(range(instance_datasize)):
            select_idxs = np.random.randint(0, src_datasize, size=slice_num)
            volume = src_dataset.gen_volume_by_blend(select_idxs)

            save_path = os.path.join(args.exp_path)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            with open(os.path.join(save_path, '{:06d}_{}.npz'.format(i*instance_datasize + j, instance)), 'wb') as f:
                np.savez(f, volume)


def load_and_vis_volume(args):
    print("3D volume visualization")
    # load volume np.array from .npz
    instance = args.instance
    type= args.type
    volume_dir_list = os.listdir(os.path.join(args.save_path, instance))
    volume_dir = [v_dir for v_dir in volume_dir_list if type in v_dir][0]
    volume_root = os.path.join(args.save_path, instance, volume_dir)
    volume_names = [vn for vn in sorted(os.listdir(volume_root)) if vn.startswith('volume_')]
    vname = volume_names[0]
    volume_path = os.path.join(volume_root, vname)
    with np.load(volume_path) as vfile:
        vdata = vfile.f.arr_0
        v_data = vdata.transpose((1,2,0))  #shape=(64,64,28)
    vis(v_data, thresh=args.thresh)


def vis(volume, thresh=12):
    print("3D volume visualization")
    # get 3D pts
    # vname = self.volume_names[0]
    # volume_path = os.path.join(self.volume_root, vname)
    v_data = volume
    pts_loc = np.where(v_data >= thresh)  # pts locations
    # pdb.set_trace()
    pts_val = np.concatenate([v_data[pts_loc], np.array([0, 0])]) / 255.0
    pts = np.concatenate([pts_loc[0][:, None], pts_loc[1][:, None], pts_loc[2][:, None]], axis=-1)
    points = np.concatenate([pts, np.array([[0, 0, 0], [63, 63, 27]])], axis=0) / 63
    point_cloud = pv.PolyData(points)
    point_cloud['point_color'] = pts_val
    plotter = pv.Plotter()
    plotter.show_bounds(grid='back', location='outer', all_edges=False)
    plotter.add_axes(
        line_width=5,
        cone_radius=0.6,
        shaft_length=0.7,
        tip_length=0.3,
        ambient=0.5,
        label_size=(0.4, 0.16))
    plotter.add_mesh(point_cloud, scalars='point_color', cmap="viridis", opacity='linear', point_size=15,
                     render_points_as_spheres=True)
    plotter.show()


def check_blend_3D(args):
    save_path = os.path.join(args.exp_path)
    flist = os.listdir(save_path)
    fn = flist[6500]
    with np.load(os.path.join(save_path, fn), allow_pickle=True) as vfile:
        volume = vfile['arr_0'].item()
    v_data = volume['data']

    # pdb.set_trace()
    print("vis volume {} of {} {}".format(fn, volume['instance'], volume['experiment_group']))
    thresh = 12  # 60
    # pdb.set_trace()
    vis(v_data, thresh)


def gen_train_test_split(args):
    # output txt files, split by time_stamp; N:0, L:1
    time_stamp = args.time_stamp.split('_')[-1]  # '3d', '7d', '30d'
    gen_data_root = os.path.join(args.data_root, "../POCD_gen_data")
    if time_stamp == 'Baseline':
        data_dir_L = os.path.join(gen_data_root, 'Lgroup', time_stamp)
        data_dir_N = os.path.join(gen_data_root, 'Ngroup', time_stamp)
    else:
        data_dir_L = os.path.join(gen_data_root, 'Lgroup', 'L_'+time_stamp)
        data_dir_N = os.path.join(gen_data_root, 'Ngroup', 'N_'+time_stamp)
    flist_L = os.listdir(data_dir_L)
    len_L = len(flist_L)
    labels_L = np.ones(len_L)
    flist_N = os.listdir(data_dir_N)
    len_N = len(flist_N)
    labels_N = np.zeros(len_N)
    flist = np.asarray(flist_L + flist_N)
    labels = np.concatenate([labels_L, labels_N])
    idx = np.random.randint(0, len(flist), len(flist))
    train_split_fn = os.path.join(args.data_root, "../split_files", 'train_{}.txt'.format(time_stamp))  # train 80%
    val_split_fn = os.path.join(args.data_root, "../split_files", 'val_{}.txt'.format(time_stamp))  # validation 20%
    val_num = len(flist) // 5
    with open(train_split_fn, 'w') as f:
        flist_train = flist[idx][:-val_num]
        labels_train = labels[idx][:-val_num]
        # pdb.set_trace()
        for i in range(len(flist_train)):
            f.write("{} {}\n".format(flist_train[i], labels_train[i]))
    with open(val_split_fn, 'w') as f:
        flist_val = flist[idx][-val_num:]
        labels_val = labels[idx][-val_num:]
        for i in range(len(flist_val)):
            f.write("{} {}\n".format(flist_val[i], labels_val[i]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str,
                        default="/Volumes/DATA-exFAT/Works/Data/POCD/", #"/Volumes/DATA-exFAT/Works/Data/POCD/" #"/Volumes/DATA-exFAT/Works/Data/POCD/"; "/Volumes/ExtremeSSD/Works/Data/POCD/"
                        help='data root path in SSD.')
    parser.add_argument('--group', type=str, default="Lgroup",
                        choices=["Lgroup", "Ngroup", "LandSgroup"],
                        help='the data group selected')
    parser.add_argument('--time_stamp', type=str, default="3d",
                        choices=["3d", "7d", "30d", "Baseline"],
                        help="time stamp of the data")
    parser.add_argument('--slice_num', type=int, default=28,
                        help="the number of slices per volume")
    parser.add_argument('--bold_num', type=int, default=4200,
                        help="the number of slices in bold folder")
    parser.add_argument('--save_path', type=str, default=None,
                        help="the path to save imgs or volume data, default to be the same as instance root")
    parser.add_argument('--thresh', type=int, default=12,
                        help="the threshold for 3D volume")
    parser.add_argument('--instance', type=str, default="Rat-22",
                        help="instance name for visualize volume 3D")
    parser.add_argument('--type', type=str, default="bold",
                        help="slices type: bold or slice")
    parser.add_argument('--exp_datasize', type=int, default=50000,
                        help="the number of slices per volume")
    parser.add_argument('--exp_path', type=str, default=None,
                        help="the path to save expand volume data, default to be the same as instance root")
    args = parser.parse_args()

    if args.time_stamp != "Baseline":
        time_stamp = args.group[:-5] + '_' + args.time_stamp
        args.time_stamp = time_stamp

    if args.save_path is None:
        args.save_path = os.path.join(args.data_root, "../POCD_data", args.group, args.time_stamp)

    if args.exp_path is None:
        args.exp_path = os.path.join(args.data_root, "../POCD_gen_data", args.group, args.time_stamp)

    # checkDataInfo(args)
    # cvDicomData(args)
    load_and_vis_volume(args)
    # create_dataset_by_random_blend(args)
    # check_blend_3D(args)
    # gen_train_test_split(args)
