import rawpy
import numpy as np
import glob, os, math
import torch
from pathlib import Path
from PIL import Image as PILImage
from torchvision.transforms import transforms as tr
from concurrent.futures import ProcessPoolExecutor


def get_file_id(raw_path):
    """
    统一提取文件 ID。

    FiveK:
        a0004-xxx.dng -> a0004

    NOD:
        DSC_1558.NEF  -> DSC_1558
        DSC01881.ARW  -> DSC01881
    """
    return Path(raw_path).stem.split("-")[0]


def collect_raw_candidates(raw_root, ext):
    """
    在 raw_root 下收集指定后缀的 RAW 文件。
    不递归，只扫描当前目录。
    后缀大小写不敏感。
    """
    raw_root = Path(raw_root)
    ext = ext.lower()

    if not raw_root.exists():
        raise FileNotFoundError(f"RAW root not found: {raw_root}")

    raw_files = [
        str(p)
        for p in raw_root.iterdir()
        if p.is_file() and p.suffix.lower() == ext
    ]

    return sorted(raw_files)


def read_label_files(label_files, camera_model_file, split_name_to_label_name=None):
    label_dict = {}
    camera_label_dict = {}

    split_name_to_label_name = split_name_to_label_name or {}

    # 读取 camera_label.txt
    with open(camera_model_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split(', ')
            if len(parts) == 2:
                camera_name = parts[0]
                label = int(parts[1])
                camera_label_dict[camera_name] = label

    for label_file in label_files:
        label_file = Path(label_file)

        split_stem = label_file.stem

        # train_Canon_EOS_5D -> Canon_EOS_5D
        # train_Nikon750 -> Nikon750
        if split_stem.startswith("train_"):
            split_name = split_stem[len("train_"):]
        elif split_stem.startswith("test_"):
            split_name = split_stem[len("test_"):]
        elif split_stem.startswith("val_"):
            split_name = split_stem[len("val_"):]
        else:
            split_name = split_stem

        # 优先用 dataset_configs 里显式指定的 label_name
        candidates = []

        if split_name in split_name_to_label_name:
            candidates.append(split_name_to_label_name[split_name])

        candidates.append(split_name)
        candidates.append(split_name.replace("_", " "))

        camera = None
        for candidate in candidates:
            if candidate in camera_label_dict:
                camera = candidate
                break

        if camera is None:
            raise KeyError(
                f"Cannot find camera label for split file: {label_file}\n"
                f"split_name: {split_name}\n"
                f"tried candidates: {candidates}\n"
                f"available cameras: {list(camera_label_dict.keys())}"
            )

        label = camera_label_dict[camera]

        with open(label_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                file_id = get_file_id(line.strip())
                if file_id:
                    label_dict[file_id] = label

    return label_dict


def get_bayer_pattern(raw_pattern):
    pattern = np.array(['R', 'G', 'B', 'G'])
    return ''.join(pattern[raw_pattern.flatten()])


def flip(raw_img, flip, reverse=False):
    if flip == 3:
        raw_img = np.rot90(raw_img, k=2).copy()
    elif flip == 5:
        raw_img = np.rot90(raw_img, k=3 if reverse else 1).copy()
    elif flip == 6:
        raw_img = np.rot90(raw_img, k=1 if reverse else 3).copy()
    return raw_img


def norm(raw_img, black_level, white_level):
    raw_norm = (raw_img - black_level) / (white_level - black_level)
    raw_norm = np.clip(raw_norm, 0, 1)


def get_grid(img, patch_size, stride=None):
    grid = []
    if stride == None:
        stride = int(patch_size / 2)
    H, W, _ = img.shape
    for top in range(0, H - patch_size + 1, stride):
        for left in range(0, W - patch_size + 1, stride):
            grid.append([top, left])
    return grid


def patch_coordinates(
    image_height, image_width, num_patches, dividable_factor=1, overlap_factor=1
):
    assert num_patches > 1, "must be implemented for num_patches == 1"
    patch_width = (
        math.ceil(image_width / num_patches / dividable_factor * overlap_factor)
        * dividable_factor
    )
    patch_height = (
        math.ceil(image_height / num_patches / dividable_factor * overlap_factor)
        * dividable_factor
    )
    x_start_all = (np.linspace(0, 1, num_patches + 1)[:-1] * image_width).astype(
        np.int32
    )
    y_start_all = (np.linspace(0, 1, num_patches + 1)[:-1] * image_height).astype(
        np.int32
    )

    result = []

    for index_x, x_start in enumerate(x_start_all):
        for index_y, y_start in enumerate(y_start_all):
            x_end = x_start + patch_width
            y_end = y_start + patch_height

            if x_end > image_width:
                x_start = image_width - patch_width
                x_end = image_width

            if y_end > image_height:
                y_start = image_height - patch_height
                y_end = image_height

            result.append((x_start, y_start, x_end, y_end))

    return result


def paired_random_crop(raw, rgb, patch_size, grid=None):
    """随机裁剪配对的RAW和RGB图像，并生成扩充的参考区域
    Args:
        raw: RAW图像 [H, W, C]
        rgb: RGB图像 [2H, 2W, 3]
        patch_size: RAW图像的裁剪大小
        ref_scale: 参考区域相对于patch_size的倍数，默认为3
    Returns:
        cropped_raw: 裁剪后的RAW图像 [patch_size, patch_size, C]
        cropped_rgb: 裁剪后的RGB图像 [2*patch_size, 2*patch_size, 3]
        cropped_raw_ref: RAW参考区域 [ref_scale*patch_size, ref_scale*patch_size, C]
        cropped_rgb_ref: RGB参考区域 [2*ref_scale*patch_size, 2*ref_scale*patch_size, 3]
    """
    H, W, _ = raw.shape
    rgb_H, rgb_W, _ = rgb.shape

    if grid is None:
        # 随机选择裁剪位置
        top = np.random.randint(0, H - patch_size)
        left = np.random.randint(0, W - patch_size)
    else:
        top, left = grid

    # 裁剪raw和rgb
    cropped_raw = raw[top:top + patch_size, left:left + patch_size, :]

    rgb_top = top * 2
    rgb_left = left * 2
    rgb_patch_size = patch_size * 2
    cropped_rgb = rgb[
        rgb_top:rgb_top + rgb_patch_size,
        rgb_left:rgb_left + rgb_patch_size,
        :
    ]

    result = {'cropped_raw': cropped_raw, 'cropped_rgb': cropped_rgb}
    return result


def save_image(raw, rgb, patch_size, raw_target_path, rgb_target_path, file_id, kwargs=None):
    """保存裁剪后的图像对和参考图像
    Args:
        raw: RAW图像
        rgb: RGB图像
        patch_size: 裁剪大小
        raw_target_path: RAW图像保存路径
        rgb_target_path: RGB图像保存路径
        ref_target_path: RGB参考图像保存路径
        raw_ref_target_path: RAW参考图像保存路径
    """
    print(f'Processing {file_id}')
    raw_target_path = raw_target_path + '/' + file_id
    rgb_target_path = rgb_target_path + '/' + file_id

    if not os.path.exists(raw_target_path):
        os.makedirs(raw_target_path)
    if not os.path.exists(rgb_target_path):
        os.makedirs(rgb_target_path)

    bl = kwargs['bl'] if 'bl' in kwargs else None
    wl = kwargs['wl'] if 'wl' in kwargs else None
    bayer_pattern = kwargs['bayer_pattern'] if 'bayer_pattern' in kwargs else get_bayer_pattern(raw.pattern)
    label = kwargs['label'] if 'label' in kwargs else None
    num_crop = kwargs['num_crop'] if 'num_crop' in kwargs else 3
    dividable_factor = kwargs['dividable_factor'] if 'dividable_factor' in kwargs else 32

    # grids = get_grid(raw, patch_size, stride=kwargs['stride'] if 'stride' in kwargs else None)
    patch_coordinates_list = patch_coordinates(
        raw.shape[0],
        raw.shape[1],
        num_crop,
        dividable_factor=dividable_factor
    )

    for i, (x_start, y_start, x_end, y_end) in enumerate(patch_coordinates_list):

        cropped_raw = raw[y_start:y_end, x_start:x_end]
        cropped_rgb = rgb[y_start * 2:y_end * 2, x_start * 2:x_end * 2]

        raw_h, raw_w, c = cropped_raw.shape
        rgb_h, rgb_w, C = cropped_rgb.shape
        assert raw_h * 2 == rgb_h and raw_w * 2 == rgb_w and c == 4 and C == 3

        # 保存RAW patch
        if bl is not None and wl is not None:
            cropped_raw = cropped_raw.astype(np.uint16)
            np.savez_compressed(
                os.path.join(raw_target_path, f'{file_id}-{i}.npz'),
                raw=cropped_raw,
                bl=bl,
                wl=wl,
                bp=bayer_pattern,
                label=label
            )
        else:
            cropped_raw = cropped_raw.astype(np.float16)
            np.save(os.path.join(raw_target_path, f'{file_id}-{i}.npy'), cropped_raw)

        # 保存RGB patch
        PILImage.fromarray(cropped_rgb).save(
            os.path.join(rgb_target_path, f'{file_id}-{i}.jpg'),
            quality=100,
            subsampling=1
        )


def process_file(args):
    i, raw_target_path, rgb_target_path, patch_size, num_crop, stride, norm_mode, label_dict, dividable_factor = args

    file_id = get_file_id(i)

    # try:
    raw = rawpy.imread(i)
    label = label_dict.get(file_id, "Unknown")
    assert label != "Unknown", f"{file_id} label is not found"

    # 归一化
    bl = raw.black_level_per_channel[0]
    wl = raw.white_level

    im = raw.raw_image_visible
    if norm_mode == True:
        im = im.astype(np.float32)
        im = np.clip(im, a_min=bl, a_max=wl)
        im = (im - bl) / (wl - bl)
        if np.max(im) > 1 or np.min(im) < 0:
            raise Exception("图像数值超出范围！")
    else:
        if im.dtype != np.uint16:
            print(f'{file_id} 图像数据类型是{im.dtype}, 已转换为int32')
        im = im.astype(np.int32)

    im = np.expand_dims(im, axis=2)
    H = im.shape[0]
    W = im.shape[1]
    H = H if H % 2 == 0 else H - 1
    W = W if W % 2 == 0 else W - 1

    # pack操作
    pattern = str(raw.color_desc, encoding='utf-8')
    if pattern[raw.raw_pattern[0, 0]] == 'R':  # RGGB
        packed_raw = np.concatenate((
            im[0:H:2, 0:W:2, :],
            im[0:H:2, 1:W:2, :],
            im[1:H:2, 0:W:2, :],
            im[1:H:2, 1:W:2, :]
        ), axis=2)
        bayer_pattern = "RGGB"

    elif pattern[raw.raw_pattern[0, 0]] == 'B':  # BGGR
        packed_raw = np.concatenate((
            im[1:H:2, 1:W:2, :],
            im[0:H:2, 1:W:2, :],
            im[1:H:2, 0:W:2, :],
            im[0:H:2, 0:W:2, :]
        ), axis=2)
        bayer_pattern = "BGGR"

    elif pattern[raw.raw_pattern[0, 0]] == 'G' and pattern[raw.raw_pattern[0, 1]] == 'R':  # GRBG
        packed_raw = np.concatenate((
            im[0:H:2, 1:W:2, :],
            im[0:H:2, 0:W:2, :],
            im[1:H:2, 1:W:2, :],
            im[1:H:2, 0:W:2, :]
        ), axis=2)
        bayer_pattern = "GRBG"

    elif pattern[raw.raw_pattern[0, 0]] == 'G' and pattern[raw.raw_pattern[0, 1]] == 'B':  # GBRG
        packed_raw = np.concatenate((
            im[1:H:2, 0:W:2, :],
            im[0:H:2, 0:W:2, :],
            im[1:H:2, 1:W:2, :],
            im[0:H:2, 1:W:2, :]
        ), axis=2)
        bayer_pattern = "GBRG"

    rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=True)
    rgb = flip(rgb, raw.sizes.flip, reverse=True)

    kwargs = {
        'bl': bl,
        'wl': wl,
        'num_crop': num_crop,
        'stride': stride,
        'bayer_pattern': bayer_pattern,
        'label': label,
        'dividable_factor': dividable_factor
    }

    save_image(
        packed_raw,
        rgb,
        patch_size=patch_size,
        raw_target_path=raw_target_path,
        rgb_target_path=rgb_target_path,
        file_id=file_id,
        kwargs=kwargs
    )

    return file_id, True


if __name__ == '__main__':
    DEBUG = True

    DATASET_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = DATASET_DIR.parent

    dataset_configs = [
        {
            "name": "FiveK Canon",
            "camera_model": "Canon EOS 5D",
            "split_name": "Canon_EOS_5D",
            "label_name": "Canon EOS 5D",
            "raw_root": "/data2/dataset/xsb/SpiralDiff-Dataset/FiveK-rgb2raw/FiveK_DNG/DNG",
            "ext": ".dng",
        },
        {
            "name": "FiveK Nikon",
            "camera_model": "NIKON D700",
            "split_name": "NIKON_D700",
            "label_name": "NIKON D700",
            "raw_root": "/data2/dataset/xsb/SpiralDiff-Dataset/FiveK-rgb2raw/FiveK_DNG/DNG",
            "ext": ".dng",
        },
        {
            "name": "NOD Nikon",
            "camera_model": "Nikon750",
            "split_name": "Nikon750",
            "label_name": "Nikon750",
            "raw_root": "/data2/dataset/xsb/SpiralDiff-Dataset/Application-OD/NOD/RAW/Nikon",
            "ext": ".NEF",
        },
        {
            "name": "NOD Sony",
            "camera_model": "Sony_RX100m7",
            "split_name": "Sony_RX100m7",
            "label_name": "Sony_RX100m7",
            "raw_root": "/data2/dataset/xsb/SpiralDiff-Dataset/Application-OD/NOD/RAW/Sony",
            "ext": ".ARW",
        },
    ]

    camera_label_file = DATASET_DIR / "camera_label.txt"
    train_id_files = DATASET_DIR / "splits"
    target_path = DATASET_DIR / "data" / "train"

    num = 0
    patch_size = 512
    norm_mode = False
    num_crop = 3
    dividable_factor = 32
    stride = patch_size

    print("=" * 80)
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("DATASET_DIR:", DATASET_DIR)
    print("camera_label_file:", camera_label_file)
    print("camera_label exists:", camera_label_file.exists())
    print("train_id_files:", train_id_files)
    print("splits exists:", train_id_files.exists())
    print("target_path:", target_path)

    total_raw_count = 0
    for cfg in dataset_configs:
        raw_root = Path(cfg["raw_root"])
        ext = cfg["ext"]

        if raw_root.exists():
            raw_count = len(collect_raw_candidates(raw_root, ext))
        else:
            raw_count = 0

        total_raw_count += raw_count

        print("-" * 80)
        print("Config name:", cfg["name"])
        print("camera_model:", cfg["camera_model"])
        print("split_name:", cfg["split_name"])
        print("label_name:", cfg["label_name"])
        print("raw_root:", raw_root)
        print("ext:", ext)
        print("matched raw count:", raw_count)

    print("=" * 80)

    if not camera_label_file.exists():
        raise FileNotFoundError(f"camera_label.txt not found: {camera_label_file}")

    if not train_id_files.exists():
        raise FileNotFoundError(f"splits directory not found: {train_id_files}")

    if total_raw_count == 0:
        raise FileNotFoundError(
            "No RAW files found.\n"
            "请检查 dataset_configs 中的 raw_root 和 ext 是否正确。"
        )

    label_files = [
        str(train_id_files / f"train_{cfg['split_name']}.txt")
        for cfg in dataset_configs
    ]

    split_name_to_label_name = {
        cfg["split_name"]: cfg["label_name"]
        for cfg in dataset_configs
    }

    print("label_files:")
    for label_file in label_files:
        print("  ", label_file, "exists:", Path(label_file).exists())
        if not Path(label_file).exists():
            raise FileNotFoundError(f"Label split file not found: {label_file}")

    label_dict = read_label_files(
        label_files,
        str(camera_label_file),
        split_name_to_label_name=split_name_to_label_name,
    )

    for cfg in dataset_configs:
        name = cfg["name"]
        camera_model = cfg["camera_model"]
        split_name = cfg["split_name"]
        raw_root = Path(cfg["raw_root"])
        ext = cfg["ext"]

        split_file = train_id_files / f"train_{split_name}.txt"

        print("-" * 80)
        print("Current dataset:", name)
        print("Current camera model:", camera_model)
        print("Current split file:", split_file)
        print("Current raw root:", raw_root)
        print("Current ext:", ext)

        if not raw_root.exists():
            raise FileNotFoundError(f"RAW root not found: {raw_root}")

        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")

        with open(split_file, 'r') as f:
            ids = sorted(
                {
                    get_file_id(line.strip())
                    for line in f.readlines()
                    if line.strip()
                }
            )

        id_set = set(ids)

        raw_candidates = collect_raw_candidates(raw_root, ext)

        raw_paths = []
        for raw_file in raw_candidates:
            file_id = get_file_id(raw_file)
            if file_id in id_set:
                raw_paths.append(raw_file)

        print("Expected id count:", len(ids))
        print("Raw candidate count:", len(raw_candidates))
        print("Matched raw count:", len(raw_paths))

        if len(raw_paths) == 0:
            print(
                f"Warning: no RAW files matched for {name} / {camera_model}. "
                f"Please check whether IDs in {split_file} match filenames under {raw_root}."
            )
            continue

        raw_target_path = target_path / camera_model / 'RAW'
        rgb_target_path = target_path / camera_model / 'RGB'

        # 创建必要的目录
        for path in [rgb_target_path, raw_target_path]:
            os.makedirs(path, exist_ok=True)

        # Prepare arguments for each file
        task_args = [
            (
                raw_file,
                str(raw_target_path),
                str(rgb_target_path),
                patch_size,
                num_crop,
                stride,
                norm_mode,
                label_dict,
                dividable_factor
            )
            for raw_file in raw_paths
        ]

        if DEBUG:
            print("Running in debug mode (single process)")
            results = []
            for args in task_args:
                result = process_file(args)
                results.append(result)
        else:
            print("Running in production mode (multiprocessing)")
            with ProcessPoolExecutor() as executor:
                results = list(executor.map(process_file, task_args))

        # Optional: log success/failure
        for file_id, success in results:
            if not success:
                print(f"Failed to process {file_id}")