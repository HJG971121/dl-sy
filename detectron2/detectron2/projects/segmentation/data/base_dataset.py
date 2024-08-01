import copy
import json
from typing import Callable, List, Tuple, Optional

from torch.utils.data import Dataset

from detectron2.utils.logger import setup_logger
from detectron2.utils.comm import get_rank, is_main_process
from detectron2.utils.file_io import PathManager

from detectron2.data.catalog import Metadata

logger = setup_logger(name=__name__, distributed_rank=get_rank())

def parse_json_annotation_file(anno_file: str) -> Tuple[Metadata, List[dict]]:
    """
    从json格式的annotation file里读取图片信息
    :param: anno_file: annotation file的文件地址
    return：
    """
    assert anno_file.endswith('.json'), "Annotation file must be a json file, but get {}.".format(anno_file)
    with open(anno_file, 'r') as f:
        annotation = json.load(f)
        meta_data = annotation['metadata']
        assert isinstance(meta_data, dict)
        meta_data = Metadata(**meta_data)
        data_list = annotation['data_list']
        assert isinstance(data_list, list)

    return meta_data, data_list

def build_filter_from_black_list(black_list_file: str):
    logger.info("build filter from black list file: {}".format(black_list_file))
    with PathManager.open(black_list_file, 'r') as f:
        black_list = [x.strip() for x in f.readlines()]
    black_list_set = set(black_list)

    def _filter_func(img_info):
        return img_info['img_name'] not in black_list_set

    return _filter_func

class BaseDataset(Dataset):
    def __init__(self,
                 anno_file: str,
                 filter_func: Optional[Callable] = None,
                 ):
        """
        :param anno_file: annotation文件地址
        :param filter_func: 一个可选的callable对象，作为filter对图像进行过滤，返回True表示保留，False表示从数据集里剔除。
        """
        self._metadata, self._data_list = self._load_annotation(anno_file)
        if filter_func is not None:
            self._data_list = list(filter(filter_func, self._data_list))
            if is_main_process():
                logger.info('{} data samples remain after filtering.'.format(len(self._data_list)))

    @property
    def metdata(self) -> Metadata:
        """
        Get meta information of dataset.

        returns:
            dict: meta information collected from ``BaseDataset.METAINFO``,
            annotation file and metainfo argument during instantiation.
        """
        return self._metadata

    @property
    def data_list(self):
        return self._data_list

    def _load_annotation(self, anno_file: str) -> Tuple[Metadata, List[dict]]:
        return parse_json_annotation_file(anno_file)

    def __getitem__(self, idx):
        return self._data_list[idx]

    def __len__(self):
        return len(self._data_list)