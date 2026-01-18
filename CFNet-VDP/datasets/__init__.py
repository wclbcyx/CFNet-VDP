from .kitti_dataset_online import KITTIDataset
from .sceneflow_dataset import SceneFlowDatset
from .underwater_dataset import UnderwaterDataset

__datasets__ = {
    "sceneflow": SceneFlowDatset,
    "SQUID": UnderwaterDataset,
    "kitti": KITTIDataset
}
