# __init__.py
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.Encoder import *
from models.Ablation import *
from models.eeg_backbone import *

# 为了兼容性，确保所有模型类都可访问
__all__ = [
    'Brain_Visual_Encoder_EEG',
    'Brain_Visual_Encoder_MEG',
    'Brain_Visual_Encoder_EEG_wo_spatial',
    'Brain_Visual_Encoder_EEG_wo_feature_adapter',
    'Brain_Visual_Encoder_EEG_wo_blur',
    'Brain_Visual_Encoder_EEG_wo_blur_wo_feature_adapter',
    'Brain_Visual_Encoder_EEG_wo_spatial_wo_feature_adapter',
    'Brain_Visual_Encoder_EEG_wo_spatial_wo_blur',
    'EEGProjectLayer',
    'Shallownet',
    'Deepnet',
    'EEGnet',
    'TSconv',
]