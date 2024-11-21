# utils/criterion.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

class CustomCriterion(nn.Module):
    def __init__(self, config=None):
        super(CustomCriterion, self).__init__()
        cost_note_type = config.get('cost_note_type', 1)
        cost_instrument = config.get('cost_instrument', 1)
        cost_pitch = config.get('cost_pitch', 1)
        cost_regression = config.get('cost_regression', 1)


        self.num_note_types = config['num_classes']['note_type']
        self.num_instruments = config['num_classes']['instrument']
        self.num_pitches = config['num_classes']['pitch']

        # Include one extra class for "no object"
        self.note_type_loss = nn.CrossEntropyLoss()
        self.instrument_loss = nn.CrossEntropyLoss()
        self.pitch_loss = nn.CrossEntropyLoss()
        self.regression_loss = nn.L1Loss()

        # Loss weights
        self.loss_weights = {
            'note_type': cost_note_type,
            'instrument': cost_instrument,
            'pitch': cost_pitch,
            'regression': cost_regression
        }

    def forward(self, outputs, targets):
        # Perform matching

        return losses, debuginfo
