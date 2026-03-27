# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pandas as pd
import json

class DatasetProcessor:
    def __init__(self, dataset_name, data_path):
        self.dataset_name = dataset_name
        self.data_path=data_path

    def load_dataset(self):
        if self.dataset_name == "LongMemEval":
            return self.load_long_mem_eval()
        else:
            raise ValueError(f"Dataset {self.dataset_name} is not supported.")

    def load_long_mem_eval(self):
        # Placeholder for loading LongMemEval dataset
        # Replace with actual loading logic
        with open(self.data_path, 'r') as file:
            data = json.load(file)
        return data
