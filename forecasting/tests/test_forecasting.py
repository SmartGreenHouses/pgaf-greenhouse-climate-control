"""Independent smoke, protocol and Test-gating checks for the released code."""
import json
import os
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from forecasting import data, metrics, models, pipeline


ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT/'config.json').read_text())
SPLITS = json.loads((ROOT/'splits.json').read_text())


class TestProtocol(unittest.TestCase):
    def test_published_model_matrix(self):
        self.assertEqual(CFG['models'], ['lstm_sergiot', 'itransformer', 'patchtst_mv'])
        self.assertEqual(len(CFG['lookbacks']), 6)
        self.assertEqual(len(CFG['horizons']), 9)
        self.assertEqual(len(CFG['versions']), 3)
        self.assertEqual(len(CFG['seeds']), 5)
        self.assertEqual(len(pipeline.jobs(CFG)), 2430)

    def test_nine_sensor_targets_and_six_actuators(self):
        self.assertEqual(len(data.SENSORS), 9)
        self.assertEqual(len(data.ACTUATORS), 6)
        self.assertEqual(data.WIND_INDEX, 7)
        self.assertEqual(len(data.COLS), 16)

    def test_partition_boundaries_do_not_overlap(self):
        from pandas import Timestamp
        for v in CFG['versions']:
            windows=[]
            for d in SPLITS['splits'][v].values():
                windows.append((Timestamp(d['start_utc']),Timestamp(d['end_utc'])))
            for a,(lo,hi) in enumerate(windows):
                self.assertLess(lo,hi)
                for b,(l2,h2) in enumerate(windows):
                    if a!=b:
                        self.assertTrue(hi<l2 or h2<lo)

    def test_complete_train_and_stability_restricted_eval(self):
        n=1000
        ts=pd.date_range('2021-01-01',periods=n,freq='15min',tz='UTC')
        frame=pd.DataFrame({'timestamp':ts})
        for sensor in data.SENSORS:
            frame[sensor]=15.0
        for actuator in data.ACTUATORS:
            frame[actuator]=0.0
        values=np.ones((n,19),dtype=np.float32)
        # One actuator change within a forecast interval. Training admits
        # fully observed windows; Validation/Test filters out affected origins.
        frame.loc[750:,'window_1_position']=50.0
        train=data.valid_starts(frame,values,48,16,None)
        constrained=data.valid_starts(frame,values,48,16,[2,0.05,5,3,1,1])
        self.assertLess(len(constrained),len(train))
        self.assertTrue(np.isin(constrained,train).all())
        origin48=data.matching_origins(frame,values,48,16,[2,0.05,5,3,1,1],672)+48-1
        origin96=data.matching_origins(frame,values,96,16,[2,0.05,5,3,1,1],672)+96-1
        self.assertTrue(np.array_equal(origin48,origin96))

    def test_angular_loss_wraps_wind(self):
        target=torch.zeros((1,1,9));prediction=target.clone()
        target[0,0,7]=179.5;prediction[0,0,7]=-179.5
        loss=metrics.mixed_circular_mse(prediction,target,1.0)
        self.assertAlmostEqual(float(loss),1.0/9.0,places=6)

    def test_no_test_without_explicit_approval(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError,'Test blocked'):
                pipeline.require_authorized(Path(tmp),{})


class TestModels(unittest.TestCase):
    def test_single_standalone_implementation(self):
        for name in CFG['models']:
            torch.manual_seed(13)
            model=models.build_model(name,48,4)
            model.eval()
            with torch.inference_mode():
                result=model(torch.zeros(1,48,19))
            self.assertEqual(tuple(result.shape),(1,4,9))
            self.assertTrue(bool(torch.isfinite(result).all()))
        for old in ['lstm_original','patchtst','itransformer_norm_true']:
            with self.assertRaises(ValueError):
                models.build_model(old,48,4)


class TestRealDataset(unittest.TestCase):
    def test_provenance_and_original_temporal_counts(self):
        target = Path(os.environ.get('ETIFA_DATA_FILE',str(pipeline.DEFAULT_DATA)))
        if not target.is_file():
            self.skipTest('Place the published processed CSV in preprocessing/data/processed')
        for version in CFG['versions']:
            frames=data.load_partitions(target,SPLITS,version,('train','validation','test'))
            self.assertEqual(len(frames['train']),SPLITS['splits'][version]['train']['rows'])
            self.assertEqual(len(frames['validation']),SPLITS['splits'][version]['validation']['rows'])
            self.assertEqual(len(frames['test']),SPLITS['splits'][version]['test']['rows'])
        frames=data.load_partitions(target,SPLITS,'V1',('train','validation'))
        vals,scale=data.scaler_and_values(frames)
        train=data.valid_starts(frames['train'],vals['train'],48,4)
        self.assertEqual(len(train),60877)
        validation=data.matching_origins(frames['validation'],vals['validation'],48,4,
              pipeline.threshold(CFG,4),672)
        self.assertEqual(len(validation),2260)


if __name__=='__main__':
    unittest.main()
