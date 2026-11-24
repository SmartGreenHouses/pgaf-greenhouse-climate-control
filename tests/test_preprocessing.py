"""Basic unit and integrity checks for the canonical ETIFA preprocessing release.

Run: python -m unittest discover -s tests -p 'test_preprocessing.py' -v
Data-oriented tests skip when raw/processed CSV files are not present.
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1] / "preprocessing"
SPEC=importlib.util.spec_from_file_location('etifa_preprocess', ROOT/'scripts/etifa_preprocess.py')
mod=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)

class TestImputationRules(unittest.TestCase):
    def test_bounded_linear_short_gap(self):
        s=pd.Series([0.0, float('nan'), float('nan'), 9.0])
        out=mod.interpolate_short_gaps(s)
        self.assertTrue(np.allclose(out.to_numpy(), [0,3,6,9]))
    def test_unbounded_or_long_gap_unfilled(self):
        s=pd.Series([float('nan'), 0.0]+[float('nan')]*5+[6.0, float('nan')])
        out=mod.interpolate_short_gaps(s)
        self.assertTrue(out.iloc[2:7].isna().all())
        self.assertTrue(np.isnan(out.iloc[0]) and np.isnan(out.iloc[-1]))
    def test_circular_shortest_path(self):
        s=pd.Series([350.0, float('nan'), 10.0])
        out=mod.interpolate_angular(s)
        self.assertTrue(np.isclose(out.iloc[1],0.0))
    def test_ambiguous_opposite_wind_not_filled(self):
        s=pd.Series([0.0,float('nan'),180.0])
        self.assertTrue(np.isnan(mod.interpolate_angular(s).iloc[1]))
    def test_humidity_deficit_identity(self):
        v=mod.humidity_deficit_from_air(np.array([20.]),np.array([100.]))
        self.assertTrue(np.isclose(v[0],0.))
    def test_missing_breaks_frozen_run(self):
        s=pd.Series([20.0]*5+[np.nan]+[20.0]*5)
        self.assertFalse(bool(mod.frozen_runs(s,8).any()))

class TestCanonicalData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw=ROOT/'data/raw/ETIFA_raw_data.csv'
        cls.clean=ROOT/'data/processed/ETIFA_clean.csv'
        cls.imputed=ROOT/'data/processed/ETIFA_clean_imputed_1h.csv'
        cls.audit=ROOT/'data/processed/ETIFA_cleaning_audit.csv'
        cls.summary=ROOT/'data/processed/ETIFA_imputation_summary.csv'
    def ready(self):
        if not all(p.exists() for p in (self.raw,self.clean,self.imputed,self.audit,self.summary)):
            self.skipTest('Dataset not supplied; data release requires provider authorization')
    def test_counts_and_missingness(self):
        self.ready()
        raw=pd.read_csv(self.raw);clean=pd.read_csv(self.clean);imputed=pd.read_csv(self.imputed)
        audit=pd.read_csv(self.audit);summary=pd.read_csv(self.summary)
        self.assertEqual((len(raw),len(clean),len(imputed),len(audit)),(114048,100866,100866,269))
        self.assertEqual(int(clean.isna().sum().sum()),376885)
        self.assertEqual(int(imputed.isna().sum().sum()),25558)
        self.assertEqual((int(summary.filled.sum()),int(summary.missing_after.sum())),(351327,25558))
        self.assertEqual((len(raw)-len(clean)),13182)
    def test_no_recorded_values_changed(self):
        self.ready()
        a=pd.read_csv(self.clean);b=pd.read_csv(self.imputed)
        self.assertTrue(a.timestamp.equals(b.timestamp))
        for col in a.columns.drop('timestamp'):
            present=a[col].notna()
            self.assertTrue(np.allclose(a.loc[present,col],b.loc[present,col],atol=1e-10,rtol=0),col)
    def test_cleaning_rule_counts(self):
        self.ready()
        a=pd.read_csv(self.audit)
        expected={'sensor_box_contradiction':162,'isolated_spike':77,
                  'temperature_equals_minus_5':23,'wet_bulb_above_dry_bulb':6,
                  'out_of_physical_range':1}
        self.assertEqual(a.rule.value_counts().to_dict(),expected)

if __name__=='__main__': unittest.main()
