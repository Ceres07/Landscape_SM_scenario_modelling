import os,sys,unittest
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gully_scenarios.core import hourly_bucket,select_events,is_gully,incision_weight

class CoreTests(unittest.TestCase):
    def test_uncertain_gully_labels(self):
        self.assertTrue(is_gully('AEG?'));self.assertTrue(is_gully('EOG'))
        self.assertFalse(is_gully('EER?'))

    def test_hourly_drainage_matches_daily_no_input(self):
        initial=np.array([60.,90.]);k=.006535
        got=hourly_bucket(np.zeros(24),np.zeros(24),np.array([180.,180.]),1.2,k,initial)[-1]
        np.testing.assert_allclose(got,initial*(1-k),rtol=1e-13)

    def test_missing_rain_is_not_silently_zero(self):
        with self.assertRaises(ValueError):hourly_bucket([0,np.nan],[0,0],[180],1.2,.006,[80])

    def test_events_are_ranked_and_separated(self):
        d=pd.DataFrame({'date':pd.to_datetime(['2026-01-01','2026-01-02','2026-01-20']),'rain_mm':[30,50,20]})
        got=select_events(d,2,10)
        self.assertEqual(got.date.tolist(),[pd.Timestamp('2026-01-02'),pd.Timestamp('2026-01-20')])

    def test_incision_exact_depth_and_compact_support(self):
        w=incision_weight(np.array([0,7.5,15,20]),15)
        np.testing.assert_allclose(w,[1,.5,0,0],atol=1e-12)
        for d in [1,4]:self.assertAlmostEqual(np.max(d*w),d)

if __name__=='__main__':unittest.main()
