import sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gully_scenarios.capacity import simulate_budget,aggregate_budget
from gully_scenarios.core import hourly_bucket

class CapacityTests(unittest.TestCase):
    def test_upper_clip_is_accounted_after_losses(self):
        b=simulate_budget([100],[10],[80],1,.1,initial=[40])
        # wet=140; ET=10; drainage=14; unclipped=116; excess=36.
        self.assertAlmostEqual(b['storage_mm'][0,0],80)
        self.assertAlmostEqual(b['excess_mm'][0,0],36)
        np.testing.assert_allclose(b['balance_residual_mm'],0,atol=1e-12)

    def test_no_input_drainage_is_preserved_by_hourly_conversion(self):
        daily=simulate_budget([0],[0],[100],1.2,.05,initial=[60])
        hourly=simulate_budget(np.zeros(24),np.zeros(24),[100],1.2,.05,timestep_hours=1,initial=[60])
        np.testing.assert_allclose(hourly['storage_mm'][-1],daily['storage_mm'][-1],rtol=1e-12)
        np.testing.assert_allclose(hourly['drainage_mm'].sum(axis=0),daily['drainage_mm'][0],rtol=1e-12)

    def test_hourly_matches_existing_experiment(self):
        rain=np.array([0,10,25,50,0.]);pet=np.array([.2,.4,.1,.3,.5]);cap=np.array([60.,120.]);initial=cap*.6
        got=simulate_budget(rain,pet,cap,1.1666,.0065,timestep_hours=1,initial=initial)
        expected=hourly_bucket(rain,pet,cap,1.1666,.0065,initial)
        np.testing.assert_allclose(got['storage_mm'],expected,rtol=0,atol=0)

    def test_capacity_changes_dynamics_not_just_offset(self):
        full=simulate_budget([10,20,40,0],[0]*4,[100],1,0)
        half=simulate_budget([10,20,40,0],[0]*4,[50],1,0)
        self.assertGreater(np.ptp(full['storage_mm']-half['storage_mm']),0)
        self.assertGreater(half['excess_mm'].sum(),full['excess_mm'].sum())
        self.assertEqual(full['initial_storage_mm'][0]/100,half['initial_storage_mm'][0]/50)

    def test_lower_clipping_is_exposed_not_hidden(self):
        b=simulate_budget([0],[1000],[10],1,0,initial=[1])
        self.assertEqual(b['storage_mm'][0,0],0)
        self.assertGreater(b['floor_correction_mm'][0,0],0)
        np.testing.assert_allclose(b['balance_residual_mm'],0,atol=1e-12)

    def test_aggregation_units_and_capacity_weighted_fullness(self):
        b=simulate_budget([0],[0],[100,200],1,0,initial=[100,0])
        a=aggregate_budget(b,np.array([True,True]),25)
        self.assertAlmostEqual(a['storage_m3'][0],2.5)
        self.assertAlmostEqual(a['capacity_m3'],7.5)
        self.assertAlmostEqual(a['fullness_pct'][0],100/3)
        self.assertAlmostEqual(a['at_capacity_area_pct'][0],50)

    def test_invalid_forcing_and_initial_state_fail(self):
        for rain,cap,initial in [([np.nan],[100],[50]),([0],[0],[0]),([0],[100],[101])]:
            with self.assertRaises(ValueError):simulate_budget(rain,[0],cap,1,.01,initial=initial)

if __name__=='__main__':unittest.main()
