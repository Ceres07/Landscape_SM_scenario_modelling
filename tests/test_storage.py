import sys,unittest
from pathlib import Path
import numpy as np
from rasterio.transform import from_origin
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gully_scenarios.storage import volume_from_vwc,volume_from_bucket,mask_from_centreline

class StorageTests(unittest.TestCase):
    def test_vwc_volume_units_and_depth_scaling(self):
        # Two 25 m2 cells at 20% and 40% contain 15 m3 in a 1 m layer.
        self.assertAlmostEqual(volume_from_vwc([20,40],25,1),15.)
        self.assertAlmostEqual(volume_from_vwc([20,40],25,.3),4.5)

    def test_bucket_mm_to_cubic_metres(self):
        self.assertAlmostEqual(volume_from_bucket([100,200],25),7.5)

    def test_nodata_must_not_silently_change_area(self):
        with self.assertRaises(ValueError):volume_from_vwc([20,np.nan],25,1)
        with self.assertRaises(ValueError):volume_from_vwc([20],25,0)

    def test_union_counts_shared_cells_once(self):
        a=np.array([True,True,False]);b=np.array([False,True,True])
        union=a|b
        self.assertEqual(volume_from_vwc(np.array([20,20,20])[union],25),15.)
        self.assertEqual(volume_from_vwc(np.array([20,20,20])[a],25)+volume_from_vwc(np.array([20,20,20])[b],25),20.)

    def test_footprint_has_tapered_ends_and_finite_width(self):
        transform=from_origin(0,100,5,5)
        coords=[[52.5,72.5],[52.5,67.5],[52.5,62.5],[52.5,57.5],[52.5,52.5]]
        m=mask_from_centreline(coords,(20,20),transform,10)
        self.assertTrue(m[7,10]);self.assertFalse(m[7,14]);self.assertFalse(m[5,10])

if __name__=='__main__':unittest.main()
