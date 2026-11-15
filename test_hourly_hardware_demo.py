import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch,Mock
import hourly_hardware_demo as h

class HardwareSafetyTests(unittest.TestCase):
    def record(self):
        return dict(hardware_complete=False,approval=None,jobs=[],quote=dict(estimated_credits_per_job=38.5),
            fleets={'5':{'hardware':dict(status='AWAITING_APPROVAL',completed_day=False,history=[[1]*5]*13,hourly=[dict(hour=13)])}})
    def device(self):
        d=Mock();d.id=h.DEVICE;d.profile=SimpleNamespace(simulator=False,pricing=SimpleNamespace(perTask=30.,perShot=.0425,perMinute=0.));return d
    def test_fixed_quote(self):
        q=h.quote(self.device());self.assertEqual(q['max_estimated_credits'],385.)
    def test_no_approval_no_provider_calls(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d,'experiment.json').write_text(json.dumps(self.record()))
            with patch.object(h,'QbraidProvider') as provider:
                with self.assertRaises(ValueError):h.run_hardware(d,0)
                provider.assert_not_called()
    def test_insufficient_ceiling_prevents_submission(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d,'experiment.json').write_text(json.dumps(self.record()));device=self.device()
            with patch.object(h,'QbraidProvider') as provider:
                provider.return_value.get_device.return_value=device
                with self.assertRaisesRegex(RuntimeError,'ceiling'):h.run_hardware(d,38.)
                device.submit.assert_not_called()
    def test_unresolved_intent_cannot_duplicate_submission(self):
        with tempfile.TemporaryDirectory() as d:
            record=self.record();record['jobs']=[dict(label='fleet5_hour13',reserved_credits=38.5,status='SUBMISSION_INTENT')]
            Path(d,'experiment.json').write_text(json.dumps(record));device=self.device()
            with patch.object(h,'QbraidProvider') as provider:
                provider.return_value.get_device.return_value=device
                with self.assertRaisesRegex(RuntimeError,'Unresolved'):h.run_hardware(d,385.)
                device.submit.assert_not_called()
    def test_superseded_proposal_cannot_submit(self):
        with tempfile.TemporaryDirectory() as d:
            record=self.record();record['superseded_by']='selected warm-start run'
            Path(d,'experiment.json').write_text(json.dumps(record))
            with patch.object(h,'QbraidProvider') as provider:
                with self.assertRaisesRegex(RuntimeError,'superseded'):h.run_hardware(d,385.)
                provider.assert_not_called()
    def test_simulator_is_not_accepted_as_hardware(self):
        d=self.device();d.profile.simulator=True
        with self.assertRaisesRegex(RuntimeError,'QPU'):h.quote(d)

if __name__=='__main__':unittest.main()
