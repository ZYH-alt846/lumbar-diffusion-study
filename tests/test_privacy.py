"""Regression checks for the release scanner without storing live sensitive data."""
from pathlib import Path
import importlib.util
import re
import unittest
import subprocess
import tempfile
import hashlib

path=Path(__file__).resolve().parents[1]/'scripts/privacy_audit.py'
spec=importlib.util.spec_from_file_location('privacy_audit',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class PrivacyTests(unittest.TestCase):
    def test_rule_source_does_not_match_itself(self):
        text=path.read_text(encoding='utf-8')
        for category,pattern in module.PATTERNS.items():
            self.assertIsNone(re.search(pattern,text),category)

    def test_user_path_patterns(self):
        examples=[chr(67)+':'+chr(92)+'Users'+chr(92)+'sample'+chr(92)+'x',
                  '/'+'home'+'/'+'sample'+'/'+'x',
                  '/'+'root'+'/'+'autodl-tmp'+'/'+'x']
        for text in examples:self.assertIsNotNone(re.search(module.PATTERNS['absolute_user_path'],text))

    def test_case_and_credential_patterns(self):
        case='lum'+'bar_'+'1234'
        assignment='TOKEN'+' = '+chr(34)+'example-only-value'+chr(34)
        self.assertIsNotNone(re.search(module.PATTERNS['real_case_id'],case))
        self.assertIsNotNone(re.search(module.PATTERNS['credential_assignment'],assignment))

    def test_staged_bytes_cannot_hide_behind_clean_working_copy(self):
        original=module.ROOT
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder)
            try:
                module.ROOT=folder
                subprocess.run(['git','init',str(folder)],check=True,capture_output=True)
                f=folder/'example.txt';f.write_bytes(b'unreviewed staged version')
                subprocess.run(['git','-C',str(folder),'add','example.txt'],check=True,capture_output=True)
                f.write_bytes(b'reviewed working version')
                expected={'example.txt':hashlib.sha256(f.read_bytes()).hexdigest()}
                findings=module.inspect_git_index(expected)
                self.assertTrue(any(r['reason']=='staged_bytes_not_reviewed' for r in findings))
            finally:module.ROOT=original

if __name__=='__main__':unittest.main()
