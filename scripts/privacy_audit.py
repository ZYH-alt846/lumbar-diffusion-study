"""Fail closed on files outside the reviewed snapshot; never prints matching text."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    'credential_assignment': r'''(?im)^\s*(?:[A-Z_]*(?:PASSWORD|PASSWD|TOKEN|SECRET|API_KEY|PRIVATE_KEY)[A-Z_]*)\s*[:=]\s*["'][^"'\n]{5,}["']''',
    'github_token': r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,})\b',
    'private_key': r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----',
    'absolute_user_path': r'(?i)[A-Z]:[\\/](?:Users|HuaweiMoveData)[\\/]|/(?:home)/[^/\s]+/|/(?:root)/(?:autodl-tmp)/',
    'cloud_endpoint': r'(?i)connect\.[a-z0-9.]+seetacloud\.com|\b(?:\d{1,3}\.){3}\d{1,3}\b',
    'real_case_id': r'\blumbar_[0-9]{4}\b',
    'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',
    'phone': r'(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)',
}

def inspect_git_index(expected):
    """Check actual staged bytes as well, so a clean working copy cannot mask them."""
    findings=[]
    try:
        top=subprocess.run(['git','-C',str(ROOT),'rev-parse','--show-toplevel'],capture_output=True,text=True,encoding='utf-8')
        if top.returncode or Path(top.stdout.strip()).resolve()!=ROOT.resolve():return findings
        result=subprocess.run(['git','-C',str(ROOT),'ls-files','-z'],capture_output=True)
        if result.returncode:return [{'file':'.git/index','reason':'index_read_failed'}]
        for raw in result.stdout.split(b'\0'):
            if not raw:continue
            name=raw.decode('utf-8')
            if name not in expected and name!='release_manifest.json':
                findings.append({'file':name,'reason':'index_not_allowlisted'});continue
            blob=subprocess.run(['git','-C',str(ROOT),'show',':'+name],capture_output=True)
            wanted=expected.get(name)
            if name=='release_manifest.json':wanted=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
            if blob.returncode or hashlib.sha256(blob.stdout).hexdigest()!=wanted:
                findings.append({'file':name,'reason':'staged_bytes_not_reviewed'})
    except (OSError,UnicodeError):
        findings.append({'file':'.git/index','reason':'index_check_failed'})
    return findings

def audit():
    manifest_path=ROOT/'release_manifest.json'
    if not manifest_path.exists():raise SystemExit('Missing reviewed release manifest')
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    expected={r['path']:r['sha256'] for r in manifest['files']}
    allowed=set(expected)|{'release_manifest.json'}
    findings=[];count=0
    for p in ROOT.rglob('*'):
        rel=p.relative_to(ROOT)
        if not p.is_file() or any(part in {'.git','__pycache__','.pytest_cache'} for part in rel.parts) or p.suffix=='.pyc':continue
        name=rel.as_posix();count+=1
        if p.is_symlink():findings.append({'file':name,'reason':'symlink'});continue
        if name not in allowed:findings.append({'file':name,'reason':'not_allowlisted'});continue
        data=p.read_bytes()
        if len(data)>1_000_000:findings.append({'file':name,'reason':'size_limit'})
        if name in expected and hashlib.sha256(data).hexdigest()!=expected[name]:
            findings.append({'file':name,'reason':'hash_changed_requires_review'})
        if p.suffix.lower() in {'.pt','.pth','.ckpt','.npz','.npy','.nii','.gz','.tar','.zip','.pdf','.docx','.log','.pem'}:
            findings.append({'file':name,'reason':'excluded_file_type'})
        try:text=data.decode('utf-8')
        except UnicodeDecodeError:
            findings.append({'file':name,'reason':'unexpected_binary'});continue
        for category,pattern in PATTERNS.items():
            for match in re.finditer(pattern,text):
                if category=='email' and match.group(0).endswith('@users.noreply.github.com'):continue
                findings.append({'file':name,'reason':category,'line':text.count('\n',0,match.start())+1})
        # Derive machine-local profile names without storing them in this repository.
        user=os.environ.get('USERNAME','')
        if user and len(user)>2 and not user.isascii() and user in text:
            findings.append({'file':name,'reason':'local_profile_name'})
    missing=set(allowed)-{p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file()}
    for name in sorted(missing):findings.append({'file':name,'reason':'missing_allowlisted_file'})
    findings.extend(inspect_git_index(expected))
    result={'pass':not findings,'files_checked':count,'findings':findings,
            'scope':'Current working tree and staged-byte allowlist, with pattern checks; no guarantee for future additions.'}
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result

if __name__=='__main__':
    raise SystemExit(0 if audit()['pass'] else 1)
