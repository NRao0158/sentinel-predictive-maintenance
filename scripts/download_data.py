"""Fetch NASA-linked PHM mirror; extract only required files, including nested ZIP."""
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile
from io import BytesIO
import hashlib, json

ROOT=Path(__file__).resolve().parents[1]
URL='https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip'
def main():
    target=ROOT/'data'; target.mkdir(exist_ok=True)
    payload=urlopen(URL,timeout=120).read()
    def unpack(blob):
        with ZipFile(BytesIO(blob)) as archive:
            for name in archive.namelist():
                base=Path(name).name
                if base in ['train_FD001.txt','test_FD001.txt','RUL_FD001.txt','readme.txt']:
                    (target/base).write_bytes(archive.read(name))
                elif base.lower()=='cmapssdata.zip': unpack(archive.read(name))
    unpack(payload)
    for name in ['train_FD001.txt','test_FD001.txt','RUL_FD001.txt']:
        if not (target/name).exists(): raise RuntimeError('Missing dataset file: '+name)
    (target/'download_manifest.json').write_text(json.dumps({'url':URL,'archive_sha256':hashlib.sha256(payload).hexdigest()},indent=2))
    print('Downloaded FD001. See docs/MODEL_CARD.md for provenance and limitations.')
if __name__=='__main__': main()
