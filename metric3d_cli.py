#!/usr/bin/env python
import argparse, json, os, sys, zipfile
from pathlib import Path
import requests
URL=os.getenv("METRIC3D_URL","http://127.0.0.1:6006")
API_KEY=os.getenv("API_KEY","")
def hdr(): return {"Authorization":f"Bearer {API_KEY}"} if API_KEY else {}
def health(a):
    r=requests.get(a.url+"/health",headers=hdr(),timeout=10); print(r.status_code); print(json.dumps(r.json(),ensure_ascii=False,indent=2))
def inspect(a):
    data={"camera_json":a.camera_json}
    with open(a.image,"rb") as f:
        r=requests.post(a.url+"/inspect_exif",data=data,files={"file":f},headers=hdr(),timeout=60)
    print(r.status_code); print(json.dumps(r.json(),ensure_ascii=False,indent=2))
def infer(a):
    data={"camera_json":a.camera_json,"points":a.points,"include_xyz":str(a.include_xyz).lower(),"include_candidates":str(a.include_candidates).lower()}
    with open(a.image,"rb") as f:
        r=requests.post(a.url+"/analyze_geometry",data=data,files={"file":f},headers=hdr(),timeout=a.timeout)
    print("status:",r.status_code)
    if r.status_code!=200: print(r.text); sys.exit(1)
    Path(a.output).write_bytes(r.content); print("saved:",a.output)
def meta(a):
    with zipfile.ZipFile(a.zip) as z: print(z.read("meta.json").decode())
p=argparse.ArgumentParser(); p.add_argument("--url",default=URL); s=p.add_subparsers(required=True)
x=s.add_parser("health"); x.set_defaults(func=health)
x=s.add_parser("inspect-exif"); x.add_argument("--image",required=True); x.add_argument("--camera-json",default="{}"); x.set_defaults(func=inspect)
x=s.add_parser("infer"); x.add_argument("--image",required=True); x.add_argument("--output",default="/tmp/molink-jobs/metric3d_auto_exif_result.zip"); x.add_argument("--camera-json",default="{}"); x.add_argument("--points",default="[]"); x.add_argument("--include-xyz",action="store_true"); x.add_argument("--include-candidates",action="store_true"); x.add_argument("--timeout",type=int,default=1800); x.set_defaults(func=infer)
x=s.add_parser("meta"); x.add_argument("--zip",required=True); x.set_defaults(func=meta)
a=p.parse_args(); a.func(a)
