import io, os, json, math, zipfile, threading
from dataclasses import dataclass, asdict
from fractions import Fraction
from typing import Any, Optional
import cv2, numpy as np, torch
import torch.nn.functional as F
from PIL import Image, ExifTags, ImageOps
from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException
from fastapi.responses import Response

try:
    import exifread
except Exception:
    exifread = None

MODEL_NAME=os.getenv("MODEL_NAME","metric3d_vit_giant2")
METRIC3D_REPO=os.getenv("METRIC3D_REPO","/opt/Metric3D")
API_KEY=os.getenv("API_KEY","")
ALLOW_UNAUTH_LOCAL=os.getenv("ALLOW_UNAUTH_LOCAL","").strip().lower() in {"1","true","yes","on"}
MAX_SIDE=int(os.getenv("MAX_SIDE","1536"))
DEFAULT_FOV_DEG=float(os.getenv("DEFAULT_FOV_DEG","60"))
FOV_CANDIDATES=os.getenv("FOV_CANDIDATES","35,45,55,60,70,80,90")
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
CKPT_PATH=os.getenv(
    "METRIC3D_CKPT_PATH",
    "/opt/molink-cache/metric3d/metric_depth_vit_giant2_800k.pth",
)
app=FastAPI(title="metric3d-v2-giant-auto-exif-api")
_MODEL=None
_LOCK=threading.Lock()

_MODEL_TYPE_CKPT_MAP={
    "metric3d_convnext_tiny":"/opt/molink-cache/metric3d/convtiny_hourglass_v1.pth",
    "metric3d_convnext_large":"/opt/molink-cache/metric3d/convlarge_hourglass_0.3_150_step750k_v1.1.pth",
    "metric3d_vit_small":"/opt/molink-cache/metric3d/metric_depth_vit_small_800k.pth",
    "metric3d_vit_large":"/opt/molink-cache/metric3d/metric_depth_vit_large_800k.pth",
    "metric3d_vit_giant2":"/opt/molink-cache/metric3d/metric_depth_vit_giant2_800k.pth",
}

@dataclass
class Intr:
    fx:float; fy:float; cx:float; cy:float; source:str; fov_deg:Optional[float]=None; note:str=""

def require_auth(a):
    if not API_KEY:
        if ALLOW_UNAUTH_LOCAL:
            return
        raise HTTPException(500, "API_KEY is not configured; set ALLOW_UNAUTH_LOCAL=1 for local development")
    if a != f"Bearer {API_KEY}":
        raise HTTPException(401, "Unauthorized")

def ratio_to_float(v:Any):
    if v is None: return None
    try:
        if isinstance(v, tuple) and len(v)==2: return float(Fraction(v[0], v[1]))
        if hasattr(v, "num") and hasattr(v, "den"): return float(v.num) / float(v.den)
        if hasattr(v, "numerator") and hasattr(v, "denominator"): return float(v.numerator) / float(v.denominator)
        s=str(v).strip()
        if "/" in s:
            a,b=s.split("/",1); return float(a)/float(b)
        return float(s)
    except Exception:
        return None

def json_safe(v):
    try:
        json.dumps(v)
        return v
    except Exception:
        return str(v)

def fov_intr(w,h,fov,source,note=""):
    fov=max(15.0,min(float(fov),130.0))
    fx=0.5*max(w,h)/math.tan(math.radians(fov)/2.0)
    return Intr(fx,fx,w/2.0,h/2.0,source,fov,note or "fx/fy estimated from FOV.")

def focal_px_intr(w,h,fp,source,note=""):
    fp=float(fp)
    fov=math.degrees(2.0*math.atan(max(w,h)/(2.0*fp)))
    return Intr(fp,fp,w/2.0,h/2.0,source,fov,note or "fx/fy supplied as focal pixels.")

def read_exif_all(raw_bytes:bytes, pil_img:Image.Image):
    out={}
    # Pillow top-level EXIF
    try:
        for k,v in pil_img.getexif().items():
            tag=str(ExifTags.TAGS.get(k,k))
            out[tag]=json_safe(v)
            out["PIL:"+tag]=json_safe(v)
    except Exception as e:
        out["PIL_ERROR"]=str(e)

    # exifread sees many EXIF IFD fields that some PIL versions do not expose.
    if exifread is not None:
        try:
            tags=exifread.process_file(io.BytesIO(raw_bytes), details=False, strict=False)
            for k,v in tags.items():
                out[str(k)] = json_safe(v)
        except Exception as e:
            out["EXIFREAD_ERROR"]=str(e)
    else:
        out["EXIFREAD_ERROR"]="exifread not installed"
    return out

def exif_orientation_value(pil_img:Image.Image):
    try:
        return json_safe(pil_img.getexif().get(274))
    except Exception:
        return None

def open_rgb_exif_aware(raw_bytes:bytes):
    pil0=Image.open(io.BytesIO(raw_bytes))
    exif=read_exif_all(raw_bytes,pil0)
    orientation=exif_orientation_value(pil0)
    pil=ImageOps.exif_transpose(pil0).convert("RGB")
    return pil0,pil,exif,orientation

def exif_was_transposed(orientation):
    return orientation not in [None, 1, "1"]

def get_first_float(d, keys):
    for k in keys:
        if k in d:
            val=ratio_to_float(d.get(k))
            if val is not None: return val, k
    return None, None

def explicit_intr(w,h,camera):
    if all(k in camera and camera[k] not in [None,""] for k in ["fx","fy","cx","cy"]):
        return Intr(float(camera["fx"]),float(camera["fy"]),float(camera["cx"]),float(camera["cy"]),"explicit_intrinsics",None,"Front-end supplied exact intrinsics.")
    if camera.get("focal_px") not in [None,""]:
        return focal_px_intr(w,h,camera["focal_px"],"frontend_focal_px")
    if camera.get("fov_deg") not in [None,""]:
        return fov_intr(w,h,camera["fov_deg"],"frontend_fov")
    return None

def exif_intr(w,h,exif,camera):
    # Prefer 35mm-equivalent. It directly gives FOV without needing sensor size.
    f35, f35_key = get_first_float(exif, [
        "FocalLengthIn35mmFilm",
        "PIL:FocalLengthIn35mmFilm",
        "EXIF FocalLengthIn35mmFilm",
        "EXIF FocalLengthIn35mmFormat",
        "FocalLengthIn35mmFormat",
    ])
    if f35 and f35 > 0:
        # 35mm full-frame horizontal width = 36mm
        fov=math.degrees(2.0*math.atan(36.0/(2.0*f35)))
        return fov_intr(w,h,fov,"exif_35mm_equivalent",f"Estimated from {f35_key}={f35}mm.")

    focal, focal_key = get_first_float(exif, [
        "FocalLength",
        "PIL:FocalLength",
        "EXIF FocalLength",
    ])
    sensor = camera.get("sensor_width_mm")
    if focal and sensor:
        fx=focal/float(sensor)*w
        fov=math.degrees(2.0*math.atan(w/(2.0*fx)))
        return Intr(fx,fx,w/2.0,h/2.0,"exif_focal_length_with_sensor_width",fov,f"Estimated from {focal_key}={focal}mm and sensor_width_mm={sensor}.")

    # If only actual focal length exists, it is not enough to compute fx.
    return None

def choose_intr(w,h,exif,camera):
    e=explicit_intr(w,h,camera)
    if e:
        return {"mode_used":"exact_intrinsics","scale_status":"precise_measurement" if e.source=="explicit_intrinsics" else "user_focal_estimated_measurement","intr":e,"warning":""}
    x=exif_intr(w,h,exif,camera)
    if x:
        return {"mode_used":"exif_camera","scale_status":"exif_estimated_measurement","intr":x,"warning":"EXIF focal data was used. Accuracy depends on preserved metadata and correct 35mm-equivalent or sensor data."}
    d=fov_intr(w,h,DEFAULT_FOV_DEG,"wild_default_fov","No usable intrinsics or EXIF focal data. Default FOV used.")
    return {"mode_used":"wild_default","scale_status":"estimated_scale","intr":d,"warning":"No usable intrinsics or EXIF focal data. Metric scale is estimated, not exact measurement."}

def get_model():
    global _MODEL
    if _MODEL is None:
        with _LOCK:
            if _MODEL is None:
                m=torch.hub.load(METRIC3D_REPO, MODEL_NAME, pretrain=False, source="local")
                ckpt=CKPT_PATH
                if os.path.isfile(ckpt):
                    state=torch.load(ckpt, map_location="cpu")
                    if "model_state_dict" in state:
                        state=state["model_state_dict"]
                    m.load_state_dict(state, strict=False)
                else:
                    m=torch.hub.load(METRIC3D_REPO, MODEL_NAME, pretrain=True, source="local")
                m.to(DEVICE).eval()
                _MODEL=m
    return _MODEL

def preprocess(rgb0):
    input_size=(616,1064)
    h0,w0=rgb0.shape[:2]
    s=min(input_size[0]/h0,input_size[1]/w0)
    rgb=cv2.resize(rgb0,(int(w0*s),int(h0*s)),interpolation=cv2.INTER_LINEAR)
    padv=[123.675,116.28,103.53]
    h,w=rgb.shape[:2]
    ph,pw=input_size[0]-h,input_size[1]-w
    pt,pb,pl,pr=ph//2,ph-ph//2,pw//2,pw-pw//2
    rgb=cv2.copyMakeBorder(rgb,pt,pb,pl,pr,cv2.BORDER_CONSTANT,value=padv)
    mean=torch.tensor([123.675,116.28,103.53]).float()[:,None,None]
    std=torch.tensor([58.395,57.12,57.375]).float()[:,None,None]
    t=torch.from_numpy(rgb.transpose(2,0,1)).float()
    return ((t-mean)/std)[None].to(DEVICE), float(s), [pt,pb,pl,pr]

def crop_depth(t,p):
    pt,pb,pl,pr=p
    he=t.shape[0]-pb if pb else t.shape[0]
    we=t.shape[1]-pr if pr else t.shape[1]
    return t[pt:he,pl:we]

def crop_normal(t,p):
    pt,pb,pl,pr=p
    he=t.shape[2]-pb if pb else t.shape[2]
    we=t.shape[3]-pr if pr else t.shape[3]
    return t[:,:,pt:he,pl:we]

def infer_once(rgb):
    h,w=rgb.shape[:2]
    t,s,p=preprocess(rgb)
    m=get_model()
    with _LOCK:
        with torch.inference_mode():
            pred,conf,out=m.inference({"input":t})
    pred=crop_depth(pred.squeeze(),p)
    pred=F.interpolate(pred[None,None],size=(h,w),mode="bilinear",align_corners=False).squeeze()
    canonical=pred.detach().cpu().numpy().astype(np.float32)
    n=out.get("prediction_normal")
    if n is None:
        normal=np.zeros((h,w,3),np.float32)
    else:
        n=crop_normal(n[:,:3],p)
        n=F.interpolate(n,size=(h,w),mode="bilinear",align_corners=False)
        n=F.normalize(n,dim=1)
        normal=n[0].permute(1,2,0).detach().cpu().numpy().astype(np.float32)
    return {"canonical_depth":canonical,"normal_xyz":normal,"resize_scale":s}

def apply_intr(base,intr:Intr):
    can=base["canonical_depth"]
    h,w=can.shape[:2]
    scale=float(intr.fx)*float(base["resize_scale"])/1000.0
    depth=np.clip(can*scale,0,300).astype(np.float32)
    yy,xx=np.meshgrid(np.arange(h),np.arange(w),indexing="ij")
    x=(xx.astype(np.float32)-float(intr.cx))/float(intr.fx)*depth
    y=(yy.astype(np.float32)-float(intr.cy))/float(intr.fy)*depth
    xyz=np.stack([x,y,depth],axis=-1).astype(np.float32)
    return {"depth_m":depth,"normal_xyz":base["normal_xyz"],"xyz_m":xyz,"intrinsics":asdict(intr),"canonical_to_real_scale":scale,"resize_scale":float(base["resize_scale"])}

def candidate_meta(base,w,h):
    default_scale=fov_intr(w,h,DEFAULT_FOV_DEG,"tmp").fx*base["resize_scale"]/1000.0
    arr=[]
    for x in FOV_CANDIDATES.split(","):
        if not x.strip(): continue
        intr=fov_intr(w,h,float(x.strip()),"wild_candidate_fov")
        sc=intr.fx*base["resize_scale"]/1000.0
        arr.append({"fov_deg":float(x.strip()),"intrinsics":asdict(intr),"canonical_to_real_scale":float(sc),"scale_relative_to_default":float(sc/default_scale)})
    return arr

def npyb(a):
    b=io.BytesIO(); np.save(b,a); return b.getvalue()

def pngb(a):
    b=io.BytesIO(); Image.fromarray(a).save(b,format="PNG"); return b.getvalue()

def depth_vis(d):
    v=d[np.isfinite(d)&(d>0)]
    if len(v)<10: return np.zeros(d.shape,np.uint8)
    lo,hi=np.percentile(v,[2,98]); hi=max(float(hi),float(lo)+1e-6)
    return np.clip((d-lo)/(hi-lo)*255,0,255).astype(np.uint8)

def normal_vis(n):
    return np.clip((n+1.0)*0.5*255.0,0,255).astype(np.uint8)

@app.get("/health")
def health():
    return {"status":"ok","model":MODEL_NAME,"device":DEVICE,"api":"metric3d-v2-giant-auto-exif-api","auto_mode":True,"roi_required":False,"exifread_available":exifread is not None}

@app.post("/inspect_exif")
async def inspect_exif(file:UploadFile=File(...), camera_json:str=Form("{}"), authorization:Optional[str]=Header(None)):
    require_auth(authorization)
    raw=await file.read()
    pil0,pil,exif,orientation=open_rgb_exif_aware(raw)
    raw_size=pil0.size
    corrected_size=pil.size
    if max(pil.size)>MAX_SIDE: pil.thumbnail((MAX_SIDE,MAX_SIDE))
    w,h=pil.size
    try: camera=json.loads(camera_json) if camera_json else {}
    except Exception: camera={}
    mode=choose_intr(w,h,exif,camera)
    return {"raw_size":list(raw_size),"original_size":list(corrected_size),"processed_size":list(pil.size),"exif_orientation":orientation,"exif_transposed":exif_was_transposed(orientation),"exif_keys":sorted([str(k) for k in exif.keys()]),"exif_subset":{k:str(exif.get(k)) for k in sorted(exif.keys()) if "Focal" in k or k in ["Make","Model","Image Make","Image Model","EXIF FNumber","FNumber","EXIF ExposureTime","ExposureTime","EXIF ISOSpeedRatings","ISOSpeedRatings","Orientation","PIL:Orientation"]},"chosen_mode":mode["mode_used"],"scale_status":mode["scale_status"],"selected_intrinsics":asdict(mode["intr"]),"warning":mode["warning"]}

@app.post("/analyze_geometry")
async def analyze_geometry(file:UploadFile=File(...), camera_json:str=Form("{}"), points:str=Form("[]"), include_xyz:bool=Form(False), include_candidates:bool=Form(True), authorization:Optional[str]=Header(None)):
    require_auth(authorization)
    raw=await file.read()
    pil0,pil,exif,orientation=open_rgb_exif_aware(raw)
    raw_size=pil0.size
    orig=pil.size
    if max(pil.size)>MAX_SIDE: pil.thumbnail((MAX_SIDE,MAX_SIDE))
    rgb=np.array(pil)
    h,w=rgb.shape[:2]
    try: camera=json.loads(camera_json) if camera_json else {}
    except Exception: raise HTTPException(400,"camera_json must be valid JSON")
    mode=choose_intr(w,h,exif,camera)
    base=infer_once(rgb)
    sel=apply_intr(base,mode["intr"])
    depth,normal,xyz=sel["depth_m"],sel["normal_xyz"],sel["xyz_m"]
    try: sample=json.loads(points) if points else []
    except Exception: raise HTTPException(400,"points must be JSON, e.g. [[100,200]]")
    sampled=[]
    for p in sample:
        x,y=int(p[0]),int(p[1])
        if 0<=x<w and 0<=y<h:
            sampled.append({"x":x,"y":y,"depth_m":float(depth[y,x]),"xyz_m":[float(v) for v in xyz[y,x]],"normal_xyz":[float(v) for v in normal[y,x]]})
    meta={"image":{"raw_size":list(raw_size),"original_size":list(orig),"processed_size":list(pil.size),"exif_orientation":orientation,"exif_transposed":exif_was_transposed(orientation)},"width":w,"height":h,"model":MODEL_NAME,"auto_mode":True,"mode_used":mode["mode_used"],"scale_status":mode["scale_status"],"scale_warning":mode["warning"],"selected_intrinsics":sel["intrinsics"],"canonical_to_real_scale":sel["canonical_to_real_scale"],"resize_scale":sel["resize_scale"],"exif_detected":bool(exif),"exif_detected_keys":sorted([str(k) for k in exif.keys()]),"exif_focal_subset":{k:str(exif.get(k)) for k in sorted(exif.keys()) if "Focal" in k},"exif_used":mode["mode_used"]=="exif_camera","wild_candidates":candidate_meta(base,w,h) if include_candidates and mode["mode_used"]=="wild_default" else [],"sampled_points":sampled,"quality_note":"Actual focal length and 35mm-equivalent focal length are different fields. This API prefers 35mm-equivalent focal length when present; actual focal length needs sensor_width_mm."}
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,"w",zipfile.ZIP_DEFLATED) as z:
        z.writestr("meta.json",json.dumps(meta,ensure_ascii=False,indent=2))
        z.writestr("depth_m.npy",npyb(depth))
        z.writestr("depth_mm_uint16.png",pngb(np.clip(depth*1000,0,65535).astype(np.uint16)))
        z.writestr("depth_vis.png",pngb(depth_vis(depth)))
        z.writestr("normal_xyz.npy",npyb(normal))
        z.writestr("normal_rgb.png",pngb(normal_vis(normal)))
        if include_xyz: z.writestr("points_xyz_m.npy",npyb(xyz))
    return Response(buf.getvalue(),media_type="application/zip",headers={"Content-Disposition":'attachment; filename="metric3d_auto_exif_result.zip"'})
