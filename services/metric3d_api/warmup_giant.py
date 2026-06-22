import os, torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
model=torch.hub.load(os.getenv("METRIC3D_REPO","/opt/Metric3D"),os.getenv("MODEL_NAME","metric3d_vit_giant2"),pretrain=True,source="local")
model.cuda().eval()
print("metric3d_vit_giant2 loaded on", next(model.parameters()).device)
