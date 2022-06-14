from turtle import width
from thop import profile
import torch
from model import VDSR

model_name = "VDSR"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检测GPU
width = 3840
height = 2160
input = torch.randn(1,1,width,height).to(device)
model = VDSR().to(device)
flops, params = profile(model, inputs=(input,18))
print("%s|param %.2f|FLOPS %.2f"%(model_name, params / (1000 ** 2), flops / (1000 ** 3)))