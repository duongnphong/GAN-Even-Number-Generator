from model import Generator
import torch
from helpers import gauss, num

NetG = Generator(in_shape=128, out_shape=15)

NetG.load_state_dict(torch.load("model.pth"))

NetG.eval()
input_gauss = []
for x in range(10):
    input_gauss.append(torch.Tensor(gauss(mean=0, std=1, n=128)))
input_gauss = torch.stack(input_gauss, dim=0)
with torch.inference_mode():
    output = NetG(input_gauss)
    output = torch.where(output>=0.5, 1, 0)
    output = output.detach().numpy()
    print([num(output[x]) for x in range(10)])

