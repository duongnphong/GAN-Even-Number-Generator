from model import *
import torch
from torch.utils.data import DataLoader
from dataloader import LoadData

BATCH_SIZE = 1024 * 8
L_RATE = 1e-2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

Generator = Generator(in_shape=128, out_shape=15).to(DEVICE)
Discriminator = Discriminator(in_shape=15, out_shape=1).to(DEVICE)

optim_g = torch.optim.Adam(params=Generator.parameters(), lr=L_RATE)
optim_d = torch.optim.Adam(params=Discriminator.parameters(), lr=L_RATE)

train_data = LoadData()
test_data = LoadData()

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
tets_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

EPOCHS = 100

for epoch in range(EPOCHS):
    Generator.train()
    Discriminator.train()
    for idx, (input_gauss, num, label) in enumerate(train_loader):
        input_gauss = input_gauss.to(DEVICE)
        num = num.to(DEVICE)
        label = label.to(DEVICE)

        # Dnet wants real data to have label of 1, generated data to have label of 0
        # Gnet wants generated data to have label of 1
        # 2 nets are adversarial, therefore their loss fn are opposite in sign

        # Feed real to D
        optim_d.zero_grad()
        out_d_real = Discriminator(num)
        loss_d_real = torch.mean(torch.abs(out_d_real - label))
        
        # Feed fake to D
        fake = Generator(input_gauss)
        out_d_fake = Discriminator(fake)
        loss_d_fake = -torch.mean(torch.abs(out_d_fake - label))

        # Update parameters
        loss_d = loss_d_real + loss_d_fake
        loss_d.backward()
        optim_d.step()
        
        # Feed to G 
        optim_g.zero_grad()
        out_d_fake = Discriminator(Generator(input_gauss))
        out_d_real = Discriminator(num)
        loss_g = torch.mean(torch.abs(out_d_real - out_d_fake))
        loss_g.backward()
        optim_g.step()

    if epoch % 1 == 0:
        print(f"Epoch: {epoch + 1} | LossDReal : {loss_d_real} | LossDFake: {loss_d_fake} | LossG : {loss_g}")

    torch.save(Generator.state_dict(), "model.pth")