import torch
checkpoint = torch.load('boltz1.ckpt', map_location=torch.device('cpu'))
del checkpoint['optimizer_states']
del checkpoint['ema']
torch.save(checkpoint, 'boltz1.ckpt')
