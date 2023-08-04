import torch
import pytorch_lightning as pl
from cldm.model import create_model, load_state_dict

# load checkpoint
checkpoint = ""
model = create_model('./cldm_v21.yaml').cpu()
model.load_state_dict(load_state_dict(checkpoint, location='cpu'))
# choose your trained nn.Module

model.eval()

# embed 4 fake images!
fake_image_batch = torch.rand(4, 28 * 28, device=model.device)
embeddings = model(fake_image_batch)
print("⚡" * 20, "\nPredictions (4 image embeddings):\n", embeddings, "\n", "⚡" * 20)