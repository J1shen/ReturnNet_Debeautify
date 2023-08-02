from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from load_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from pytorch_lightning.callbacks import ModelCheckpoint


# Configs
#resume_path = './checkpoints/control_sd21_ini.ckpt'
resume_path = './checkpoints/control_sd15_ini.ckpt'
batch_size = 4
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./cldm_v15.yaml').cpu()
#model = create_model('./cldm_v21.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control
model.find_unused_parameters=True


# Misc
dataset = MyDataset()
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
checkpoint_callback = ModelCheckpoint(
    monitor='train_loss',
    filename='./output/sample-{epoch:02d}-{val_loss:.2f}',
    save_top_k=3,
    mode='min',
    save_last=True
)
trainer = pl.Trainer(num_nodes=1,
                     precision=32, 
                     max_epochs=10000,
                     callbacks=[logger,checkpoint_callback])


# Train!
trainer.fit(model=model, 
            train_dataloaders=dataloader
            )

trainer.save_checkpoint("./output/final.ckpt")