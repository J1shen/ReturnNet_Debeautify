import torch
from PIL import Image
import numpy as np
import pytorch_lightning as pl
from cldm.model import create_model, load_state_dict
from annotator.util import resize_image, HWC3
from pytorch_lightning import seed_everything
import einops,random
from cldm.ddim_hacked import DDIMSampler

save_memory = False

# load checkpoint
checkpoint = "output/final2.ckpt"
model = create_model('./cldm_v21.yaml').cpu()
model.load_state_dict(load_state_dict(checkpoint, location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)
model.eval()

def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta):
    with torch.no_grad():
        img = resize_image(HWC3(input_image), image_resolution)
        H, W, C = img.shape

        control = torch.from_numpy(img.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    return [255 - img] + results,results

if __name__ == '__main__':
    img = Image.open('test_imgs/human.png')
    img = np.asarray(img)
    _,img_result = process(input_image=img, 
                         prompt='a handsome western man', 
                         a_prompt='best quality, extremely detailed',
                         n_prompt='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality', 
                         num_samples=1,
                        image_resolution=512, 
                        ddim_steps=20, 
                        guess_mode=False, 
                        strength=1.0, 
                        scale=9.0, 
                        seed=2023, 
                        eta=0.0)
    
    img_result = Image.fromarray(img_result[0])
    img_result.save('test2.jpg')