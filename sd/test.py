import pipeline
from PIL import Image
from model_loader import preload_models_from_standard_weights
import time

if __name__ == "__main__":
    device = 'cpu'
    model_dict = preload_models_from_standard_weights("data/v1-5-pruned-emaonly.ckpt",device)
    start_time = time.time()
    img_array = pipeline.generate(prompt = "A Dog playing in the field",uncond_prompt = "",models=model_dict)
    print("Generation Time: ",time.time()-start_time)
    image = Image.fromarray(img_array)
    image.save('generated_image.png')
