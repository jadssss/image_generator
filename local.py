import asyncio
from concurrent.futures import ProcessPoolExecutor
from diffusers import StableDiffusionPipeline
import os
import torch
import multiprocessing

# ID модели, которую мы будем использовать (меняйте если знаете что это)
model_id = "CompVis/stable-diffusion-v1-4"

# Отключение фильтрации NSFW контента: **kwargs
def dummy_checker(images, **kwargs):
    # Возвращаем изображения и список из False для каждого изображения
    return images, [False] * len(images)

# Функция для загрузки модели (чтобы каждый процесс загружал свою копию)
def load_model(device):
    # Загружаем модель и перемещаем ее на указанное устройство (CPU или GPU)
    pipe = StableDiffusionPipeline.from_pretrained(model_id).to(device)
    pipe.safety_checker = dummy_checker
    return pipe

# Функция для генерации и сохранения изображения
def generate_and_save_image(index, prompt, num_inference_steps, device):
    print(f'[INFO prcss] Процесс {index} стартовал')  # Лог-сообщение для проверки параллелизма (НЕ ТРОГАТЬ)
    pipe = load_model(device)
    # Генерация изображения на основе текстового запроса с уменьшенным количеством шагов
    image = pipe(prompt, num_inference_steps=num_inference_steps).images[0]
    print(f'[INFO genxt] Генерация завершена для изображения {index}')  # Сообщение для логирования: генерация изображения завершена (НЕ ТРОГАТЬ)

    # Сохранение сгенерированного изображения в файл
    output_name = f"output/battle_flag_legion_{index:03}.png"
    os.makedirs(os.path.dirname(output_name), exist_ok=True)
    image.save(output_name)
    print(f'[INFO imgsv] Изображение сохранено как {output_name}')  # Сообщение для логирования: изображение сохранено (НЕ ТРОГАТЬ)

# Асинхронная функция для управления задачами
async def main(device):
    # Количество изображений для генерации (за забег)

    num_images = 5

    # Задание для нейросети, предпочтительно на английском языке
    prompt = (
        'two burly dwarves in a fantasy setting.'
        'detailed.'
        'high definition.'
        'cinematic lighting.'
        'epic scene.'
        'highly detailed.'
        'realistic'
    )

    # Количество шагов генерации, чем больше тем лучше
    num_inference_steps = 20

    print('[INFO cycli] Цикл начался')

    # Создаем пул процессов
    num_workers = multiprocessing.cpu_count()  # Используем все доступные ядра
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        loop = asyncio.get_running_loop()
        tasks = [
            loop.run_in_executor(executor, generate_and_save_image, i, prompt, num_inference_steps, device)
            for i in range(1, num_images + 1)
        ]
        await asyncio.gather(*tasks)

    print('[INFO aimgs] Все изображения сгенерированы и сохранены')

if __name__ == "__main__":

    device = 'cpu' # CPU == использование процессора для генерации / CUDA == использование Видеокарту
    # ВНИМАНИЕ!!!!!!!!!! CUDA работает только на видеокартах поддерживающих CUDA (20 NVIDIA серия и выше  у amd не знаю)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Количество запусков
    runs = 10

    print(f'[INFO] Модель выбрана, используется устройство: {device}')
    for i in range(runs):
        print(f'[INFO runco] Забег {i}')
        asyncio.run(main(device))