# Pose Estimation

## Data Science Workshop

### Authors: ...



### Training regime (proof of concept)

Testujemy na CPU.

Dla każdego modelu przygotowujemy obraz dokerowy w którym będzie istnieć plik /app/lib.py z funkcją inference(filename). Dokleimy do nich main.py i wyliczymy metryki/score'y.

/app/data - TODO: do decyzji format


Każdy tworzy lib.py dla swojego modelu
/app/lib.py
```py
# import packages, set up module (initialize, load weights, etc.)
def inference(filename) -> list[tuple[float, float]]:
    # returns list of 17 keypoints location (may be changed to other format if needed - to elimate keypoints with low score value)
    ...
```


Jedna osoba tworzy funkcje scorującą, druga funkcje liczącą czas i wyliczająca metryki (95,98,99 kwartyl), ew. zapisujemy wszystko do plików, i następnie możemy to wyliczyć - bezpieczniejsze jeśli będziemy chcieli coś zmienić/policzyć więcej metryk, etc.
/app/main.py
```py
# calculate times/metrics
from .lib import inference

# decorator
def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        end = time.time()
        # TODO: append time taken to file (path based on model name, output mounted to host (writable))
        # print(f"Time taken: {end-start}")
    return wrapper

def scoreit(func):
    def wrapper(*args, **kwargs):
        scores = func(*args, **kwargs)
        # TODO: append scores to file (path based on model name, output mounted to host (writable))
        # print(f"Scores: {scores}")
        return scores
    return wrapper

@timeit
@scoreit
def inference_it(filename):
    return inference(filename)


for filename in filenames:
    # TODO: some warmup runs to avoid cold start
    scores = inference_it(filename)
```
