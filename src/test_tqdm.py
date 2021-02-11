import tqdm
import time

outer_list = list(range(10))
inner_list = list(range(5))
for j in tqdm.tqdm(outer_list):
    for i in tqdm.tqdm(inner_list, leave=False):
        time.sleep(0.1)
