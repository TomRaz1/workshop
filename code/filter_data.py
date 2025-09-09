import os
import shutil

src = "/home/yandex/0368352201_BrainWS2025b/tomraz/dFC_DimReduction/data/flattened_output" 
dst = "/home/yandex/0368352201_BrainWS2025b/tomraz/dFC_DimReduction/data/flattened_output_neutral"
os.makedirs(dst, exist_ok=True)

for fname in os.listdir(src):
    if "_n_" in fname: #for neutral: _b_ -> _n_
        shutil.copy(os.path.join(src, fname), os.path.join(dst, fname))
