#!/bin/bash
# nuscenes-devkit không build được trên Colab (pin cũ): cài --no-deps + phụ thuộc cần cho NuScenes().
pip install -q ultralytics
pip install -q --no-deps nuscenes-devkit
pip install -q pyquaternion descartes cachetools shapely tqdm
python -c "import ultralytics, torch, nuscenes.nuscenes; print('ultralytics', ultralytics.__version__, 'cuda', torch.cuda.is_available())"
echo SETUP_DONE
