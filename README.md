Package                      Version
---------------------------- -----------
matplotlib                   3.10.1
numpy                        1.26.4
scipy                        1.15.2
tensorflow-macos             2.16.2
tensorflow-metal             1.2.0
---------------------------- ----------

tree del progetto
.
├── dataset
│   ├── train
│   │   ├── cani
│   │   └── gatti
│   └── val
│       ├── cani
│       └── gatti
├── images
├── logs
│   ├── fit
│   │   ├── train
│   │   └── validation
│   ├── train
│   └── validation
├── predict.py
└── train.py

- in dataset/train/cani e dataset/train/gatti vanno immagini 150x150.
- in dataset/val/cani e dataset/val/gatti vanno immagini 150x150. le immagini non devono essere presenti in train, quindi devono essere diverse.

solo immagini.jpg, senza trasparenza, mai bianco e nero.