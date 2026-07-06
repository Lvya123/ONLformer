# ONLformer: An Efficient Transformer with Only Nonlocal Information-modeled Self Attention for Image Restoration

## Training

Please download the corresponding training datasets and modify the dataset paths in `train.yml`.

Different datasets may require different training hyper-parameters. Please train the model with the settings reported in the paper.

Before training the model, start the Visdom server in another command window for visualization:

```bash
python -m visdom.server -port=4567
```

Then run:

```bash
python train.py
```

The trained model weights will be saved in the `logs` folder.

---

## Testing

Please download the corresponding testing datasets and modify the dataset paths in `test.yml`.

Then run:

```bash
python test.py
```

The restored results will be saved in the `logs` folder.
