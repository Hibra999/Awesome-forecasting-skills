# dl-4-tsc Classifier Map

Official support is defined by `main.py:create_classifier()` and `utils/constants.py`. The current code exposes these classifier keys:

| Key | Class | Notes |
| --- | --- | --- |
| `fcn` | `classifiers.fcn.Classifier_FCN` | Fully convolutional network. Constructor receives `input_shape`, `nb_classes`, and `verbose`. |
| `mlp` | `classifiers.mlp.Classifier_MLP` | MLP baseline. Constructor receives `input_shape`, `nb_classes`, and `verbose`. |
| `resnet` | `classifiers.resnet.Classifier_RESNET` | ResNet classifier. Constructor receives `input_shape`, `nb_classes`, and `verbose`. |
| `encoder` | `classifiers.encoder.Classifier_ENCODER` | Encoder-style convolutional classifier. Constructor receives `input_shape`, `nb_classes`, and `verbose`. |
| `cnn` | `classifiers.cnn.Classifier_CNN` | Time-CNN model. Constructor receives `input_shape`, `nb_classes`, and `verbose`. |
| `inception` | `classifiers.inception.Classifier_INCEPTION` | InceptionTime-style classifier included in current code and constants. |
| `mcdcnn` | `classifiers.mcdcnn.Classifier_MCDCNN` | Multi-channel deep CNN. Constructor receives `input_shape`, `nb_classes`, and `verbose`. |
| `mcnn` | `classifiers.mcnn.Classifier_MCNN` | Multi-scale CNN. `main.py` constructor call differs from the `input_shape` pattern; check the class before custom data. |
| `tlenet` | `classifiers.tlenet.Classifier_TLENET` | t-LeNet. `main.py` constructor call differs from the `input_shape` pattern. |
| `twiesn` | `classifiers.twiesn.Classifier_TWIESN` | Time warping invariant echo state network. `main.py` constructor call differs from the `input_shape` pattern. |

The README text says the `classifiers` folder contains nine DNN implementations and its published result table lists MLP, FCN, ResNet, Encoder, MCNN, t-LeNet, MCDCNN, Time-CNN, and TWIESN. The current source also includes `inception.py` and lists `inception` in `CLASSIFIERS`; agents should mention this version difference instead of treating the README table as the full current model list.

## APIs And Limitations

- Training entry point is the repository's `main.py`, not an installed package API.
- `main.py` reads data, one-hot encodes labels, creates a classifier, and calls `classifier.fit(x_train, y_train, x_test, y_test, y_true)`.
- The repository does not document sklearn-compatible estimators, `cross_val_score`, `predict_proba`, calibration, or model-selection APIs.
- Probabilistic classification is not documented as a public interface. If probabilities are needed, load the saved Keras model and document the custom inference path.
- Built-in metrics are accuracy, macro precision, macro recall, and duration. Add F1-macro, balanced accuracy, ROC-AUC, or average precision outside the official code only when predictions/probabilities are available and valid.

## Model Choice Guidance

- Use `fcn`, `resnet`, or `inception` as strong deep-learning baselines when GPU training is available.
- Use `mlp` or `cnn` for simpler baselines.
- Use `mcdcnn` only when multivariate/channel structure is part of the experiment contract.
- Treat `mcnn`, `tlenet`, and `twiesn` as reproduction models; inspect their class files before adapting new data because their constructor pattern differs in `main.py`.
