

### Sensing Reasoning Pipeline: MNIST dataset

---

+ noise/

  + main.pth.tar: the main MNIST model (10-way classification).
  + attr.pth.tar.* : the attribute model based on pseudo labelling.

+ third_party/* : Code which is directly used from Cohen's repo.
+ archs: Main model and attribute models' architecture def.

+ /:
  + certify.py: interface model certification to create the pA file, whcih can be used during MLN inference.
  + predict.py: interface model bening accuracy evaluation.
  + simplified.py: Run MLN evaluation based on the pseudo labelled hierarchy rules.
  + train_cohen.py: Standard randomized-smoothing training.
  + train_utils: Some utils funcs for training / evaluation.

