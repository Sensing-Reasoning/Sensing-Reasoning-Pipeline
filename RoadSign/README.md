

### Sensing Reasoning Pipeline: RoadSign dataset

---

+ codefiles/

  + pA*.pkl: Intermediate sampled model output confidence under gaussian, which will be used during the inference.
  + newrules.pt: The pre-constructed knowledge rules.
  + simplified.py: Run the evaluation on MLN-based sensing reasoning pipeline. 

+ main/

  + label_mapping.txt: The mapping between model (data) indices to their actual represented attributes.

+ raw/
  + doa_train.py / ROA.py: interface sensing model's training / robustness evaluation - legacy code used in KEMLP.
  + model*.py: sensing model's architecture definition.
  + train.py: attribute sensing model's training.
  + test.py: attribute sensing model's evaluation.

