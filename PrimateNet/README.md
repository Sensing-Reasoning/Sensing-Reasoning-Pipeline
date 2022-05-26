

### Sensing Reasoning Pipeline: PrimateNet dataset

---

+ ckpt/

  + file*.pkl: Intermediate sampled model output confidence under gaussian, which will be used during the inference.

+ sigma*/

  + submodel*.pt: Interface sensing models' checkpoints

+ /
  + inference.py: Run the evaluation on MLN-based sensing reasoning pipeline.
  + interface_model_training.py: Train the interface sensing models.
  + net.py: The sensing model's architecture.
  + load2.py: Data preprocessing.
  + mln.py: Fundamental MLN architecture definition.

