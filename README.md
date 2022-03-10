

### Sensing Reasoning Pipeline: Word50-10 dataset

---

+ data/

  + "word50.mat" is the original word50 dataset.
  + "word10.pt" is the subset of the word50 dataset, which contains 10 selected words class in it.

+ codefiles/

  + Train the main classifier and the character classifiers:

    + main classifier:

      `python train_word10.py --dataset word10_main --arch MLP --outdir ../ckpts/model/main --epochs 25 --batch 64 --lr_step_size 7 --noise_sd 0.25 --gpu 0 --suffix 1`

    + character classifier:

      `python train_word10.py --dataset word10_character --arch MLP --outdir ../ckpts/model/character --epochs 90 --batch 256 --lr_step_size 40 --noise_sd 0.25 --gpu 0 --suffix 1`

  + Generate the corresponding pA for each interface variables, pick the top2 characters here:

    `python pA_generator.py --top 2 --noise_sd 0.25 --gpu 1`

  + Certification with MLN, the full rules for these 10 words are contained in 'word10_rules.pkl':

    `python certify.py --sigma 0.25 --r 0.25 --wH 2`

+ ckpts/

  + 'model' contains all the trained classifiers
  + 'pA' contains the record of the corresponding pA for each interface variables.

