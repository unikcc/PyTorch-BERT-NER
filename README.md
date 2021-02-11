#### PyTorch-BERT-NER 
This is the BERT Named Entity Recognition code implements with PyTorch
+ Download bert-based-cased.
+ Run Steps:
    1. `git clone git@github.com:unikcc/PyTorch-BERT-NER.git`
    2. `cd PyTorch-BERT-NER/scripts`
    3. `python preprocess.py`
    4. `sh run_ner.sh`
+ Result

           precision    recall  f1-score   support

       LOC    0.9659    0.9728    0.9694      1837
      MISC    0.9018    0.9067    0.9043       922
       PER    0.9714    0.9767    0.9740      1842
       ORG    0.9236    0.9292    0.9264      1341
       Weight   0.9481    0.9539    0.9510     5942 
       Micro-score   0.9481    0.9539    0.9510     5942 
       Macro-score   0.9407    0.9463    0.9435     5942 