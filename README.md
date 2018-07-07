# Generative Probabilistic Novelty Detection with Adversarial Autoencoders

**Stanislav Pidhorskyi, Ranya Almohsen, Donald A Adjeroh, Gianfranco Doretto**

Lane Department of Computer Science and Electrical Engineering, West
    Virginia University\
    Morgantown, WV 26508\
    {stpidhorskyi, ralmohse, daadjeroh, gidoretto} @mix.wvu.edu
    
The e-prepint of the article will be available soon on arxiv, [here is the pdf](http://github.com/podgorskiy/GPND/blob/master/document.pdf).

*The code is going to be cleaned up soon.*
*The code for other datasets will be added soon.*

### Content

* **partition_mnist.py** - code for preparing MNIST dataset.
* **train_AAE.py** - code for training the autoencoder.
* **novelty_detector.py** - code for running novelty detector
* **net.py** - contains definitions of network architectures. 

### How to run

You will need to run **partition_mnist.py** first.

Then from **train_AAE.py**, you need to call *main* function:

    train_AAE.main(
      folding_id,
      inliner_classes,
      total_classes,
      folds=5
    )
  
   Args:
   -  folding_id: Id of the fold. For MNIST, 5 folds are generated, so folding_id must be in range [0..5]
   -  inliner_classes: List of classes considered inliers.
   -  total_classes: Total count of classes.
   -  folds: Number of folds.
   
After autoencoder was trained, from **novelty_detector.py**, you need to call *main* function:

    novelty_detector.main(
      folding_id,
      inliner_classes,
      total_classes,
      folds=5
    )
  
   Set of arguments is the same.
