# Generative Probabilistic Novelty Detection with Adversarial Autoencoders

**Stanislav Pidhorskyi, Ranya Almohsen, Donald A Adjeroh, Gianfranco Doretto**

Lane Department of Computer Science and Electrical Engineering, West
    Virginia University\
    Morgantown, WV 26508\
    {stpidhorskyi, ralmohse, daadjeroh, gidoretto} @mix.wvu.edu
    
[The e-preprint of the article on arxiv](https://arxiv.org/abs/1807.02588).

[NeurIPS Proceedings](https://papers.nips.cc/paper/7915-generative-probabilistic-novelty-detection-with-adversarial-autoencoders).


    @inproceedings{pidhorskyi2018generative,
      title={Generative probabilistic novelty detection with adversarial autoencoders},
      author={Pidhorskyi, Stanislav and Almohsen, Ranya and Doretto, Gianfranco},
      booktitle={Advances in neural information processing systems},
      pages={6822--6833},
      year={2018}
    }


### Content

* **partition_mnist.py** - code for preparing MNIST dataset.
* **train_AAE.py** - code for training the autoencoder.
* **novelty_detector.py** - code for running novelty detector
* **net.py** - contains definitions of network architectures. 

### How to run

You will need to run **partition_mnist.py** first.

Then run **schedule.py**. It will run as many concurent experiments as many GPUs are available. Reusults will be written to **results.csv** file

___
Alternatively, you can call directly functions from **train_AAE.py** and **novelty_detector.py**

Train autoenctoder with **train_AAE.py**, you need to call *train* function:

    train_AAE.train(
      folding_id,
      inliner_classes,
      ic
    )
  
   Args:
   -  folding_id: Id of the fold. For MNIST, 5 folds are generated, so folding_id must be in range [0..5]
   -  inliner_classes: List of classes considered inliers.
   -  ic: inlier class set index (used to save model with unique filename).
   
After autoencoder was trained, from **novelty_detector.py**, you need to call *main* function:

    novelty_detector.main(
      folding_id,
      inliner_classes,
      total_classes,
      mul,
      folds=5
    )
   -  folding_id: Id of the fold. For MNIST, 5 folds are generated, so folding_id must be in range [0..5]
   -  inliner_classes: List of classes considered inliers.
   -  ic: inlier class set index (used to save model with unique filename).
   -  total_classes: Total count of classes (deprecated, moved to config).
   -  mul: multiplier for power correction. Default value 0.2.
   -  folds: Number of folds (deprecated, moved to config).
   
### Generated/Reconstructed images

![MNIST Reconstruction](images/reconstruction_58.png?raw=true "MNIST Reconstruction")

*MNIST Reconstruction. First raw - real image, second - reconstructed.*

<br><br>

![MNIST Reconstruction](images/sample_58.png?raw=true "MNIST Generation")

*MNIST Generation.*

<br><br>
![COIL100 Reconstruction](images/reconstruction_59_one.png?raw=true "COIL100 Reconstruction")

*COIL100 Reconstruction, single category. First raw - real image, second - reconstructed. Only 57 images were used for training.*

<br><br>

![COIL100 Generation](images/sample_59_one.png?raw=true "COIL100 Generation")

*COIL100 Generation. First raw - real image, second - reconstructed. Only 57 images were used for training.*

<br><br>

![COIL100 Reconstruction](images/reconstruction_59_seven.png?raw=true "COIL100 Reconstruction")

*COIL100 Reconstruction, 7 categories. First raw - real image, second - reconstructed. Only about 60 images per category were used for training*

<br><br>

![COIL100 Generation](images/sample_59_seven.png?raw=true "COIL100 Generation")

*COIL100 Generation. First raw - real image, second - reconstructed. Only about 60 images per category were used for training.*

<br><br>

![PDF](images/PDF.png?raw=true "PDF")

*PDF of the latent space for MNIST. Size of the latent space - 32*

