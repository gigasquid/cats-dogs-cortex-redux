# cats-dogs-redux


This example borrows heavily from the [RESNET-RETRAIN](https://github.com/thinktopic/cortex/tree/master/examples/resnet-retrain) example in the cortex project.


[dataset](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data)


## Try out the walkthrough with the Jupyter Notebook

Have Jupyter Notebook already? Try the walkthough.

This Jupyter Notebook uses the lein-jupyter plugin to be able to execute Clojure code in project setting. The first time that you run it you will need to install the kernal with `lein jupyter install-kernel`. After that you can open the notebook in the project directory with `lein jupyter notebook`


## Traditional Intstructions

unzip to train.zip data directory.  Should have this directory structure:

```
data/train
```

Get the RESNET 50 model
./get-resnet50.sh 

- you should now have a directory `/models` with a resnet50.nippy file in it

Start repl

```
(build-image-data)
```
- This will setup the training pictures into the correct training and test folder structure that cortex expects under the `data` directory.

You should be ready to train now.


```
(train)
```

Or if you have mem issues, you might want to try it from the uberjar

`lein uberjar` and then `java -jar target/resnet-retrain-0.9.23-SNAPSHOT.jar`

After only retaining the RESNET50 for only one epoch we get pretty good results. Try it out with

```
(label-one)
```


If you are interested in continuing the training, you can run train-again or from the uberjar `java -jar target/resnet-retrain.jar batch-size true`


If you want to run the kaggle tests for classification and submission:

You will need to do a bit more setup for this. First, you need to get the Kaggle test images for classification. There are 12500 of these in the test.zip file from the site. Under the data directory, create a new directory called kaggle-test. Now unzip the contents of test.zip inside that folder. The full directory with all the test images should now be:
data/kaggle-test/test
This step takes a long time and you might have to tweak the batch size again depending on your memory. There are 12500 predications to be made. The main logic for this is in function called `(kaggle-results batch-size)`. It will take a long time to run. It will print the results as it goes along to the kaggle-results.csv file. If you want to check progress you can do wc -l kaggle-results.csv

```
(kaggle-result 100)
```
or adjust the batch size for your memory


Copyright © 2017 FIXME

Distributed under the Eclipse Public License either version 1.0 or (at
your option) any later version.
