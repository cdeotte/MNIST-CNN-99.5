# MNIST-CNN-99.5  
  
I created this C program to compete in Kaggle's MNIST digit classifier [competition](https://www.kaggle.com/c/digit-recognizer). This program loads Kaggle's 42000 training digit images, displays them, builds and trains a convolutional neural network to classify them while displaying validation accuracy progress plots. It applies data augmentation to create new training images and achieves an accuracy of 99.5% I enjoyed coding everything from scratch and learned tons, but I also built a solution using Python, Keras, and TensorFlow that scores 99.75% [here](https://www.kaggle.com/cdeotte/25-million-images-0-99757-mnist).  
  
To compile this program, you must download the file `webgui.c` from [here](https://ccom.ucsd.edu/~cdeotte/webgui/webgui.tar.gz). That library creates the graphic user interface (GUI). Compile everything with the single command `gcc CNN.c webgui.c -lm -lpthread`. Also download the Kaggle MNIST images [here](http://playagricola.com/Kaggle/KaggleMNIST.tar.gz), uncompress them, and place `KaggleTrain42k.csv` in the directory with your executable.  
  
# How to run 
Upon running the compiled program, your terminal window will say `webgui: Listening on port 15000 for web browser...`. Open a web browser and enter the address `localhost:15000`, and you will see the interface. (Note that you can also run this program on a remote server and enter that server's IP address followed by `:15000`) Next hit the buttons: `Load`, `Display`, `Init-Net`, `Data Augment`, `Train-Net`, and you will see the following:  
  
  
![begin](http://playagricola.com/Kaggle/CNNbegin.png)  
  
# Now be patient :-)
Let the CNN train. Unfortunately I didn't implement GPU yet, so each epoch takes a painfully slow 20 minutes. Be patient and let it train for 100 epochs and you will see the net achieve 99.5% accuracy! 
  
![progress](http://playagricola.com/Kaggle/CNNprogress.png)  
  
# Make predictions for Kaggle
To make predictions for Kaggle, click the `+` sign next to the `Load` button and set `trainSet = 0` and select `dataFile = KaggleTest28k.csv`. Then click `Load`. Next click `Predict` which writes `Submit.csv` to your hard drive. Upload that file to Kaggle and celebrate!  
  
![results](http://playagricola.com/Kaggle/MNIST-result-DA4a.png)
