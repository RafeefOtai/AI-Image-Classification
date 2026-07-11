# AI - Task 1: Image Classification

## Step 1: Training the Model
I uploaded images for two classes, Hello Kitty and Real Cat, and trained the model on Teachable Machine.

---

## Step 2: Testing Each Class in the Preview Panel
After training, I tested the model in the preview panel with a new image for each class to check the accuracy.

### Hello Kitty Class Test
![Hello Kitty Class Test](Class1_Test.png)

### Real Cat Class Test
![Real Cat Class Test](Class2_Test.png)

---

## Step 3: Setting up the Environment
I created an Anaconda environment named `Teachable-Machine-tf212` using Python 3.10.20 to ensure compatibility with TensorFlow 2.12.1 and the exported Keras model.

I installed the required libraries inside this environment:
- TensorFlow
- OpenCV
- Pillow
- NumPy

After that, I configured Visual Studio Code to use the same Anaconda environment as the Python interpreter.

---

## Step 4: Running the Script in VS Code
I ran the Python script in VS Code using the configured Anaconda environment. The model correctly classified both test images with high confidence scores.

![Model output](screenshot_of_output.png)
