### Worksheet: Building and Using an Image Classification Model

This worksheet will guide you through creating an image classification model using Google Teachable Machine, converting it into a mobile-friendly format (TensorFlow Lite), and deploying it into a sample mobile app using MediaPipe Studio.

---

### Step 1: Create and Train a Model on Teachable Machine

1. **Open Google Teachable Machine:**
   - Go to [Teachable Machine](https://teachablemachine.withgoogle.com/).
   
2. **Create a New Image Classification Project:**
   - Select **Image Project** and create a **Standard Image Model**.

3. **Define and Train Model Classes:**
   - Add at least three classes, each representing a category (e.g., "Phone," "Bottle," "Wallet").
   - For each class, upload images or use the camera to capture images.
   - Click on **Train Model** and wait for the model to finish training.

4. **Export the Model:**
   - Once trained, click **Export Model**.
   - Select **TensorFlow Lite** as the export format.
   - Set **Model Conversion Type** to **Quantized** to optimize it for mobile performance.
   - Download the model files.

---

### Step 2: Prepare the Model for Mobile Use in Colab

Now, we need to add metadata to the model using Google Colab. Metadata allows mobile applications to interpret the model’s output correctly.

1. **Open a New Google Colab Notebook:**
   - Go to [Google Colab](https://colab.research.google.com/) and create a new notebook.

2. **Upload Model Files to Colab:**
   - In Colab, upload the `model.tflite` and `labels.txt` files you downloaded from Teachable Machine.

3. **Install Required Libraries:**
   - Run the following command in a new cell to install the TensorFlow Lite Support library:
     ```python
     !pip install tflite-support-nightly
     ```

4. **Add Metadata to the Model:**
   - Copy and paste the following code into a Colab cell, which will add metadata to the model:

     ```python
     from tflite_support.metadata_writers import image_classifier
     from tflite_support.metadata_writers import writer_utils

     ImageClassifierWriter = image_classifier.MetadataWriter
     _MODEL_PATH = "model.tflite"
     _LABEL_FILE = "labels.txt"
     _SAVE_TO_PATH = "model_metadata.tflite"
     _INPUT_NORM_MEAN = 127.5
     _INPUT_NORM_STD = 127.5

     writer = ImageClassifierWriter.create_for_inference(
         writer_utils.load_file(_MODEL_PATH), [_INPUT_NORM_MEAN], [_INPUT_NORM_STD], [_LABEL_FILE]
     )

     print(writer.get_metadata_json())
     writer_utils.save_file(writer.populate(), _SAVE_TO_PATH)
     ```
     
5. **Run the Code in Colab:**
   - Execute the cells to generate a new file, `model_metadata.tflite`, which includes metadata for easier integration with mobile platforms.

6. **Download `model_metadata.tflite`:**
   - Once the process completes, download `model_metadata.tflite` to your local machine.

---

### Step 3: Deploy the Model in a Mobile Application with MediaPipe Studio

MediaPipe Studio provides code examples that allow us to integrate our TensorFlow Lite model into a sample mobile app.

1. **Open MediaPipe Studio Image Classifier:**
   - Go to [MediaPipe Studio Image Classifier Demo](https://mediapipe-studio.webapps.google.com/studio/demo/image_classifier).

2. **Upload and Test the Model:**
   - Upload the `model_metadata.tflite` file to the MediaPipe Studio demo.

3. **Download a Sample Code Example:**
   - Choose the platform for your sample app (Android or iOS).
   - Download the sample code provided by MediaPipe Studio, which comes with a ready-to-use app skeleton.

4. **Integrate `model_metadata.tflite` into the Mobile App:**
   - Add `model_metadata.tflite` to the project files in your Android or iOS app, following MediaPipe’s instructions on file placement and usage.

5. **Run the App on a Mobile Device:**
   - Open the project in your IDE (e.g., Android Studio for Android apps, Xcode for iOS apps).
   - Compile and run the app on a connected mobile device.

6. **Test the Model's Accuracy:**
   - Point your camera at objects from each class (e.g., a phone, a bottle, a wallet) and observe the model's classification results.
   - Make adjustments as necessary, like gathering more training data if the accuracy is low.
