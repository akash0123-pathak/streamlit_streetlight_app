# Smart Streetlight AI – Streamlit Application

Smart Streetlight AI is an end-to-end intelligent system that predicts whether a streetlight should be turned ON or OFF by analysing traffic activity, visibility conditions, ambient brightness, and time-based context. The project simulates a real-world smart city scenario where streetlights are controlled dynamically instead of following fixed schedules.

The application is fully functional, runs locally or on Streamlit Cloud, and is implemented using computer vision, machine learning, and an interactive Streamlit interface. All processing is done locally using open-source libraries, and no API keys, tokens, or external services are required.

---

## What This Project Is About

In most cities, streetlights operate on fixed timers, which leads to unnecessary energy consumption during low-traffic hours and insufficient lighting during poor visibility conditions such as fog or rain. This project addresses that problem by using artificial intelligence to make data-driven streetlight decisions based on actual road conditions.

The system observes road scenes through video input, extracts meaningful features such as vehicle count, pedestrian activity, brightness, and fog level, and then uses trained machine learning models to decide whether the streetlight should be ON or OFF at that moment. The decision is displayed visually on each video frame along with a confidence score.

---

## How the System Works Internally

The application follows a complete AI workflow: feature extraction, data generation, model training, and real-time inference.

Each video frame is first converted to grayscale and analysed to compute brightness and contrast. Fog or visibility is estimated in two ways. If TensorFlow is available, a lightweight convolutional neural network trained on synthetic fog images is used to estimate a fog score. If TensorFlow is not available, a heuristic calculation based on brightness and contrast is used as a fallback. This ensures the system works in all environments.

Vehicle and pedestrian detection is performed using YOLOv8 when it is available. If YOLO cannot be loaded, a background subtraction method is used to estimate moving objects in the frame. This fallback logic makes the system robust and prevents failure due to missing heavy dependencies.

Using these techniques, the system extracts the following features from each frame:
- Vehicle count
- Pedestrian count
- Brightness
- Contrast
- Fog score

These visual features are combined with time-based features such as hour of the day and day of the week, along with a synthetic weather label, to form a complete feature vector for decision-making.

---

## Dataset Generation and Model Training

Since real-world labelled streetlight datasets are difficult to obtain, the application generates a synthetic tabular dataset that closely mimics real road conditions. The dataset simulates peak and non-peak traffic hours, day and night lighting conditions, fog and rain effects, and realistic traffic distributions.

Each data row represents a real-world scenario and includes traffic counts, brightness, fog score, time information, weather conditions, and the correct streetlight status (ON or OFF). This dataset is saved locally and can be downloaded directly from the application.

Multiple machine learning models are trained using Scikit-learn pipelines, including Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, and a Voting Ensemble that combines Random Forest and Gradient Boosting. Each pipeline includes proper preprocessing such as feature scaling and one-hot encoding. Model performance metrics are displayed during training, and trained models are saved locally for reuse.

---

## How to Use the Application (User Perspective)

Once the application is running, the user interacts with it through four main sections.

In the dataset and training section, the user can generate a synthetic dataset with a single click and then train all machine learning models. The dataset and trained models are automatically saved and can be downloaded for further analysis or reporting.

In the inference section, the user can upload a short road traffic video or choose to run the system using automatically generated sample frames. When inference is started, the application processes the video frame by frame. For each frame, features are extracted, the best available trained model is selected, and a prediction is made indicating whether the streetlight should be ON or OFF.

The prediction result is overlaid directly on the video frame along with the confidence score and traffic counts. This makes it easy to understand not just the final decision, but also the factors influencing that decision. If no trained model is available, the system gracefully falls back to a simpler decision logic.

The final section allows users to download trained models and datasets, making the application useful not just for demonstration but also for academic and experimental purposes.

---

## Results and Output

The project produces tangible and verifiable outputs. These include a realistic synthetic dataset, trained machine learning models stored locally, and live streetlight ON/OFF predictions displayed on video frames. The results clearly demonstrate how streetlight behavior changes based on traffic density, visibility conditions, and time of day.

For example, the system predicts streetlights to be ON during high traffic, low brightness, or foggy conditions, and OFF during low traffic and sufficient ambient lighting. Confidence scores provide transparency into the model’s decisions.

---

## Why This Project Matters

This project demonstrates a practical application of artificial intelligence in smart city infrastructure. It shows how computer vision and machine learning can be combined to reduce energy consumption, automate decision-making, and improve road safety. The design is modular, robust, and suitable for further extension into real-world deployments such as IoT-controlled streetlight systems.

---

## How to Run the Project

Create and activate a virtual environment, install the required dependencies, and start the Streamlit application using the provided Python script. Once running, the application opens in a web browser where all interactions are handled through the graphical interface. The project can also be deployed on Streamlit Cloud without any additional configuration.

---

## Conclusion

Smart Streetlight AI is a complete, working prototype that demonstrates how artificial intelligence can be used to solve real-world urban problems. It integrates computer vision, machine learning, and an interactive user interface into a single system that is easy to understand, run, and evaluate. The project is well-suited for academic submissions, technical interviews, and as a foundation for future smart city solutions.
