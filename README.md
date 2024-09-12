Property Price Prediction in London
This application is part of the evaluation process for the Big Data and Business Analytics Master's program at UNED 2023/2024.

Description
The application predicts the nightly rental price of properties located in London based on a machine learning model trained with Airbnb data. Users can input information such as the number of guests, bathrooms, beds, reviews, and other property attributes to receive a predicted nightly price.

Dataset
The dataset used to perform the analyses and train the model comes from Kaggle: Airbnb London Listings Data.

Usage
Enter the property details in the respective input fields.
Select the neighborhood from the available list.
Click "Predict Price" to get the estimated nightly price.
Full bathrooms are counted as whole numbers, while half-baths are added as 0.5 (e.g., 1 full bathroom and a half-bath should be entered as 1.5).
Requirements
Python 3.x
Streamlit
LightGBM
Scikit-learn
Pandas
Numpy
To install the dependencies, run:

bash
Copiar código
pip install -r requirements.txt
Author
This project was developed as part of the requirements for the Big Data and Business Analytics Master's program by Miguel Ángel Montero

