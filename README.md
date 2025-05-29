# Housing Price Prediction using Linear Regression and Multiple Regression

## Objective
This project aims to implement and understand simple and multiple linear regression models to predict housing prices. The primary tools used are Scikit-learn, Pandas, and Matplotlib. 

The key learning goals include:
* Understanding regression modeling. 
* Applying various evaluation metrics (MAE, MSE, R²). 
* Interpreting model coefficients and results. 

## Dataset
The project uses the `Housing.csv` dataset, which contains information about various features of houses and their corresponding prices. 

Features include:
* `price`: The target variable. 
* `area`: The area of the house in square feet. 
* `bedrooms`: Number of bedrooms.
* `bathrooms`: Number of bathrooms. 
* `stories`: Number of stories. 
* `mainroad`: Whether the house is connected to the main road (yes/no). 
* `guestroom`: Whether the house has a guest room (yes/no). 
* `basement`: Whether the house has a basement (yes/no). 
* `hotwaterheating`: Whether the house has hot water heating (yes/no). 
* `airconditioning`: Whether the house has air conditioning (yes/no). 
* `parking`: Number of parking spots. 
* `prefarea`: Whether the house is in a preferred area (yes/no).
* `furnishingstatus`: Furnishing status (furnished, semi-furnished, unfurnished).

## Process Implemented

The project follows these key steps:

### 1. Setup and Library Imports
Necessary Python libraries for data manipulation, modeling, and visualization were imported:
* **Pandas**: For data loading and manipulation.
* **NumPy**: For numerical operations.
* **Matplotlib & Seaborn**: For plotting and visualizations.
* **Scikit-learn**: For machine learning tasks including:
    * `train_test_split`: To split data into training and testing sets.
    * `LinearRegression`: To build the regression models.
    * `mean_absolute_error`, `mean_squared_error`, `r2_score`: For model evaluation.

### 2. Data Loading and Initial Inspection
* The `Housing.csv` dataset was loaded into a Pandas DataFrame.
* Initial inspections like `head()`, `info()`, and `isnull().sum()` were performed to understand the data structure, data types, and check for missing values. 

### 3. Data Preprocessing
To prepare the data for the regression models:
* **Binary Categorical Conversion**: Columns with 'yes'/'no' values (`mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, `prefarea`) were converted to numerical format (1 for 'yes', 0 for 'no'). 
* **One-Hot Encoding**: The `furnishingstatus` column, which is a multi-class categorical variable, was converted into numerical format using one-hot encoding with `pd.get_dummies()`. `drop_first=True` was used to avoid multicollinearity.

### 4. Simple Linear Regression (Price vs. Area)
A simple linear regression model was built to predict `price` based solely on `area`.
* **Data Splitting**: The data was split into features (`X_simple` containing 'area') and the target (`y_simple` containing 'price'). This was further divided into training (80%) and testing (20%) sets using `train_test_split`.
* **Model Training**: A `LinearRegression` model was instantiated and trained (fitted) using the training data (`X_train_simple`, `y_train_simple`).
* **Prediction**: The trained model was used to make predictions on the test set (`X_test_simple`).
* **Evaluation**: The model's performance was evaluated using:
    * Mean Absolute Error (MAE)
    * Mean Squared Error (MSE)
    * Root Mean Squared Error (RMSE)
    * R-squared (R²)
* **Visualization & Interpretation**:
    * A scatter plot of actual prices vs. area was overlaid with the regression line predicted by the model.
    * The intercept and the coefficient for 'area' were printed and interpreted. The coefficient indicates the predicted change in price for a one-unit increase in area.

### 5. Multiple Linear Regression (Price vs. All Features)
A multiple linear regression model was built to predict `price` using all available preprocessed features. 
* **Data Splitting**: The data was split with `X_multiple` containing all preprocessed features (except 'price') and `y_multiple` containing 'price'.  This was also divided into training (80%) and testing (20%) sets.
* **Model Training**: A `LinearRegression` model was instantiated and trained using the multiple feature training data (`X_train_multiple`, `y_train_multiple`).
* **Prediction**: The trained model made predictions on the test set (`X_test_multiple`).
* **Evaluation**: Similar to simple linear regression, the model was evaluated using MAE, MSE, RMSE, and R².
* **Visualization & Interpretation**:
    * Since plotting a multi-dimensional regression line is not straightforward, a scatter plot of actual prices vs. predicted prices was generated. A diagonal line representing a perfect prediction was also plotted for reference.
    * The intercept and coefficients for all features were printed. Each coefficient was interpreted as the predicted change in price for a one-unit change in that feature, holding all other features constant.

## Key Learnings from the Implementation
* **Regression Modeling**: Successfully implemented both simple and multiple linear regression.
* **Data Preprocessing**: Gained experience in converting categorical features (binary and multi-class) into a numerical format suitable for machine learning models. 
* **Model Evaluation**: Understood and applied key regression evaluation metrics:
    * **MAE**: Provides the average magnitude of errors.
    * **MSE/RMSE**: Penalize larger errors more significantly; RMSE is in the same unit as the target.
    * **R²**: Indicates the proportion of variance in the target variable explained by the model. The multiple linear regression model generally showed a higher R² value, indicating it explained more variability in price compared to the simple linear regression model.
* **Model Interpretation**: Learned to interpret the intercept and coefficients of linear regression models to understand the relationship between features and the target variable. 
