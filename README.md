# Credit Card Customer Segmentation

This project aims to perform customer segmentation analysis on a credit card dataset using various machine learning techniques. The dataset contains information about credit card customers, including their purchase behavior, credit limit, and payment history.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Methods](#methods)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Installation

To run the code in this project, you will need to have the following dependencies installed:

- Python 3
- pandas
- numpy
- scikit-learn
- Flask
- joblib
- pickle
- matplotlib
- seaborn

You can install the required dependencies using `pip`:

```shell
pip install pandas numpy scikit-learn Flask joblib matplotlib seaborn
```

## Usage

1. Clone this repository to your local machine.
2. Navigate to the project directory.
3. Run the Jupyter Notebook file `eaadjei_ics.ipynb` to execute the code.
4. The notebook contains several sections labeled with markdown headings, each addressing a specific question or task.
5. Follow the instructions and code provided in the notebook to perform data preprocessing, feature engineering, clustering, and classification tasks.
6. The notebook also includes code for a Flask web application. You can run the application to visualize the results or interact with the model.

## Data

The dataset used in this project is stored in the file `CC GENERAL.csv`. It contains information about credit card customers, including various numerical and categorical features. The dataset is loaded using the `pd.read_csv()` function from the pandas library.

## Methods

The project utilizes several machine learning techniques and libraries:

- Data preprocessing: The `SimpleImputer` class from scikit-learn is used to handle missing values, while the `StandardScaler` and `LabelEncoder` classes are applied to standardize and encode the features, respectively.
- Clustering: The `KMeans` algorithm from scikit-learn is employed to perform customer segmentation based on purchase behavior.
- Classification: Two classifiers, `RandomForestClassifier` and `LogisticRegression`, are used to predict customer behavior.
- Model evaluation: The `silhouette_score` and `balanced_accuracy_score` functions are employed to evaluate the clustering and classification models, respectively.
- Feature selection: The `SelectKBest`, `f_classif`, and `SelectFromModel` functions are applied to select the most relevant features for the classification task.
- Model tuning: The `GridSearchCV` class is utilized to perform hyperparameter tuning for the classification models.
- Visualization: The `matplotlib` and `seaborn` libraries are used to create various visualizations, such as bar plots and scatter plots.

## Results

The project aims to provide insights and analysis on credit card customer segmentation and behavior. The results are presented in the Jupyter Notebook, including visualizations and evaluation metrics.

## Contributing

Contributions to this project are welcome. You can contribute by opening an issue to report a bug or suggest improvements, or by submitting a pull request to add new features or fix existing issues.
