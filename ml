//1 simple 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load Excel file
df = pd.read_excel("your_file.xlsx")  # Replace with your Excel file name

# Select features (1 independent variable) and target
X = df[['Age']]        # Independent variable
y = df['Salary']       # Dependent variable

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("R² Score:", r2_score(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

# Plot regression line
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted Line')
plt.title('Simple Linear Regression')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.legend()
plt.show()

# Plot residuals
residuals = y_test - y_pred
sns.residplot(x=y_pred, y=residuals, lowess=True, color='purple')
plt.title('Residual Plot')
plt.xlabel('Predicted Salary')
plt.ylabel('Residuals')
plt.show()

//2 multiple 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Load Excel file
df = pd.read_excel("your_file.xlsx")  # Replace with your Excel file

# Features and target
X = df[['Age', 'Experience']]   # Replace with your actual independent variables
y = df['Salary']               # Replace with your dependent variable

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("R² Score:", r2_score(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("Intercept:", model.intercept_)
print("Coefficients:", list(zip(X.columns, model.coef_)))

# Plot actual vs predicted
plt.scatter(y_test, y_pred, color='green')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('Actual Salary')
plt.ylabel('Predicted Salary')
plt.title('Actual vs Predicted Salary')
plt.show()

# Residual plot
residuals = y_test - y_pred
sns.residplot(x=y_pred, y=residuals, lowess=True, color='red')
plt.title('Residual Plot')
plt.xlabel('Predicted Salary')
plt.ylabel('Residuals')
plt.show()

//3 polynomial 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load Excel data
df = pd.read_excel("your_file.xlsx")  # Replace with your Excel file

# Feature and target
X = df[['Age']]     # Independent variable
y = df['Salary']    # Dependent variable

# Create polynomial features (e.g., degree=2)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("R² Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

# Plotting polynomial curve
X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
X_plot_poly = poly.transform(X_plot)
y_plot = model.predict(X_plot_poly)

plt.scatter(X, y, color='blue', label='Data')
plt.plot(X_plot, y_plot, color='red', label='Polynomial Fit')
plt.title('Polynomial Regression Curve')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.legend()
plt.show()

# Residual plot
residuals = y_test - y_pred
sns.residplot(x=y_pred, y=residuals, lowess=True, color='purple')
plt.title('Residual Plot')
plt.xlabel('Predicted Salary')
plt.ylabel('Residuals')
plt.show()

//4 lasso ridge
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load Excel file
df = pd.read_excel("your_file.xlsx")  # Replace with your Excel file

# Features and target
X = df[['Age', 'Experience']]  # Independent variables
y = df['Salary']              # Dependent variable

# Feature scaling (important for regularization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Lasso Regression
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)

# Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)

# Evaluation
print("LASSO Regression:")
print("R² Score:", r2_score(y_test, y_pred_lasso))
print("MSE:", mean_squared_error(y_test, y_pred_lasso))
print("Coefficients:", lasso.coef_)

print("\nRIDGE Regression:")
print("R² Score:", r2_score(y_test, y_pred_ridge))
print("MSE:", mean_squared_error(y_test, y_pred_ridge))
print("Coefficients:", ridge.coef_)

# Plot: Actual vs Predicted
plt.scatter(y_test, y_pred_lasso, color='blue', label='Lasso')
plt.scatter(y_test, y_pred_ridge, color='green', label='Ridge')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
plt.xlabel('Actual Salary')
plt.ylabel('Predicted Salary')
plt.title('Lasso vs Ridge Regression')
plt.legend()
plt.show()

//5 6  naive and logistic
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix, roc_curve,
    auc, r2_score
)
from sklearn.preprocessing import LabelEncoder

# Load Excel data
df = pd.read_excel("your_file.xlsx")  # Replace with your filename

# Encode target if it's categorical
le = LabelEncoder()
df['Purchased'] = le.fit_transform(df['Purchased'])  # Yes=1, No=0

# Split features and target
X = df[['Age', 'Salary']]           # Replace with your features
y = df['Purchased']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = GaussianNB()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for ROC

# Metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("R2 Score (for classification output):", r2_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix Plot
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ROC Curve (for binary classification only)
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label='AUC = %0.2f' % roc_auc)
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

//7 artificial neural network 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load Excel data
df = pd.read_excel("your_file.xlsx")  # Replace with your Excel file

# Encode target if needed
le = LabelEncoder()
df['Purchased'] = le.fit_transform(df['Purchased'])  # e.g., Yes=1, No=0

# Features and target
X = df[['Age', 'Salary']]  # Replace with your actual feature columns
y = df['Purchased']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Build ANN model
model = Sequential()
model.add(Dense(8, input_dim=X_train.shape[1], activation='relu'))  # hidden layer
model.add(Dense(4, activation='relu'))                              # hidden layer
model.add(Dense(1, activation='sigmoid'))                           # output layer

# Compile
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train
model.fit(X_train, y_train, epochs=100, verbose=0)

# Predict
y_prob = model.predict(X_test).flatten()
y_pred = (y_prob > 0.5).astype(int)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label='AUC = %0.2f' % roc_auc)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

//8 knn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc, r2_score
)

# Load Excel file
df = pd.read_excel("your_file.xlsx")  # Replace with your actual file

# Encode target if it's categorical
le = LabelEncoder()
df['Purchased'] = le.fit_transform(df['Purchased'])  # Yes=1, No=0

# Features and target
X = df[['Age', 'Salary']]   # Replace with your actual feature columns
y = df['Purchased']

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train KNN model
model = KNeighborsClassifier(n_neighbors=3)  # You can change 'k'
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # For ROC

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label='AUC = %0.2f' % roc_auc)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

//9 decision tree
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc, r2_score
)

# Load Excel data
df = pd.read_excel("your_file.xlsx")  # Replace with your actual Excel file

# Encode categorical target
le = LabelEncoder()
df['Purchased'] = le.fit_transform(df['Purchased'])  # e.g., Yes=1, No=0

# Features and target
X = df[['Age', 'Salary']]  # Replace with your actual feature columns
y = df['Purchased']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGn')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Plot Decision Tree
plt.figure(figsize=(10,6))
plot_tree(model, feature_names=['Age', 'Salary'], class_names=['No', 'Yes'], filled=True)
plt.title("Decision Tree")
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label='AUC = %0.2f' % roc_auc)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

// 10 svm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc, r2_score
)

# Load Excel file
df = pd.read_excel("your_file.xlsx")  # Replace with your Excel filename

# Encode categorical target
le = LabelEncoder()
df['Purchased'] = le.fit_transform(df['Purchased'])  # e.g., Yes=1, No=0

# Features and target
X = df[['Age', 'Salary']]  # Replace with your feature columns
y = df['Purchased']

# Feature scaling (important for SVM)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train SVM model (with probability=True for ROC)
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='BuPu')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label='AUC = %0.2f' % roc_auc)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

//11 k means
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Load Excel file
df = pd.read_excel("your_file.xlsx")  # Replace with your Excel file name

# Select features (no labels needed for unsupervised learning)
X = df[['Age', 'Salary']]  # Replace with appropriate features

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow Method to find optimal K
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot Elbow Curve
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS')
plt.show()

# Fit KMeans with chosen k (e.g., 3)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# Add cluster labels to DataFrame
df['Cluster'] = clusters

# Optional: Silhouette Score
score = silhouette_score(X_scaled, clusters)
print("Silhouette Score:", score)

# Plot Clusters
sns.scatterplot(data=df, x='Age', y='Salary', hue='Cluster', palette='Set1')
plt.title("K-Means Clusters")
plt.show()

//12 hierarchical 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

# Load Excel file
df = pd.read_excel("your_file.xlsx")  # Replace with your Excel file

# Select features (unsupervised)
X = df[['Age', 'Salary']]  # Change columns as per your data

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create linkage matrix using Ward’s method
linked = linkage(X_scaled, method='ward')

# Plot Dendrogram
plt.figure(figsize=(10, 6))
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title("Dendrogram (Hierarchical Clustering)")
plt.xlabel("Samples")
plt.ylabel("Euclidean Distance")
plt.show()

# Form clusters (e.g., 3 clusters)
clusters = fcluster(linked, 3, criterion='maxclust')

# Add cluster labels to original DataFrame
df['Cluster'] = clusters

# Optional: Silhouette Score
score = silhouette_score(X_scaled, clusters)
print("Silhouette Score:", score)

# Plot Clusters
sns.scatterplot(data=df, x='Age', y='Salary', hue='Cluster', palette='Set2')
plt.title("Hierarchical Clusters")
plt.show()

// without sk learn 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_excel("your_file.xlsx")  # Replace with your actual file
X = df[['Age', 'Experience']].values
y = df['Salary'].values
n = len(y)

# Simple Linear Regression
x = df['Age'].values
m = np.sum((x - np.mean(x)) * (y - np.mean(y))) / np.sum((x - np.mean(x))**2)
c = np.mean(y) - m * np.mean(x)
y_pred = m * x + c
r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
print("Simple Linear Regression R2:", r2)

# Polynomial Regression (degree 2)
X_poly = np.column_stack((np.ones(n), x, x ** 2))
theta_poly = np.linalg.inv(X_poly.T @ X_poly) @ X_poly.T @ y
y_poly_pred = X_poly @ theta_poly
print("Polynomial Regression R2:", 1 - np.sum((y - y_poly_pred)**2) / np.sum((y - np.mean(y))**2))

# Multiple Linear Regression
X_mlr = np.column_stack((np.ones(n), X))
theta = np.linalg.inv(X_mlr.T @ X_mlr) @ X_mlr.T @ y
y_pred_mlr = X_mlr @ theta
print("Multiple Linear Regression R2:", 1 - np.sum((y - y_pred_mlr)**2) / np.sum((y - np.mean(y))**2))

# Logistic Regression
def sigmoid(z): return 1 / (1 + np.exp(-z))
y_class = (y > np.mean(y)).astype(int)
X_log = np.column_stack((np.ones(n), x))
theta = np.zeros(X_log.shape[1])
for _ in range(1000):
    z = X_log @ theta
    h = sigmoid(z)
    theta -= 0.01 * X_log.T @ (h - y_class) / n
preds = sigmoid(X_log @ theta) > 0.5
print("Logistic Regression Accuracy:", np.mean(preds == y_class))

# KNN
def knn(X_train, y_train, X_test, k=3):
    preds = []
    for test_pt in X_test:
        dists = np.linalg.norm(X_train - test_pt, axis=1)
        indices = np.argsort(dists)[:k]
        preds.append(np.round(np.mean(y_train[indices])))
    return np.array(preds)
X_train, X_test = X[:int(0.8*n)], X[int(0.8*n):]
y_train, y_test = y_class[:int(0.8*n)], y_class[int(0.8*n):]
print("KNN Accuracy:", np.mean(knn(X_train, y_train, X_test) == y_test))

# K-Means
k = 2
centroids = X[np.random.choice(n, k, replace=False)]
for _ in range(10):
    labels = np.argmin(np.linalg.norm(X[:, None] - centroids[None, :], axis=2), axis=1)
    centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
print("K-Means Centroids:", centroids)

# Decision Tree (stump)
threshold = np.mean(X[:, 0])
preds = (X[:, 0] > threshold).astype(int)
print("Decision Tree Accuracy:", np.mean(preds == y_class))

# ANN (1 hidden layer)
X_ann = np.column_stack((np.ones(n), x))
w = np.random.randn(X_ann.shape[1])
for _ in range(1000):
    z = sigmoid(X_ann @ w)
    w -= 0.01 * X_ann.T @ (z - y_class) / n
print("ANN Accuracy:", np.mean((sigmoid(X_ann @ w) > 0.5) == y_class))

# Naive Bayes (Gaussian)
print("Naive Bayes Classifier:")
classes = np.unique(y_class)
mean_std = {c: (X[y_class == c].mean(axis=0), X[y_class == c].std(axis=0)) for c in classes}
def gaussian(x, mu, sigma): return np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (np.sqrt(2 * np.pi) * sigma)
probs = []
for row in X:
    class_probs = []
    for c in classes:
        mu, sigma = mean_std[c]
        prob = np.prod(gaussian(row, mu, sigma))
        class_probs.append(prob)
    probs.append(np.argmax(class_probs))
print("Naive Bayes Accuracy:", np.mean(np.array(probs) == y_class))

# Ridge Regression
lmbda = 1
theta_ridge = np.linalg.inv(X_mlr.T @ X_mlr + lmbda * np.identity(X_mlr.shape[1])) @ X_mlr.T @ y
y_pred_ridge = X_mlr @ theta_ridge
print("Ridge Regression R2:", 1 - np.sum((y - y_pred_ridge)**2) / np.sum((y - np.mean(y))**2))

# Lasso (Gradient Descent)
theta_lasso = np.zeros(X_mlr.shape[1])
for _ in range(1000):
    y_hat = X_mlr @ theta_lasso
    grad = -2 * X_mlr.T @ (y - y_hat) / n + 0.1 * np.sign(theta_lasso)
    theta_lasso -= 0.01 * grad
y_pred_lasso = X_mlr @ theta_lasso
print("Lasso Regression R2:", 1 - np.sum((y - y_pred_lasso)**2) / np.sum((y - np.mean(y))**2))

# SVM (linear, binary class, gradient descent)
X_svm = np.column_stack((np.ones(n), X))
y_svm = 2*y_class - 1
w = np.zeros(X_svm.shape[1])
lr = 0.01
for _ in range(1000):
    margin = y_svm * (X_svm @ w)
    grad = -np.mean((margin < 1)[:, None] * (y_svm[:, None] * X_svm), axis=0) + 0.01 * w
    w -= lr * grad
print("SVM Accuracy:", np.mean((X_svm @ w > 0) == (y_svm > 0)))

# Hierarchical Clustering (single-link)
from scipy.spatial.distance import pdist, squareform
dists = squareform(pdist(X))
clusters = [[i] for i in range(len(X))]
while len(clusters) > 2:
    min_dist = float('inf')
    pair = None
    for i in range(len(clusters)):
        for j in range(i+1, len(clusters)):
            d = min(dists[p1, p2] for p1 in clusters[i] for p2 in clusters[j])
            if d < min_dist:
                min_dist = d
                pair = (i, j)
    i, j = pair
    clusters[i] += clusters[j]
    del clusters[j]
print("Hierarchical Clusters:", clusters)
