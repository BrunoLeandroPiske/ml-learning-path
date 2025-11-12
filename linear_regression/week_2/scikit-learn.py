import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
np.set_printoptions(precision=2)

def load_house_data():
    import numpy as np

    # Generate synthetic training data for houses
    np.random.seed(42)  # For reproducibility

    # Parameters
    n_samples = 100  # Number of data points
    size = np.random.uniform(500, 4000, n_samples)  # Size in sqft
    bedrooms = np.random.randint(1, 6, n_samples)  # Number of bedrooms
    floors = np.random.randint(1, 4, n_samples)  # Number of floors
    age = np.random.uniform(0, 50, n_samples)  # Age of the house in years

    # Generate price based on a simple linear model with some noise
    price = (
            50 * size / 1000  # Price increases with size
            + 20 * bedrooms  # Price increases with bedrooms
            + 10 * floors  # Price increases with floors
            - 2 * age  # Price decreases with age
            + np.random.normal(0, 10, n_samples)  # Add some noise
    )

    # Combine features into X and price into Y
    X_train = np.column_stack((size, bedrooms, floors, age))
    y_train = price

    # Print shapes for verification
    return X_train, y_train

X_train, y_train = load_house_data()
X_features = ['size(sqft)','bedrooms','floors','age']

scaler = StandardScaler()
X_norm = scaler.fit_transform(X_train)
print(f"Peak to Peak range by column in Raw        X:{np.ptp(X_train,axis=0)}")
print(f"Peak to Peak range by column in Normalized X:{np.ptp(X_norm,axis=0)}")

sgdr = SGDRegressor(max_iter=1000)
sgdr.fit(X_norm, y_train)
print(sgdr)
print(f"number of iterations completed: {sgdr.n_iter_}, number of weight updates: {sgdr.t_}")

b_norm = sgdr.intercept_
w_norm = sgdr.coef_
print(f"model parameters:                   w: {w_norm}, b:{b_norm}")
print( "model parameters from previous lab: w: [110.56 -21.27 -32.71 -37.97], b: 363.16")

# make a prediction using sgdr.predict()
y_pred_sgd = sgdr.predict(X_norm)
# make a prediction using w,b.
y_pred = np.dot(X_norm, w_norm) + b_norm
print(f"prediction using np.dot() and sgdr.predict match: {(y_pred == y_pred_sgd).all()}")

print(f"Prediction on training set:\n{y_pred[:4]}" )
print(f"Target values \n{y_train[:4]}")

# plot predictions and targets vs original features
fig,ax=plt.subplots(1,4,figsize=(12,3),sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:,i],y_train, label = 'target')
    ax[i].set_xlabel(X_features[i])
    ax[i].scatter(X_train[:,i],y_pred, label = 'predict')
ax[0].set_ylabel("Price"); ax[0].legend();
fig.suptitle("target versus prediction using z-score normalized model")
plt.show()