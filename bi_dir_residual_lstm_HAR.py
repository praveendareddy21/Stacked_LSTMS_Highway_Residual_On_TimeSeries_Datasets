from HAR_data_handler import get_HAR_data

X_train, y_train, X_test, y_test = get_HAR_data()
print("X_train shape : " + str(X_train))
print("X_test shape : " + str(X_test))
