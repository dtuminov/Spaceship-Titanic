from sklearn.metrics import mean_absolute_error
MAE_metric_value = []
def calculate_MAE(cat_val_prob,lgb_val_prob, xgb_val_prob, y_test):
    for x in range(11):
        for y in range(11):
            z = 10 - x - y
            if 0 <= z <= 10:
                ens_prob = 0.1 * x * cat_val_prob + 0.1 * y * lgb_val_prob + 0.1 * z * xgb_val_prob

                ens_prob1 = (ens_prob[:, 1] >= 0.4).astype(int)  # Assuming a binary classification and threshold of 0.5

                # Считаем Mean Absolute Error (MAE)
                if ((1 - mean_absolute_error(y_test, ens_prob1)) > 0.80):
                    MAE_metric_value.append([str(x), str(y), str(z), 1 - mean_absolute_error(y_test, ens_prob1)])
                    print("x:" + str(x) + " y:" + str(y) + " z:" + str(z) + "   Mean Absolute Error: " + str(
                        1 - mean_absolute_error(y_test, ens_prob1)))
    return MAE_metric_value
