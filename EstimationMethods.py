import numpy as np
from random import choices
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def bootstrap_confidence_interval(x, alpha=0.05):
    return [np.quantile(x, alpha / 2), np.quantile(x, 1 - (alpha / 2))]


def S_learner(X_text_dict, X_user_description_dict, X_user_covariates_dict, T_dict, Y_dict,
              t1, t2, domain, model, sample_with_replacements, test_portion=0.1, random_state=42):
    # take relevant domain data
    X_text = X_text_dict[domain]
    X_user_description = X_user_description_dict[domain]
    X_user_covariates = X_user_covariates_dict[domain]
    T = T_dict[domain]
    Y = Y_dict[domain]

    # take rows of t1, t2
    idx = ((T == t1) | (T == t2)).nonzero()
    X_text = X_text[idx]
    X_user_description = X_user_description[idx]
    X_user_covariates = X_user_covariates[idx]
    T = T[idx]
    Y = Y[idx]

    # transform t1 -> 0, t2 -> 1 and expand dimension
    T = np.array([0 if t == t1 else 1 for t in T])
    T = np.expand_dims(T, axis=1)

    # sample with replacements
    if sample_with_replacements:
        idx = choices(range(X_text.shape[0]), k=X_text.shape[0])
        X_text = X_text[idx]
        X_user_description = X_user_description[idx]
        X_user_covariates = X_user_covariates[idx]
        T = T[idx]
        Y = Y[idx]

    # transform (concat & scale) covariates
    scaler = StandardScaler()
    X_scaled_user_covariates = scaler.fit_transform(X_user_covariates)
    X = np.hstack([X_text, X_user_description, X_scaled_user_covariates])

    # combine X and T into X_combined
    X_combined = np.hstack([X, T])

    # split to train and test
    X_train, X_test, Y_train, Y_test = train_test_split(X_combined, Y, test_size=test_portion, random_state=random_state)

    # fit and predict model
    model.fit(X_train, Y_train)

    # validate fitted model on test set
    Y_train_pred = model.predict(X_train)
    Y_test_pred = model.predict(X_test)
    train_mse = mean_squared_error(Y_train, Y_train_pred)
    test_mse = mean_squared_error(Y_test, Y_test_pred)

    # calculate and return ATE
    X_0, X_1 = np.hstack([X, np.zeros_like(T)]), np.hstack([X, np.ones_like(T)])
    Y_pred_0, Y_pred_1 = model.predict(X_0), model.predict(X_1)
    ate = np.mean(Y_pred_1 - Y_pred_0)
    return ate, model, train_mse, test_mse


def T_learner(X_text_dict, X_user_description_dict, X_user_covariates_dict, T_dict, Y_dict,
              t1, t2, domain, model_0, model_1, sample_with_replacements, test_portion=0.1, random_state=42):
    # take relevant domain data
    X_text = X_text_dict[domain]
    X_user_description = X_user_description_dict[domain]
    X_user_covariates = X_user_covariates_dict[domain]
    T = T_dict[domain]
    Y = Y_dict[domain]

    # take rows of t1, t2
    idx = ((T == t1) | (T == t2)).nonzero()
    X_text = X_text[idx]
    X_user_description = X_user_description[idx]
    X_user_covariates = X_user_covariates[idx]
    T = T[idx]
    Y = Y[idx]

    # transform t1 -> 0, t2 -> 1
    T = np.array([0 if t == t1 else 1 for t in T])

    # sample with replacements
    if sample_with_replacements:
        idx = choices(range(X_text.shape[0]), k=X_text.shape[0])
        X_text = X_text[idx]
        X_user_description = X_user_description[idx]
        X_user_covariates = X_user_covariates[idx]
        T = T[idx]
        Y = Y[idx]

    # transform (concat & scale) covariates
    scaler = StandardScaler()
    X_scaled_user_covariates = scaler.fit_transform(X_user_covariates)
    X = np.hstack([X_text, X_user_description, X_scaled_user_covariates])

    # Split data according to T
    X0 = X[(T == 0).nonzero()]
    X1 = X[(T == 1).nonzero()]
    Y0 = Y[(T == 0).nonzero()]
    Y1 = Y[(T == 1).nonzero()]

    # split to train and test
    X0_train, X0_test, Y0_train, Y0_test = train_test_split(X0, Y0, test_size=test_portion, random_state=random_state)
    X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1, Y1, test_size=test_portion, random_state=random_state)

    # fit and predict models
    model_0.fit(X0_train, Y0_train)
    model_1.fit(X1_train, Y1_train)

    # validate fitted models on test set
    Y0_train_pred = model_0.predict(X0_train)
    Y0_test_pred = model_0.predict(X0_test)
    train_mse_0 = mean_squared_error(Y0_train, Y0_train_pred)
    test_mse_0 = mean_squared_error(Y0_test, Y0_test_pred)

    Y1_train_pred = model_1.predict(X1_train)
    Y1_test_pred = model_1.predict(X1_test)
    train_mse_1 = mean_squared_error(Y1_train, Y1_train_pred)
    test_mse_1 = mean_squared_error(Y1_test, Y1_test_pred)

    # calculate and return ATE
    Y_pred_0, Y_pred_1 = model_0.predict(X), model_1.predict(X)
    ate = np.mean(Y_pred_1 - Y_pred_0)
    return ate, model_0, model_1, train_mse_0, test_mse_0, train_mse_1, test_mse_1


def X_learner(X_text_dict, X_user_description_dict, X_user_covariates_dict, T_dict, Y_dict,
              t1, t2, f_0, f_1, tau_hat_0, tau_hat_1, g, test_portion=0.1, random_state=42, print_progress=True):
    # ------------------------- PHASE 0: PREPARE DATA -------------------------
    if print_progress: print('PREPARE DATA')
    # Collect all domain datasets
    domains = np.array(list(X_text_dict.keys()))
    X_text = np.vstack([X_text_dict[domain] for domain in domains])
    X_user_description = np.vstack([X_user_description_dict[domain] for domain in domains])
    X_user_covariates = np.vstack([X_user_covariates_dict[domain] for domain in domains])

    X_domain = list()
    for i in range(len(domains)):
        X_domain += [i]*X_text_dict[domains[i]].shape[0]
    X_domain = np.eye(len(domains))[np.array(X_domain)]
    T = np.hstack([T_dict[domain] for domain in domains])
    Y = np.hstack([Y_dict[domain] for domain in domains])

    # take rows of t1, t2
    idx = ((T == t1) | (T == t2)).nonzero()
    X_text = X_text[idx]
    X_user_description = X_user_description[idx]
    X_user_covariates = X_user_covariates[idx]
    X_domain = X_domain[idx]
    T = T[idx]
    Y = Y[idx]

    # transform t1 -> 0, t2 -> 1 and expand dimension
    T = np.array([0 if t == t1 else 1 for t in T])
    # T = np.expand_dims(T, axis=1)

    # transform (concat & scale) covariates
    scaler = StandardScaler()
    X_scaled_user_covariates = scaler.fit_transform(X_user_covariates)
    X = np.hstack([X_text, X_user_description, X_scaled_user_covariates, X_domain])

    # ------------------------- PHASE 1: T-LEARNER -------------------------
    if print_progress: print('T-LEARNER')
    # Split data according to T
    X0 = X[(T == 0).nonzero()]
    X1 = X[(T == 1).nonzero()]
    Y = np.expand_dims(Y, axis=1)
    Y0 = Y[(T == 0).nonzero()]
    Y1 = Y[(T == 1).nonzero()]

    # split to train and test
    X0_train, X0_test, Y0_train, Y0_test = train_test_split(X0, Y0, test_size=test_portion, random_state=random_state)
    X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1, Y1, test_size=test_portion, random_state=random_state)

    # fit and predict models
    f_0.fit(X0_train, Y0_train)
    f_1.fit(X1_train, Y1_train)

    # validate fitted models on test set
    Y0_train_pred = f_0.predict(X0_train)
    Y0_test_pred = f_0.predict(X0_test)
    train_mse_0 = mean_squared_error(Y0_train, Y0_train_pred)
    test_mse_0 = mean_squared_error(Y0_test, Y0_test_pred)

    Y1_train_pred = f_1.predict(X1_train)
    Y1_test_pred = f_1.predict(X1_test)
    train_mse_1 = mean_squared_error(Y1_train, Y1_train_pred)
    test_mse_1 = mean_squared_error(Y1_test, Y1_test_pred)

    # ------------------------- PHASE 2: GENERATE PSEUDO CATE LABELS -------------------------
    if print_progress: print('GENERATE PSEUDO CATE LABELS')
    tau_1 = Y1[:,0] - f_0.predict(X1)
    tau_0 = f_1.predict(X0) - Y0[:,0]

    # ------------------------- PHASE 3: TRAIN CATE MODELS -------------------------
    if print_progress: print('TRAIN CATE MODELS')
    tau_hat_0.fit(X0, tau_0)
    tau_hat_1.fit(X1, tau_1)
    # cate_model = lambda x: g(x) * tau_hat_0(x) + (1 - g(x)) * tau_hat_1(x)

    return tau_hat_0, tau_hat_1, f_0, f_1, train_mse_0, test_mse_0, train_mse_1, test_mse_1