# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import random
import time
from scipy.io import loadmat
import sklearn
from sklearn import svm, datasets
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, GroupKFold, cross_val_score, cross_validate


class PCA:

    def __init__(self, PATH, reg_path, log_train_path, log_test_path, svm_path):
        self.path = PATH
        self.reg_path = reg_path
        self.log_train_path = log_train_path
        self.log_test_path = log_test_path
        self.svm_path = svm_path

    # Q1 METHODS
    def load_images(self):
        images = []
        for filename in os.listdir(self.path):
            img = cv2.imread(os.path.join(self.path, filename))
            if img is not None:
                images.append(img)
        return images

    def preprocess(self, images):
        # = self.load_images(self.path)
        reshaped_imgs = np.zeros((877, 4096, 3))

        # Reshape to (4096, 3): reshaped_imgs is the X array
        for i in range(len(images)):
            reshaped_imgs[i] = images[i].reshape(-1, images[i].shape[-1])

        # Each slice of reshaped images for RGB
        x1 = reshaped_imgs[:, :, 0]
        x2 = reshaped_imgs[:, :, 1]
        x3 = reshaped_imgs[:, :, 2]

        return x1, x2, x3

    def SVD(self, x1, x2, x3):
        # Find means
        mean1 = x1.mean(axis=1)
        mean2 = x2.mean(axis=1)
        mean3 = x3.mean(axis=1)

        # Originate vectors from the center of mass
        for i in range(len(x1)):
            x1[i] = x1[i] - mean1[i]
            x2[i] = x2[i] - mean2[i]
            x3[i] = x3[i] - mean3[i]

        U1, s1, Vt1 = np.linalg.svd(x1)
        U2, s2, Vt2 = np.linalg.svd(x2)
        U3, s3, Vt3 = np.linalg.svd(x3)

        eigens1 = np.square(s1)
        eigens2 = np.square(s2)
        eigens3 = np.square(s3)

        return eigens1, eigens2, eigens3

    def pca_first_hundred(self, x1, x2, x3):
        """To find the first 100 values, apply SVD and take the values of diagonal Sigma."""

        eigens1, eigens2, eigens3 = self.SVD(x1, x2, x3)

        #%% Build bar plots
        x = np.arange(100)

        plt.bar(x, eigens1[:100])
        plt.title("First Hundered of X1")
        plt.xlabel("Principal Components")
        plt.ylabel("PC Values")
        plt.show()
        plt.bar(x, eigens2[:100])
        plt.title("First Hundered of X2")
        plt.ylabel("PC Values")
        plt.xlabel("Principal Components")
        plt.show()
        plt.bar(x, eigens3[:100])
        plt.title("First Hundered of X3")
        plt.xlabel("Principal Components")
        plt.ylabel("PC Values")
        plt.show()

        # Calculate Proportion of Variance (PVE)
        pve1 = []
        pve2 = []
        pve3 = []
        for i in eigens1:
            pve1.append((i/sum(eigens1))*100)
            pve2.append((i/sum(eigens2))*100)
            pve3.append((i/sum(eigens3))*100)

        print("Proportion of Variances' -----------")
        print("PVE of X1 ", pve1[:10])
        print("PVE of X2 ", pve2[:10])
        print("PVE of X3 ", pve3[:10])

    def noisy_van_gogh(self):
        imgs = self.load_images()
        imgs_sum = np.array((64, 64, 3))
        for i in range(len(imgs)):
            imgs_sum = imgs_sum + imgs[i]

        img_mean = imgs_sum / len(imgs)

        img_var = np.array((64, 64, 3))
        for i in range(len(imgs)):
            img_var = img_var + (np.square(imgs[i] - img_mean))

        img_var = img_var / (len(imgs)-1)

        mean = 0.01 * img_mean
        std = 0.01 * np.sqrt(img_var)

        # create Gaussian Noise
        noise = np.random.normal(mean, std, (64, 64, 3))

        noised_imgs = imgs
        for i in range(len(imgs)):
            noised_imgs[i] = imgs[i] + noise

        x1_noised, x2_noised, x3_noised = self.preprocess(noised_imgs)

        eigen1, eigen2, eigen3 = self.SVD(x1_noised, x2_noised, x3_noised)

        x = np.arange(100)

        plt.bar(x, eigen1[:100])
        plt.title("First Hundered of X1 Noisy Dataset")
        plt.xlabel("Principal Components")
        plt.ylabel("PC Values")
        plt.show()
        plt.bar(x, eigen2[:100])
        plt.title("First Hundered of X2 Noisy Dataset")
        plt.xlabel("Principal Components")
        plt.ylabel("PC Values")
        plt.show()
        plt.bar(x, eigen3[:100])
        plt.title("First Hundered of X3 Noisy Dataset")
        plt.xlabel("Principal Components")
        plt.ylabel("PC Values")
        plt.show()

    # Q2 METHODS
    def beta_parameter(self, fold, fold_label):
        beta = np.linalg.inv(np.transpose(fold).dot(fold)).dot(np.transpose(fold)).dot(fold_label)
        return beta

    def kfold_concatenate(self, f1, f2, f3, f4):
        fold_feat = np.concatenate((f1, f2, f3, f4))
        return fold_feat

    def linear_reg_data(self):
        # Read data: Shape 500 x 8
        data = np.loadtxt(self.reg_path, delimiter=',', skiprows=1)

        random.seed(2019)
        random.shuffle(data)

        features = np.array([[float(row[col_i]) for col_i in range(len(row) - 1)] for row in data])
        chance = [row[-1] for row in data]

        fold1_feat = features[:100]
        fold1_label = chance[:100]
        fold2_feat = features[100:200]
        fold2_label = chance[100:200]
        fold3_feat = features[200:300]
        fold3_label = chance[200:300]
        fold4_feat = features[300:400]
        fold4_label = chance[300:400]
        fold5_feat = features[400:]
        fold5_label = chance[400:]

        return fold1_feat, fold1_label, fold2_feat, fold2_label, fold3_feat, fold3_label, fold4_feat, fold4_label, fold5_feat, fold5_label

    def r_sqr(self, error, label):
        r = 1 - (np.sum(np.square(error))) / (np.sum(np.square(label - np.mean(label))))
        return r

    def normalize(self, set):
        """Normalization method for Lasso part of Linear Regression"""
        norm_set = set/np.sqrt(np.sum(np.square(set)))
        return norm_set

    def lasso(self, train, label, lmbd=0.1):
        norm_train = self.normalize(train)
        norm_label = self.normalize(label)
        weights = np.zeros(len(train[0]))
        for i in range(len(train)):
            r = r + ((label[i] - train[i] * weights))**2 + lmbd * np.sum(weights)
            weights = r
        return r, weights

    def linear_regression(self):
        """
        Q2: Linear Regression
        :return:
        """

        print("\n")
        print("---Q2: LINEAR REGRESSION---------------------")

        f1, f1label, f2, f2label, f3, f3label, f4, f4label, f5, f5label = self.linear_reg_data()

        # Create folds
        fold5 = self.kfold_concatenate(f1, f2, f3, f4)
        fold5_labels = self.kfold_concatenate(f1label, f2label, f3label, f4label)

        fold4 = self.kfold_concatenate(f1, f2, f3, f5)
        fold4_labels = self.kfold_concatenate(f1label, f2label, f3label, f5label)

        fold3 = self.kfold_concatenate(f1, f2, f5, f4)
        fold3_labels = self.kfold_concatenate(f1label, f2label, f5label, f4label)

        fold2 = self.kfold_concatenate(f1, f5, f3, f4)
        fold2_labels = self.kfold_concatenate(f1label, f5label, f3label, f4label)

        fold1 = self.kfold_concatenate(f5, f2, f3, f4)
        fold1_labels = self.kfold_concatenate(f5label, f2label, f3label, f4label)

        # Take betas
        beta5 = self.beta_parameter(fold5, fold5_labels)
        beta4 = self.beta_parameter(fold4, fold4_labels)
        beta3 = self.beta_parameter(fold3, fold3_labels)
        beta2 = self.beta_parameter(fold2, fold2_labels)
        beta1 = self.beta_parameter(fold1, fold1_labels)

        # Apply and take predictions
        pred5 = f5.dot(beta5)
        pred4 = f4.dot(beta4)
        pred3 = f3.dot(beta3)
        pred2 = f2.dot(beta2)
        pred1 = f1.dot(beta1)

        # Take test errors
        error5 = np.subtract(f5label, pred5)
        error4 = np.subtract(f4label, pred4)
        error3 = np.subtract(f3label, pred3)
        error2 = np.subtract(f2label, pred2)
        error1 = np.subtract(f1label, pred1)

        print("Beta (Fold1 is test): ", beta1)
        print("Beta (Fold2 is test): ", beta2)
        print("Beta (Fold3 is test): ", beta3)
        print("Beta (Fold4 is test): ", beta4)
        print("Beta (Fold5 is test): ", beta5)

        # R-Squared
        rsq1 = self.r_sqr(error1, f1label)
        rsq2 = self.r_sqr(error2, f2label)
        rsq3 = self.r_sqr(error3, f3label)
        rsq4 = self.r_sqr(error4, f4label)
        rsq5 = self.r_sqr(error5, f5label)

        print("\n R-Squared------")
        print("R^2 (Fold1 is test): ", rsq1)
        print("R^2 (Fold2 is test): ", rsq2)
        print("R^2 (Fold3 is test): ", rsq3)
        print("R^2 (Fold4 is test): ", rsq4)
        print("R^2 (Fold5 is test): ", rsq5)

        # MSE
        mse1 = np.mean((np.square(error1)))
        mse2 = np.mean((np.square(error2)))
        mse3 = np.mean((np.square(error3)))
        mse4 = np.mean((np.square(error4)))
        mse5 = np.mean((np.square(error5)))
        print("\n MSE------")
        print("MSE (Fold1 is test): ", mse1)
        print("MSE (Fold2 is test): ", mse2)
        print("MSE (Fold3 is test): ", mse3)
        print("MSE (Fold4 is test): ", mse4)
        print("MSE (Fold5 is test): ", mse5)

        # MAE
        mae1 = np.mean(abs(error1))
        mae2 = np.mean(abs(error2))
        mae3 = np.mean(abs(error3))
        mae4 = np.mean(abs(error4))
        mae5 = np.mean(abs(error5))
        print("\n MAE------")
        print("MAE (Fold1 is test): ", mae1)
        print("MAE (Fold2 is test): ", mae2)
        print("MAE (Fold3 is test): ", mae3)
        print("MAE (Fold4 is test): ", mae4)
        print("MAE (Fold5 is test): ", mae5)

        # MAPE
        mape1 = (1 / len(f1label)) * np.sum(abs(error1 / f1label))
        mape2 = (1 / len(f2label)) * np.sum(abs(error2 / f2label))
        mape3 = (1 / len(f3label)) * np.sum(abs(error3 / f3label))
        mape4 = (1 / len(f4label)) * np.sum(abs(error4 / f4label))
        mape5 = (1 / len(f5label)) * np.sum(abs(error5 / f5label))
        print("\n MAPE------")
        print("MAPE (Fold1 is test): ", mape1)
        print("MAPE (Fold2 is test): ", mape2)
        print("MAPE (Fold3 is test): ", mape3)
        print("MAPE (Fold4 is test): ", mape4)
        print("MAPE (Fold5 is test): ", mape5)

        fig, axs = plt.subplots(5)
        fig.suptitle('Prediction vs. Ground Truth')
        axs[0].plot(pred1, label="Predicted Values")
        axs[0].plot(f1label, label="True Values")
        axs[1].plot(pred2, label="Predicted Values")
        axs[1].plot(f2label, label="True Values")
        axs[2].plot(pred3, label="Predicted Values")
        axs[2].plot(f3label, label="True Values")
        axs[3].plot(pred4, label="Predicted Values")
        axs[3].plot(f4label, label="True Values")
        axs[4].plot(pred5, label="Predicted Values")
        axs[4].plot(f5label, label="True Values")
        axs[4].legend(loc="right")
        plt.show()

        # LASSO
        #lasso1, w = self.lasso(f1, f1label)
        #print(lasso1, w)

    # Q3 METHODS
    def predict_current(self, w0, weights, feat):
        res = np.exp(w0 + np.sum(np.multiply(weights, feat)))
        pr = res/(1+res)
        if pr > 0.5:
            pred = 1
        else:
            pred = 0
        return pred

    def create_mini_batches(self, train_feat, train_labels, batch_size=32):
        size = batch_size
        number_of_batches = int(len(train_feat) / size)

        mini_batches = []
        for i in range(number_of_batches-1):
            mini_feats = train_feat[i*size:(i+1)*size]
            mini_labels = train_labels[i*size:(i+1)*size]
            mini_batches.append((mini_feats, mini_labels))

        mini_feats = train_feat[number_of_batches*size:]
        mini_labels = train_labels[number_of_batches*size:]
        mini_batches.append((mini_feats, mini_labels))

        return mini_batches

    def gradient_ascent(self, train_feat, train_labels, learning_rate=0.0001, iterations=1000):
        w0 = np.random.normal(0, 0.01, 1)
        weights = np.random.normal(0, 0.01, len(train_feat[0]))
        errors = []
        for i in range(iterations):
            mini_batches = self.create_mini_batches(train_feat, train_labels)
            for batch in mini_batches:
                mini_feat, mini_labels = batch
                each_pred = np.asarray([mini_labels[i] - self.predict_current(w0, weights, mini_feat[i]) for i in range(len(mini_feat))])
                w0 = w0 + learning_rate * np.sum(each_pred)
                gradient = mini_feat * each_pred[:, np.newaxis]
                gradient = gradient.sum(axis=0)
                weights = weights + learning_rate * gradient
                errors.append(weights)
        return w0, weights

    def stochastic(self, train_feat, train_labels, learning_rate=0.0001, iterations=1000):
        w0_full = np.random.normal(0, 0.01, 1)
        weights_full = np.random.normal(0, 0.01, len(train_feat[0]))
        for i in range(iterations):
            for j in range(len(train_feat)):
                pred = np.asarray([train_labels[j] - self.predict_current(w0_full, weights_full, train_feat[j])])
                w0_full = w0_full + learning_rate * pred
                gradient = train_feat[j] * pred
                weights_full = weights_full + learning_rate * gradient
        return w0_full, weights_full

    def full_batch(self, train_feat, train_labels, learning_rate=0.0001, iterations=1000):
        """Method to apply Full-Batch Gradient Ascent"""
        w0_full = np.random.normal(0, 0.01, 1)
        weights_full = np.random.normal(0, 0.01, len(train_feat[0]))
        print("\nLearning Rate: ", learning_rate)
        for i in range(iterations):
            pred = np.asarray([train_labels[i] - self.predict_current(w0_full, weights_full, train_feat[i]) for i in range(len(train_feat))])
            w0_full = w0_full + learning_rate * np.sum(pred)
            gradient = train_feat * pred[:, np.newaxis]
            gradient = gradient.sum(axis=0)
            weights_full = weights_full + learning_rate * gradient
            if i % 100 == 0:
                print("\n{}th Iteration: ".format(i))
                print("w0: ", w0_full)
                print("Weights: ", weights_full)
        return w0_full, weights_full

    def logistic_reg(self):
        print("\n")
        print("---Q3: LOGISTIC REGRESSION---------------------")
        # Read data: Shape 500 x 8
        train = np.loadtxt(self.log_train_path, delimiter=',', skiprows=1, dtype=object)
        test = np.loadtxt(self.log_test_path, delimiter=',', skiprows=1, dtype=object)

        train_feat = np.array([[row[col_i] for col_i in range(1, len(row))] for row in train])
        train_labels = np.array([row[0] for row in train])

        test_feat = np.array([[row[col_i] for col_i in range(1, len(row))] for row in test])
        test_labels = np.array([row[0] for row in test])

        # Replace 'Male' with 0, and 'Female' with 1
        # Replace 'Cherbourg' with 0, 'Queenstown' with 1, 'Southampton' with 2
        for row in range(len(train_feat)):
            if train_feat[row][1] == 'male':
                train_feat[row][1] = 0
            else:
                train_feat[row][1] = 1
            if train_feat[row][-1] == 'C':
                train_feat[row][-1] = 0
            elif train_feat[row][-1] == 'Q':
                train_feat[row][-1] = 1
            else:
                train_feat[row][-1] = 2

        for row in range(len(test_feat)):
            if test_feat[row][1] == 'male':
                test_feat[row][1] = 0
            else:
                test_feat[row][1] = 1
            if test_feat[row][-1] == 'C':
                test_feat[row][-1] = 0
            elif test_feat[row][-1] == 'Q':
                test_feat[row][-1] = 1
            else:
                test_feat[row][-1] = 2

        train_feat = train_feat.astype(np.float)
        train_labels = train_labels.astype(np.float)
        test_feat = test_feat.astype(np.float)
        test_labels = test_labels.astype(np.float)

        # Normalize
        mean = np.mean(train_feat, axis=0)
        std = np.std(train_feat, axis=0)

        mean_test = np.mean(test_feat, axis=0)
        std_test = np.std(test_feat, axis=0)

        for row in range(len(train_feat)):
            train_feat[row] = np.subtract(train_feat[row], mean) / std
        for row in range(len(test_feat)):
            test_feat[row] = np.subtract(test_feat[row], mean_test) / std_test

        start_mini = time.time()
        # Apply Mini-Batch Gradient Ascent
        w0, weights = self.gradient_ascent(train_feat, train_labels)
        print("Mini-Batch Duration: ", (time.time() - start_mini), "seconds")
        w0_2, weights_2 = self.gradient_ascent(train_feat, train_labels, learning_rate=0.01)
        w0_3, weights_3 = self.gradient_ascent(train_feat, train_labels, learning_rate=0.001)

        # Apply Stochastic Gradient Ascent
        w0_sto, weights_sto = self.stochastic(train_feat, train_labels)
        w0_sto_2, weights_sto_2 = self.stochastic(train_feat, train_labels, learning_rate=0.01)
        w0_sto_3, weights_sto_3 = self.stochastic(train_feat, train_labels, learning_rate=0.001)

        # Apply Full-Batch Gradient Ascent
        w0_full_batch, weights_full_batch = self.full_batch(train_feat, train_labels)
        w0_full_batch_2, weights_full_batch_2 = self.full_batch(train_feat, train_labels, learning_rate=0.01)
        start_full = time.time()
        w0_full_batch_3, weights_full_batch_3 = self.full_batch(train_feat, train_labels, learning_rate=0.001)
        print("\nFull-Batch Duration: ", (time.time() - start_full), "seconds")

        # Declare all variables to keep track the accuracy and all
        preds, preds_2, preds_3 = np.zeros((len(test_feat))), np.zeros((len(test_feat))), np.zeros((len(test_feat)))
        preds_sto, preds_sto_2, preds_sto_3 = np.zeros((len(test_feat))), np.zeros((len(test_feat))), np.zeros((len(test_feat)))
        preds_full_batch, preds_full_batch_2, preds_full_batch_3 = np.zeros((len(test_feat))), np.zeros((len(test_feat))), np.zeros((len(test_feat)))
        falses, falses_2, falses_3 = 0, 0, 0
        falses_sto, falses_sto_2, falses_sto_3 = 0, 0, 0
        falses_full_batch, falses_full_batch_2, falses_full_batch_3 = 0, 0, 0
        corrects, corrects_2, corrects_3 = 0, 0, 0
        corrects_sto, corrects_sto_2, corrects_sto_3 = 0, 0, 0
        corrects_full_batch, corrects_full_batch_2, corrects_full_batch_3 = 0, 0, 0

        # FN = Survived, dead predicted
        # FP = Dead, survived predicted
        # TN = Dead
        # TP = Survived

        tp, tn, fp, fn = 0, 0, 0, 0
        tp_2, tn_2, fp_2, fn_2 = 0, 0, 0, 0
        tp_3, tn_3, fp_3, fn_3 = 0, 0, 0, 0

        tp_sto, tn_sto, fp_sto, fn_sto = 0, 0, 0, 0
        tp_sto_2, tn_sto_2, fp_sto_2, fn_sto_2 = 0, 0, 0, 0
        tp_sto_3, tn_sto_3, fp_sto_3, fn_sto_3 = 0, 0, 0, 0

        tp_full, tn_full, fp_full, fn_full = 0, 0, 0, 0
        tp_full_2, tn_full_2, fp_full_2, fn_full_2 = 0, 0, 0, 0
        tp_full_3, tn_full_3, fp_full_3, fn_full_3 = 0, 0, 0, 0

        # Predict all for test set
        for row in range(len(test_feat)):

            pred = self.predict_current(w0, weights, test_feat[row])
            pred_2 = self.predict_current(w0_2, weights_2, test_feat[row])
            pred_3 = self.predict_current(w0_3, weights_3, test_feat[row])

            pred_sto = self.predict_current(w0_sto, weights_sto, test_feat[row])
            pred_sto_2 = self.predict_current(w0_sto_2, weights_sto_2, test_feat[row])
            pred_sto_3 = self.predict_current(w0_sto_3, weights_sto_3, test_feat[row])

            pred_full_batch = self.predict_current(w0_full_batch, weights_full_batch, test_feat[row])
            pred_full_batch_2 = self.predict_current(w0_full_batch_2, weights_full_batch_2, test_feat[row])
            pred_full_batch_3 = self.predict_current(w0_full_batch_3, weights_full_batch_3, test_feat[row])

            preds[row], preds_2[row], preds_3[row] = pred, pred_2, pred_3
            preds_sto[row], preds_sto_2[row], preds_sto_3[row] = pred_sto, pred_sto_2, pred_sto_3
            preds_full_batch[row], preds_full_batch_2[row], preds_full_batch_3[row] = pred_full_batch, pred_full_batch_2, pred_full_batch_3

            # If statements to collect false and correct predictions for mini-batch
            if abs(test_labels[row] - pred) != 0:
                falses += 1
                if pred == 0:
                    fn += 1
                else:
                    fp += 1
            else:
                corrects += 1
                if pred == 1:
                    tp += 1
                else:
                    tn += 1

            if abs(test_labels[row] - pred_2) != 0:
                falses_2 += 1
                if pred_2 == 0:
                    fn_2 += 1
                else:
                    fp_2 += 1
            else:
                corrects_2 += 1
                if pred_2 == 1:
                    tp_2 += 1
                else:
                    tn_2 += 1

            if abs(test_labels[row] - pred_3) != 0:
                falses_3 += 1
                if pred_3 == 0:
                    fn_3 += 1
                else:
                    fp_3 += 1
            else:
                corrects_3 += 1
                if pred_3 == 1:
                    tp_3 += 1
                else:
                    tn_3 += 1

            # If statements to collect false and correct predictions for stochastic
            if abs(test_labels[row] - pred_sto) != 0:
                falses_sto += 1
                if pred_sto == 0:
                    fn_sto += 1
                else:
                    fp_sto += 1
            else:
                corrects_sto += 1
                if pred_sto == 1:
                    tp_sto += 1
                else:
                    tn_sto += 1

            if abs(test_labels[row] - pred_sto_2) != 0:
                falses_sto_2 += 1
                if pred_sto_2 == 0:
                    fn_sto_2 += 1
                else:
                    fp_sto_2 += 1
            else:
                corrects_sto_2 += 1
                if pred_sto_2 == 1:
                    tp_sto_2 += 1
                else:
                    tn_sto_2 += 1

            if abs(test_labels[row] - pred_sto_3) != 0:
                falses_sto_3 += 1
                if pred_sto_3 == 0:
                    fn_sto_3 += 1
                else:
                    fp_sto_3 += 1
            else:
                corrects_sto_3 += 1
                if pred_sto_3 == 1:
                    tp_sto_3 += 1
                else:
                    tn_sto_3 += 1

            # If statements to collect false and correct predictions for full-batch
            if abs(test_labels[row] - pred_full_batch) != 0:
                falses_full_batch += 1
                if pred_full_batch == 0:
                    fn_full += 1
                else:
                    fp_full += 1
            else:
                corrects_full_batch += 1
                if pred_full_batch == 1:
                    tp_full += 1
                else:
                    tn_full += 1

            if abs(test_labels[row] - pred_full_batch_2) != 0:
                falses_full_batch_2 += 1
                if pred_full_batch_2 == 0:
                    fn_full_2 += 1
                else:
                    fp_full_2 += 1
            else:
                corrects_full_batch_2 += 1
                if pred_full_batch_2 == 1:
                    tp_full_2 += 1
                else:
                    tn_full_2 += 1

            if abs(test_labels[row] - pred_full_batch_3) != 0:
                falses_full_batch_3 += 1
                if pred_full_batch_3 == 0:
                    fn_full_3 += 1
                else:
                    fp_full_3 += 1
            else:
                corrects_full_batch_3 += 1
                if pred_full_batch_3 == 1:
                    tp_full_3 += 1
                else:
                    tn_full_3 += 1

        #error = np.sum(np.abs(test_labels - preds) / len(test_labels))
        print("\nFOR LR 10^4: -----------------------")
        print("False preds: ", falses)
        print("Correct preds: ", corrects)
        accuracy = corrects / len(test_labels)
        print("Accuracy: ", accuracy)
        FPR = fp / (fp + tn)
        print("TP, TN, FP, FN: ", tp ,tn, fp, fn)
        print("False Positive Rate: ", FPR)
        NPV = tn / (fn + tn)
        print("Negative Predictive Value: ", NPV)
        precision = tp / (tp + fp)
        FDR = fp / (fp + tp)
        print("False Discovery Rate: ", FDR)
        print("Precision: ", precision)
        recall = tp / (tp + fn)
        print("Recall: ", recall)
        f1 = (2 * precision * recall) / (precision + recall)
        f2 = (5 * precision * recall) / (4 * precision + recall)
        print("F1:", f1, "F2: ", f2)

        print("----STOCHASTIC----")
        print("False preds: ", falses_sto)
        print("Correct preds: ", corrects_sto)
        accuracy_sto = corrects_sto / len(test_labels)
        print("Accuracy: ", accuracy_sto)
        print("TP, TN, FP, FN: ", tp_sto, tn_sto, fp_sto, fn_sto)
        FPR_sto = fp_sto / (fp_sto + tn_sto)
        print("False Positive Rate: ", FPR_sto)
        NPV_sto = tn_sto / (fn_sto + tn_sto)
        print("Negative Predictive Value: ", NPV_sto)
        precision_sto = tp_sto / (tp_sto + fp_sto)
        FDR_sto = fp_sto / (fp_sto + tp_sto)
        print("False Discovery Rate: ", FDR_sto)
        print("Precision: ", precision_sto)
        recall_sto = tp_sto / (tp_sto + fn_sto)
        print("Recall: ", recall_sto)
        f1_sto = (2 * precision_sto * recall_sto) / (precision_sto + recall_sto)
        f2_sto = (5 * precision_sto * recall_sto) / (4 * precision_sto + recall_sto)
        print("F1:", f1_sto, "F2: ", f2_sto)

        print("----FULL BATCH----")
        print("False preds: ", falses_full_batch)
        print("Correct preds: ", corrects_full_batch)
        accuracy_full_batch = corrects_full_batch / len(test_labels)
        print("Accuracy: ", accuracy_full_batch)
        print("TP, TN, FP, FN: ", tp_full, tn_full, fp_full, fn_full)
        FPR_full = fp_full / (fp_full + tn_full)
        print("False Positive Rate: ", FPR_full)
        NPV_full = tn_full / (fn_full + tn_full)
        print("Negative Predictive Value: ", NPV_full)
        precision_full = tp_full / (tp_full + fp_full)
        FDR_full = fp_full / (fp_full + tp_full)
        print("False Discovery Rate: ", FDR_full)
        print("Precision: ", precision_full)
        recall_full = tp_full / (tp_full + fn_full)
        print("Recall: ", recall_full)
        f1_full = (2 * precision_full * recall_full) / (precision_full + recall_full)
        f2_full = (5 * precision_full * recall_full) / (4 * precision_full + recall_full)
        print("F1:", f1_full, "F2: ", f2_full)

        print("\nFOR LR 10^3: -----------------------")
        print("False preds: ", falses_3)
        print("Correct preds: ", corrects_3)
        accuracy_3 = corrects_3 / len(test_labels)
        print("TP, TN, FP, FN: ", tp_3, tn_3, fp_3, fn_3)
        print("Accuracy: ", accuracy_3)
        FPR_3 = fp_3 / (fp_3 + tn_3)
        print("False Positive Rate: ", FPR_3)
        NPV_3 = tn_3 / (fn_3 + tn_3)
        print("Negative Predictive Value: ", NPV_3)
        precision_3 = tp_3 / (tp_3 + fp_3)
        FDR_3 = fp_3 / (fp_3 + tp_3)
        print("False Discovery Rate: ", FDR_3)
        print("Precision: ", precision_3)
        recall_3 = tp_3 / (tp_3 + fn_3)
        print("Recall: ", recall_3)
        f1_3 = (2 * precision_3 * recall_3) / (precision_3 + recall_3)
        f2_3 = (5 * precision_3 * recall_3) / (4 * precision_3 + recall_3)
        print("F1:", f1_3, "F2: ", f2_3)

        print("----STOCHASTIC----")
        print("False preds: ", falses_sto_3)
        print("Correct preds: ", corrects_sto_3)
        accuracy_sto_3 = corrects_sto_3 / len(test_labels)
        print("Accuracy: ", accuracy_sto_3)
        print("TP, TN, FP, FN: ", tp_sto_3, tn_sto_3, fp_sto_3, fn_sto_3)
        FPR_sto_3 = fp_sto_3 / (fp_sto_3 + tn_sto_3)
        print("False Positive Rate: ", FPR_sto_3)
        NPV_sto_3 = tn_sto_3 / (fn_sto_3 + tn_sto_3)
        print("Negative Predictive Value: ", NPV_sto_3)
        precision_sto_3 = tp_sto_3 / (tp_sto_3 + fp_sto_3)
        FDR_sto_3 = fp_sto_3 / (fp_sto_3 + tp_sto_3)
        print("False Discovery Rate: ", FDR_sto_3)
        print("Precision: ", precision_sto_3)
        recall_sto_3 = tp_sto_3 / (tp_sto + fn_sto_3)
        print("Recall: ", recall_sto_3)
        f1_sto_3 = (2 * precision_sto_3 * recall_sto_3) / (precision_sto_3 + recall_sto_3)
        f2_sto_3 = (5 * precision_sto_3 * recall_sto_3) / (4 * precision_sto_3 + recall_sto_3)
        print("F1:", f1_sto_3, "F2: ", f2_sto_3)

        print("----FULL BATCH----")
        print("False preds: ", falses_full_batch_3)
        print("Correct preds: ", corrects_full_batch_3)
        accuracy_full_batch_3 = corrects_full_batch_3 / len(test_labels)
        print("Accuracy: ", accuracy_full_batch_3)
        print("TP, TN, FP, FN: ", tp_full_3, tn_full_3, fp_full_3, fn_full_3)
        FPR_full_3 = fp_full_3 / (fp_full_3 + tn_full_3)
        print("False Positive Rate: ", FPR_full_3)
        NPV_full_3 = tn_full_3 / (fn_full_3 + tn_full_3)
        print("Negative Predictive Value: ", NPV_full_3)
        precision_full_3 = tp_full_3 / (tp_full_3 + fp_full_3)
        FDR_full_3 = fp_full_3 / (fp_full_3 + tp_full_3)
        print("False Discovery Rate: ", FDR_full_3)
        print("Precision: ", precision_full_3)
        recall_full_3 = tp_full_3 / (tp_full_3 + fn_full_3)
        print("Recall: ", recall_full_3)
        f1_full_3 = (2 * precision_full_3 * recall_full_3) / (precision_full_3 + recall_full_3)
        f2_full_3 = (5 * precision_full_3 * recall_full_3) / (4 * precision_full_3 + recall_full_3)
        print("F1:", f1_full_3, "F2: ", f2_full_3)

        print("\nFOR LR 10^2: -----------------------")
        print("False preds: ", falses_2)
        print("Correct preds: ", corrects_2)
        accuracy_2 = corrects_2 / len(test_labels)
        print("Accuracy: ", accuracy_2)
        print("TP, TN, FP, FN: ", tp_2, tn_2, fp_2, fn_2)
        FPR_2 = fp_2 / (fp_2 + tn_2)
        print("False Positive Rate: ", FPR_2)
        NPV_2 = tn_2 / (fn_2 + tn_2)
        print("Negative Predictive Value: ", NPV_2)
        precision_2 = tp_2 / (tp_2 + fp_2)
        FDR_2 = fp_2 / (fp_2 + tp_2)
        print("False Discovery Rate: ", FDR_2)
        print("Precision: ", precision_2)
        recall_2 = tp_2 / (tp_2 + fn_2)
        print("Recall: ", recall_2)
        f1_2 = (2 * precision_2 * recall_2) / (precision_2 + recall_2)
        f2_2 = (5 * precision_2 * recall_2) / (4 * precision_2 + recall_2)
        print("F1:", f1_2, "F2: ", f2_2)

        print("----STOCHASTIC----")
        print("False preds: ", falses_sto_2)
        print("Correct preds: ", corrects_sto_2)
        accuracy_sto_2 = corrects_sto_2 / len(test_labels)
        print("Accuracy: ", accuracy_sto_2)
        print("TP, TN, FP, FN: ", tp_sto_2, tn_sto_2, fp_sto_2, fn_sto_2)
        FPR_sto_2 = fp_sto_2 / (fp_sto_2 + tn_sto_2)
        print("False Positive Rate: ", FPR_sto_2)
        NPV_sto_2 = tn_sto_2 / (fn_sto_2 + tn_sto_2)
        print("Negative Predictive Value: ", NPV_sto_2)
        precision_sto_2 = tp_sto_2 / (tp_sto_2 + fp_sto_2)
        FDR_sto_2 = fp_sto_2 / (fp_sto_2 + tp_sto_2)
        print("False Discovery Rate: ", FDR_sto_2)
        print("Precision: ", precision_sto_2)
        recall_sto_2 = tp_sto_2 / (tp_sto + fn_sto_2)
        print("Recall: ", recall_sto_2)
        f1_sto_2 = (2 * precision_sto_2 * recall_sto_2) / (precision_sto_2 + recall_sto_2)
        f2_sto_2 = (5 * precision_sto_2 * recall_sto_2) / (4 * precision_sto_2 + recall_sto_2)
        print("F1:", f1_sto_2, "F2: ", f2_sto_2)

        print("----FULL BATCH----")
        print("False preds: ", falses_full_batch_2)
        print("Correct preds: ", corrects_full_batch_2)
        accuracy_full_batch_2 = corrects_full_batch_2 / len(test_labels)
        print("Accuracy: ", accuracy_full_batch_2)
        print("TP, TN, FP, FN: ", tp_full_2, tn_full_2, fp_full_2, fn_full_2)
        FPR_full_2 = fp_full_2 / (fp_full_2 + tn_full_2)
        print("False Positive Rate: ", FPR_full_2)
        NPV_full_2 = tn_full_2 / (fn_full_2 + tn_full_2)
        print("Negative Predictive Value: ", NPV_full_2)
        precision_full_2 = tp_full_2 / (tp_full_2 + fp_full_2)
        FDR_full_2 = fp_full_2 / (fp_full_2 + tp_full_2)
        print("False Discovery Rate: ", FDR_full_2)
        print("Precision: ", precision_full_2)
        recall_full_2 = tp_full_2 / (tp_full_2 + fn_full_2)
        print("Recall: ", recall_full_2)
        f1_full_2 = (2 * precision_full_2 * recall_full_2) / (precision_full_2 + recall_full_2)
        f2_full_2 = (5 * precision_full_2 * recall_full_2) / (4 * precision_full_2 + recall_full_2)
        print("F1:", f1_full_2, "F2: ", f2_full_2)

    # Q4 METHODS
    def svm(self):
        print("\n")
        print("---Q4: SVM---------------------")
        # Take Data
        mat = loadmat(self.svm_path)

        print(mat.keys())

        # Inception = (1250, 2048), images = (1250, 64, 64, 3), labels = (1250, 1)
        inception = mat['inception_features']
        images = mat['images']
        labels = mat['class_labels']
        C = [10**(-6), 10**(-4), 10**(-2), 1, 10, 10**10]

        #inception = np.array(inception).astype(np.object)

        f1_feat = inception[:250]
        f1_label = labels[:250]
        f2_feat = inception[250:500]
        f2_label = labels[250:500]
        f3_feat = inception[500:750]
        f3_label = labels[500:750]
        f4_feat = inception[750:1000]
        f4_label = labels[750:1000]
        f5_feat = inception[1000:]
        f5_label = labels[1000:]

        fold1 = np.concatenate((f3_feat, f4_feat, f5_feat))
        fold1_val = f2_feat
        fold2 = np.concatenate((f1_feat, f4_feat, f5_feat))
        fold2_val = f3_feat
        fold3 = np.concatenate((f1_feat, f2_feat, f5_feat))
        fold3_val = f4_feat
        fold4 = np.concatenate((f1_feat, f2_feat, f3_feat))
        fold4_val = f5_feat
        fold5 = np.concatenate((f2_feat, f3_feat, f4_feat))
        fold5_val = f1_feat

        fold1_label = np.concatenate((f3_label, f4_label, f5_label))
        fold2_label = np.concatenate((f1_label, f4_label, f5_label))
        fold3_label = np.concatenate((f1_label, f2_label, f5_label))
        fold4_label = np.concatenate((f1_label, f2_label, f3_label))
        fold5_label = np.concatenate((f2_label, f3_label, f4_label))
        fold1_val_label = f2_label
        fold2_val_label = f3_label
        fold3_val_label = f4_label
        fold4_val_label = f5_label
        fold5_val_label = f1_label

        #k_fold = GroupKFold(n_splits=5)
        #k_fold.get_n_splits(inception, labels)
        #print(k_fold)

        parameters = {'kernel': ('linear', 'rbf'), 'C': C}

        grid = GridSearchCV(estimator=svm.SVC, param_grid=parameters)
        #grid.fit(fold1_val)


        #clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5)
        #clf.fit(train_feat, label_train)

        #print(clf.best_params_)

        #svc_classifier = SVC(C=10, kernel='linear')
        #svc_classifier.fit(train_feat, label_train)

        #pred = svc_classifier.predict(test_feat)
        #print(pred)

        #print(classification_report(test_feat, pred))


if __name__ == '__main__':
    pca = PCA("datasets/van_gogh", "datasets/q2_dataset.csv", "datasets/q3_train_dataset.csv", "datasets/q3_test_dataset.csv", "datasets/q4_dataset.mat")

    # Q1: PCA
    images = pca.load_images()
    x1, x2, x3 = pca.preprocess(images)
    pca.pca_first_hundred(x1, x2, x3)
    pca.noisy_van_gogh()

    # Q2: Linear Regression
    pca.linear_regression()

    # Q3: Logistic Regression
    pca.logistic_reg()

    # Q4: SVM
    pca.svm()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/