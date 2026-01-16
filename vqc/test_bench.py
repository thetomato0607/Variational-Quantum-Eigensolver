import numpy as np
import matplotlib.pyplot as plt 
from sklearn.svm import SVC 
from sklearn.metrics import roc_curve, auc

# To distinguish the b-jets, we have to use two variables
# Impact parameter: Distance of track from the main collision
# Secondary vertex mass: Mass of the particles at the displaced vertex

n_events = 10000

# Signal from b-jets: High Ip and measurable SVM
sig_ip = np.random.normal(3.5, 1, n_events)
sig_mass = np.random.normal(2.5, 0.8, n_events)
sig_labels = np.ones(n_events)

# Background noisy signals from light jets
bkg_ip = np.random.exponential(1, n_events)
bkg_mass = np.random.exponential(0.5, n_events)
bkg_labels = np.zeros(n_events)

# Combine into one dataset
x = np.column_stack((np.concatenate([sig_ip, bkg_ip]), 
                     np.concatenate([sig_mass, bkg_mass])))
y = np.concatenate([sig_labels, bkg_labels])

# I will only demonstrate a simple code to classify the jet signals from the background
# But in real project, I am going to replace this SVC with a quantum kernal 
# Using Qiskit or PennyLane to find a better decision boundary.

classifier = SVC(probability=True)
classifier.fit(x, y)

# To plot a graph abt the rejection power
y_probs = classifier.predict_proba(x)[:, 1]
fpr, tpr, _ = roc_curve(y, y_probs)

plt.plot(tpr, 1-fpr, lw = 2, label = "ML Tagger")
plt.plot([0, 1], [1,0], color = "navy", linestyle = "--", label = "Random Cut")
plt.xlabel("b-jet Efficiency / Signal Kept")
plt.ylabel("Light jet Rejection / 1 - Background Eff")
plt.yscale("log")
plt.title("Background Rejection Performance")
plt.legend(loc = "lower left")
plt.grid(True, which = "both", ls = "-", alpha = 0.2)
plt.show