# SVM Kernel Explorer

An interactive visualization tool built with **Python**, **Matplotlib**, and **Scikit-learn** to understand how Support Vector Machines (SVMs) behave with different kernels and hyperparameters.

The goal of this project is to **learn the mathematics and intuition behind SVMs**, rather than simply using them as a machine learning library.

---

## Features

* Compare **Linear**, **RBF**, and **Polynomial** kernels side-by-side.
* Interactive **C** parameter slider.
* Interactive **Gamma** slider for nonlinear kernels.
* Switch between multiple datasets:

  * Linear
  * Blobs
  * Moons
  * Circles
  * XOR
* Visualize:

  * Decision boundary
  * Margin (Linear SVM)
  * Decision function contours
  * Support vectors
* Dataset explanation panel.
* Reset button to restore default parameters.

---

## Preview

### SVM Explorer

> Add a screenshot here.

```
images/explorer.png
```

---

## Project Structure

```
KernelTrickVisualizer/
│
├── app.py
├── explorer.py
├── plotting.py
├── svm.py
├── datasets.py
├── config.py
├── requirements.txt
└── README.md
```

---

## Technologies Used

* Python
* NumPy
* Matplotlib
* Scikit-learn

---

## What I Learned

While building this project I explored:

* How Support Vector Machines maximize the margin.
* Why only support vectors determine the decision boundary.
* The effect of the **C** parameter.
* The role of **Gamma** in the RBF kernel.
* Polynomial kernel behavior.
* Linear vs nonlinear decision boundaries.
* The Kernel Trick and why it allows nonlinear separation.
* Building interactive visualizations using Matplotlib widgets.
* Organizing Python projects into modular components.

---

## Running the Project

Clone the repository:

```bash
git clone https://github.com/<your-username>/KernelTrickVisualizer.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
python app.py
```

---

## Future Improvements

* Add Sigmoid kernel.
* Display training time.
* Animate how the decision boundary changes.
* Export generated plots.
* Add custom dataset generation.

---

## License

This project is open source and available under the MIT License.
