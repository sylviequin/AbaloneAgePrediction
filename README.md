
# 🐚 Abalone Age Prediction

This project aims to predict the **age of abalone** large, slow-growing marine snails using physical measurements and weights, based on a dataset of ~4,000 harvested specimens. The goal is to find a **non-invasive** method of estimating age, without needing to dissect the animal.

---

##  Abalone Properties Context

**Abalone** are large, <span style="color:rgb(37, 150, 190);"> slow-growing marine snails </span>. In many parts of the world, they are an **economically significant fishery**, both as **commercial operations** and a **traditionally-important food source** for many cultures.

Abalone are **harvested from the wild** rather than farmed, and **sustainability** of a **slow-growing resource** is an important issue. A key issue is determining the **age** of a specimen. The rigorous approach is to **harvest and dissect** the specimen to **count growth rings** in the flesh. Obviously, a reliable **non-fatal means** of estimating specimen **age** is highly desirable.

This dataset, provided as **`abalone_growth.csv`**, collects measurements on around **4000 harvested specimens**, and can be used to **identify reliable predictors of sample age**.

---

## Dataset Description

| Feature Name      | Description                                                                 |
|-------------------|-----------------------------------------------------------------------------|
| `Sex`             | Categorical variable: **M** (Male), **F** (Female), **I** (Infant)          |
| `Length`          | Longest shell measurement (in mm)                                           |
| `Diameter`        | Perpendicular to length (in mm)                                             |
| `Height`          | With meat in shell (in mm)                                                  |
| `Whole weight`    | Weight of whole abalone (in grams)                                          |
| `Shucked weight`  | Weight of meat (i.e., edible portion)                                       |
| `Viscera weight`  | Gut weight after bleeding (non-edible organs)                              |
| `Shell weight`    | After being dried (in grams)                                                |
| `Rings`           | Number of growth rings → **Age = Rings + 1.5** (approximate in years)       |

---

## The aims of project

- Cleaning the raw and dirty dataset
- Explore correlations between physical features and abalone age  
- Apply the basic of machine learning models to predict age  
- Formula the linear metrics between ages and rings

---

## Model Types

- Linear Regression
- Data Cleaning
- Age Prediction formular
- Numerical analysis
- Multi-variable plot 

---

## Files

- `abalone_asm.ipynb`: Jupyter notebook with EDA and modeling
- `abalone_asm.py`: Python script version
- `abalone_growth.csv`: The entire dataset
- `abalone_asm.pdf/html`: Exported visible reports


## Author
[💡 sylviequin](https://github.com/sylviequin)
