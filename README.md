# 🏠 Min-Max Normalization & PCA Workshop

## 👥 Our Team

**Team Number:** Group 4

**Team Members:**

- Davisen Veerasamy
- Carlos Gutierrez
- Rangeetha
- Nnamdi Ikengah

---

## 🎯 What This Workshop Is About

Machine learning models learn from numbers. If those numbers are on very different scales, the models get confused.

In this workshop we used a dataset of **2,000 houses** and did two things:

1. **Min-Max Normalization:** put every feature on the same scale (0 to 1).
2. **PCA (Principal Component Analysis):** looked for the main patterns in the data.

---

## 📊 The Data

Each house has these columns:

| Column | Smallest | Largest | What it means |
|---|---|---|---|
| Price | $100,000 | $1,077,909 | How much the house costs |
| Area_sqft | 400 | 3,763 | Size of the house in square feet |
| Num_Bedrooms | 1 | 5 | Number of bedrooms |
| Num_Bathrooms | 1 | 3 | Number of bathrooms |
| Year_Built | 1950 | 2022 | When the house was built |
| Lot_Size | 1,000 | 13,088 | Size of the land |

**The problem:** Price goes up to about **1 million**, but bathrooms only go up to **3**.
To a computer, bigger numbers look more important, even when they aren't.

---

## 📏 Part 1: Min-Max Normalization

### What is it?

A way to shrink every column so its values sit between **0 and 1**.

- The smallest value becomes **0**
- The largest value becomes **1**
- Everything else falls somewhere in between

### The formula

$$
x_{normalized} = \frac{x - x_{min}}{x_{max} - x_{min}}
$$

**In plain words:** subtract the smallest value, then divide by the range (largest minus smallest).

### Example

| Feature | Raw Value | Min | Max | Normalized |
|---|---|---|---|---|
| Bedrooms | 3 | 1 | 5 | (3 − 1) ÷ (5 − 1) = **0.5** |
| Price | 650,000 | 250,000 | 800,000 | (650,000 − 250,000) ÷ 550,000 = **0.72** |

Now a "0.5" in bedrooms and a "0.72" in price are on the **same scale** and can be compared fairly.

### What we built

- We wrote the normalization **by hand in Python** (no sklearn or numpy), so we had to understand every step.
- We normalized **Price, Area_sqft, Num_Bedrooms, Num_Bathrooms and Lot_Size**.
- We saved the normalized values in a copy of the table (`df_norm`), so the original data stays unchanged and we could compare before and after.
- We handled one tricky case: if every value in a column is the same, max − min = 0 and we would be **dividing by zero**. In that case we set the values to 0.

### Result

| Feature | Range Before | Range After |
|---|---|---|
| Price | 977,909 | 1.0 |
| Lot_Size | 12,088 | 1.0 |
| Area_sqft | 3,363 | 1.0 |
| Num_Bedrooms | 4 | 1.0 |
| Num_Bathrooms | 2 | 1.0 |

✅ Every feature now has the **same range**.

### 🗣️ Talking Points #1: Why Normalization Matters

**1. KNN: without normalization, "closest house" just means "closest price."**
KNN finds similar houses by measuring the distance between them. We measured how much each feature adds to the distance between the first two houses:

| Feature | Share of Distance (Before) | Share of Distance (After) |
|---|---|---|
| Price | **99.999%** | 2.9% |
| Area_sqft | 0.001% | 1.9% |
| Num_Bedrooms | ~0% | 19.0% |
| Num_Bathrooms | ~0% | 76.2% |
| Lot_Size | ~0% | ~0% |

Before normalizing, **Price was almost the entire distance**. After normalizing, the other features finally count.

**2. Gradient descent: equal scales mean balanced learning.**
Many models learn by making small adjustments, called gradient descent. The size of each adjustment depends on how big the feature's numbers are.
- Price (in the hundreds of thousands) gets **huge** adjustments.
- Bathrooms (1 to 3) gets **tiny** adjustments.

That makes training slow or unstable. With everything between 0 and 1, all features learn at a similar pace.

**3. Different units shouldn't decide importance.**
Dollars, square feet and room counts are completely different units. Normalization removes the units, so each feature gets a fair chance to matter.

---

## 🔍 Part 2: PCA (Principal Component Analysis)

### What is it?

PCA looks at all the features together and asks:
> **"In which direction do the houses differ the most?"**

It creates new summary features called **principal components (PCs)**:
- **PC1** is the direction where houses differ the **most**.
- **PC2** is the **next** most, and so on.
- Each PC is a mix of the original features. The **loadings** tell you how much of each feature is in that mix.

**Simple analogy:** imagine describing people using height, weight, shoe size and arm length. These all go up together, so PCA might combine them into one PC called "body size." One number then tells you most of the story.

### ⚠️ Important: PCA does NOT look at Price

We used Price as the **target** (the thing we would want to predict), so we left it out of PCA.
PCA only finds where the features spread out the most. It does **not** know which features affect price.

### Why normalize before PCA?

We ran PCA on both the raw data and the normalized data:

| | PC1 | PC2 | PC3 | PC4 |
|---|---|---|---|---|
| **Raw data** | 94.1% | 5.9% | 0.0% | 0.0% |
| **Normalized data** | 49.5% | 36.0% | 8.0% | 6.4% |

On raw data, PC1 was **just Lot_Size**, because it has the biggest numbers. That's about units, not real patterns.
After normalizing, the results reflect the actual data.

### What we did

1. Used the 4 normalized features: Area_sqft, Num_Bedrooms, Num_Bathrooms and Lot_Size.
   (We left out House_ID because it's just a label, Price because it's the target, and Year_Built because it measures age, not size.)
2. Ran PCA.
3. Checked how many PCs we need to explain **90%** of the variation.
4. Checked which features make up each PC.
5. Checked which PC is most related to Price.

### ✅ Challenge Answers

| Question | Answer |
|---|---|
| How many PCs explain at least 90%? | **3** (49.5% + 36.0% + 8.0% = 93.6%) |
| What makes up PC1? | **Num_Bathrooms** (almost 100%) |
| What makes up PC2? | **Num_Bedrooms** (almost 100%) |
| Which PC is most related to Price? | **PC3** (Lot_Size), but the correlation is only **0.017**, which is basically zero |
| What does PC1 represent? | **The number of bathrooms**, not "overall house size" |

### 🗣️ Talking Points #2: What PCA Told Us

**1. Normalizing changed the PCA results completely.**
Without normalizing, PCA said one component explains 94%, but only because Lot_Size has big numbers. After normalizing, we needed **3 out of 4 components** to reach 90%. That means PCA couldn't simplify this data much.

**2. Each PC is basically one original feature.**
PC1 = bathrooms, PC2 = bedrooms, PC3 = lot size, PC4 = area.
This happens when the features **aren't related to each other**. For example, bigger houses in this dataset don't have more bedrooms, so there's nothing for PCA to combine.
Bathrooms came first because they only have 3 values (0, 0.5 and 1 after normalizing). Those values are spread far apart, which gives bathrooms the most variation.

**3. "Most variation" does not mean "most important for price."**
None of the PCs were related to Price (every correlation was below 0.02).
PCA never looks at the target, so it can't tell you what drives price. To find that, you need a method that does use the target, like **regression**.

---

## 🏁 Conclusions

- **Always normalize before** using KNN, gradient descent, neural networks or PCA. Otherwise the feature with the biggest numbers takes over.
- **Min-Max normalization is simple:** subtract the min and divide by the range. It puts everything on a 0-to-1 scale.
- **PCA finds where the data spreads out**, not what predicts the target.
- **In this dataset**, the features aren't related to each other or to Price, so PCA couldn't combine them into a few strong patterns.

---

## 📂 Files in This Repository

| File | What it is |
|---|---|
| `MinMax_Normalization_Workshop.ipynb` | Our completed notebook with all code, results and answers |
| `data/housing_data.csv` | The housing dataset (2,000 houses) |
| `requirements.txt` | Python packages needed to run the notebook |
| `README.md` | This summary |

## ▶️ How to Run

```bash
pip install -r requirements.txt
jupyter notebook MinMax_Normalization_Workshop.ipynb
```

Then run all cells from top to bottom.
