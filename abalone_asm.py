# %% [markdown]
# # ABALONE REPORT DESCRIPTION
# Abalone are large slow-growing marine snails. In many parts of the world they are an economically significant fishery as both commercial operations and a traditionally-important food source for many cultures.
# 
# Abalone are harvested from the wild rather than farmed, and sustainability of a slow-growing resources is an important issue. A key issue is determining the age of a specimen. The rigorous approach is to harvest and dissect the specimen to count growth rings in the flesh. Obviously, a reliable non-fatal means of estimating specimen age is highly desirable.
# 
# This dataset, provided as abalone_growth_data.csv, collects a large number of measurements on around 4000 harvested specimens and can be used to identify reliable predictors of the sample age.
# 
# As well as the spatial dimensions of the shell, the quantities provided include the whole weight, the shucked weight (ie. the mass of the flesh that is eaten), the shell weight, and the viscera weight (the non-edible organs that are discarded)
# 

# %%
# Library loading
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# ## TASK 1: Initial loading [5 marks]
# a. Download one of the datasets, and write code to read in the data to a Pandas DataFrame
# 
# b. Write code to print out the column headings provided in the datasets

# %%
# Loading the data file
marine_df = pd.read_csv('/Users/quinhuonn/Downloads/abalone_growth.csv')

# Column heading in the dataset
print('columns of the dataset:')
display(marine_df.columns)


# %%
# Get insight about the dataset 
print('Shape of the dataset:')
display(marine_df.shape)
# It means the dataset has total 4041 observation 
# after handling missing values and 10 attributes for each observation.

# %%
# Information about dataset 
print('Info of the dataset:')
display(marine_df.info())

# %% [markdown]
# **Overview**
# 
# All the attributes are <span style='color: lightcoral'> numerical values </span>, apart from "Sex" attributes.
# The Abalone dataset provides a wealth of information that can lead to valuable insights regarding the physical characteristics and age estimation of abalones.
# 
# **Dataset Structure**
# 
# The dataset consists of `4,177 entries` and `9 columns`, which include both `categorical and continuous` variables.
# Physical Measurements
# 
# **The physical measurements** 
# 
# <span style='color: lightcoral'> Length, Diameter, Height, Whole Weight, Shucked Weight, Viscera Weight, Shell Weight </span>: are all continuous variables that can significantly influence the age estimation of abalones.
# 
# 
# **Missing Values and Data Quality**
# 
# The dataset appears to have some missing value and non-sensical values across all columns, which is neccesary for cleaning data analysis steps as it ensures data integrity and reliability.

# %% [markdown]
# ## TASK 2: Data cleaning [5 marks]
# a. Unfortunately, all the datasets have been damaged and contain missing, nonsensical, or NaN fields. Write code to look for any problems with the data and remove problematic entries
# 
# b. In your report.pdf file, add a section named “Data Cleaning” which describes what data cleaning you have done, and what other cleaning might be possible

# %% [markdown]
# ### Step 1: Handling missing value 

# %%
#check if missing value is existing in individual columns
marine_df.isnull().sum()

# %%
#check total value for each individual columns
marine_df.count()

# %% [markdown]
# Obseving that certain columns (such as Length (mm), Diameter (mm), Height (mm)) contain `approximately 1% or fewer missing values`. Upon further checking of the distribution of these columns, it becomes apparent that the data is considerably skewed, either to the right or left. Given that the total proportion of missing values is relatively low, I am considering the removal of rows with missing values in these columns to <span style='color: #0e6655'> maintain the integrity </span> of the dataset and <span style='color: #0e6655'> minimize potential distortions </span>.

# %% [markdown]
# Usually, we're looking for <Median> instead of <Mean> causes Mean affected too heavily by outliers => Not a good metrics to analyze and gain a valuable insight. 
# 
# [Save for numerical analysis]

# %%
# Drop columns whose missing values is less than 1%
marine_df.dropna(
    axis= 'index', 
    how= 'any', 
    subset= ['Length (mm)', 'Diameter (mm)', 'Height (mm)'], 
    inplace= True
)

# Check total of NA after dropping
print('Total of NA:\n',marine_df.isnull().sum())


# %% [markdown]
# For the rest of missing value, we considered to replace them with the average value

# %%
# Fill NA with average for each column
cols_to_fill = [
    'Height (mm)', 'Whole weight (g)',
    'Shucked weight (g)', 'Viscera weight (g)', 
    'Shell weight (g)', 'Rings', 'Age (y)'
]

marine_df[cols_to_fill] = marine_df[cols_to_fill].fillna(
    marine_df[cols_to_fill].mean()
)

#check missing value again
marine_df.isnull().sum()


# %%
# Calling out df after cleaning
marine_df

# %% [markdown]
# Ater filling Na with the mean and removing all the missing value from three columns named "...", I erase total of 4177 - 4041 observations. We can see that our dataset is now free of all the missing values and after dropping the data the number of rows also <span style='color: #0e6655'> reduced from 4155 to 4041 </span>

# %% [markdown]
# The dataset can be divided into `numerical` and `categorical columns`:
# 
# `Numerical Columns:`
# 
# - Length
# 
# - Diameter
# 
# - Height
# 
# - Whole weight
# 
# - Shucked weight
# 
# - Viscera weight
# 
# - hell weight
# 
# - Rings
# 
# `Categorical Columns:`
# 
# - Sex
# 

# %%
# Categorical data type unique values
print(
    'Total count of each unique value for categorical variable gender: '
)
display(marine_df['Sex'].value_counts())

# %% [markdown]
# ### Step 2: Non-sensical values
# 
# <span style='color:rgb(6, 64, 52)'> Problems </span>: 
# 
# those measurements are to tell us about the appearance features of those marines but we observed some negative values for ratio data.

# %%
# Define negative value
numeric_cols = [
    'Length (mm)', 'Diameter (mm)', 
    'Height (mm)', 'Whole weight (g)',
    'Shucked weight (g)', 'Viscera weight (g)', 
    'Shell weight (g)', 'Rings', 'Age (y)']
negative_values = (marine_df[numeric_cols] < 0).any(axis=1)

# Call back nonsensical values to check
marine_df.loc[negative_values]

# %%
# Count total of negative value in each individual column
negative_values = (marine_df[numeric_cols] < 0)
negative_values.sum()

# %% [markdown]
# When looking at distribution of each column, it is clearly that most of them is followed approxiamately normal distribution, i decided to <span style='color:rgb(38, 170, 144)'>replace all the negative values with `median` </span> to ensure the interity and keep most of rows as much as possible. 

# %%
# Convert them to Nan value 
marine_df[marine_df[numeric_cols] < 0] = np.nan
marine_df

# %%
# Replace them into median
num_df = marine_df[numeric_cols].fillna(marine_df[numeric_cols].median())
num_df.isnull().sum()

#describe the numeric df
num_df



# %% [markdown]
# ### Step 3: Duplicates 

# %%
# Define duplicates rows
marine_df.duplicated().sum()

# %%
# remove duplicate row if any 
marine_df.drop_duplicates()

# %% [markdown]
# <span style='color:rgb(5, 51, 42)'> IN CONCLUSION, there's no duplicates obsereved in the whole dataset </span>

# %% [markdown]
# ## DESCRPIPTVE ANALYSIS. 
# 

# %%
# Descriptive statistic
print("Statistical Summary:")
display(num_df.describe())


# %% [markdown]
# ### Handling Outliers
# 
# - Focus on num_df meaning numerical data types, identifying the key outliers can be a key significantly impact for analysis later on 
# 
# - We'll use <span style='color:rgb(26, 127, 107)'> the Interquartile Range (IQR) method </span> to identify outliers in these variables. The IQR method is robust as it defines outliers based on the statistical spread of the data.

# %%
# Calculate IQR for numerical data_ choosing key metrics: AGE
Q1_age = num_df['Age (y)'].quantile(0.25)
Q3_age = num_df['Age (y)'].quantile(0.75)
IQR_age = Q3_age - Q1_age

# %%
#Define those values which is out of normal range 
outlier_age = num_df[
    (num_df['Age (y)'] < (Q1_age - 1.5*IQR_age)) | 
    (num_df['Age (y)'] > (Q3_age + 1.5*IQR_age))
                     ]
outlier_age.head(15)

# %%
marine_df['Age (y)']

# %%
# Create a boxplot for age columns
plt.figure(figsize=(10.4,6.4))
plt.boxplot(num_df['Age (y)'], orientation= 'horizontal')
plt.title('Box Plot of Age Measurements', color= "#0c483c")
plt.ylabel('Age', color='#0e6655')
plt.grid(True)
plt.show()

# %%
# Display the number of outliers detected
print('Total outlier values exist currently in our dataset:', outlier_age.shape[0])

# %% [markdown]
# ## TASK 3: Numerical analysis [10 marks]
# a. Using NumPy techniques, pick two or more numerical valued columns in the dataset and find the mean, median and standard deviation of the data.
# 
# b. In your report.pdf file, add a section named “Numerical Analysis”. Include a table showing the generated statistical information. Add 3-4 sentences that describe what the numbers you have calculated above mean in the context of the data.

# %%
#Choosing four main numeric values for numerical analyis task:
main_cols = ['Height (mm)', 'Whole weight (g)', 'Age (y)', 'Rings']
print('Descriptive analysis table for 4 numerical variables:')
display(num_df[main_cols].describe())

# %%
# Numpy technique to find the mean, median and standard deviation
print('The average age of abalone:', num_df['Age (y)'].mean())
print('The median age of abalone:', num_df['Age (y)'].median())
print('The standard deviation of abalone age:',num_df['Age (y)'].std())

# %% [markdown]
# *** Some findings: 
# 
# Wide Range in Physical Size
# 
# For example, eight ranges from 0.002 g to 2.83 g, showing huge diversity.
# 
# ==> Suggests a mix of very young/small and large/mature abalones. 

# %% [markdown]
# Once again, we focus on numerical data:

# %%
#Seperating those numerical data and focusing on unique values in each column
unique = num_df.nunique()
print('Display total unique values for those numerical data:')
display(unique)

# %% [markdown]
# ### Key insight form Numerical features
# 
# There are a total of 4041 entries after the data cleaning task across all features. Among these, we can see that the numerical values have a wide range of distinct values as shown in the table, indicating the diversity and richness in the measurement and collection of physical body data for the abalones.
# 
# At the same time, it can be observed that "rings" and "age" have the same number of unique recorded values.
# 
# `Futher analysis` during task 4 intergrated with task 3.

# %% [markdown]
# ## TASK 4: Simple plot [10 marks]
# 
# Write code to make a simple plot showing, for example, the relationship between two quantities in the data using a scatterplot, a time series using a lineplot, or the distribution of one or more parameters using a histogram plot. The description of each of the datasets includes some suggestions for relations you could use.
# 
# Ensure that your plot:
# 
# a.  Includes a legend (if there is more than one curve), and suitable axis labels indicating the quantity and any units.
# 
# b.  Uses axis scales that are set to reliably reveal any trends in the data (you may need to scale the data and adjust the unit labels if that is required to make an attractive plot).
# 
# c.  Is generally attractive with appropriate point size, line width and/or and colours.
# 
# d.  Is accompanied by a short analysis (2-3 sentences) of what the plot reveals about your data.

# %% [markdown]
# Pick up the key columns: Age, Whole weight, Shucked weight, Shell Weight

# %%
# Define the list of columns to plot
cols = [
    'Height (mm)', 'Length (mm)', 
    'Shucked weight (g)', 'Whole weight (g)', 
    'Age (y)', 'Rings']

# Set up the grid (2x2 for 4 columns)
fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.4))

# Flatten the axes array for easy iteration
axes = axes.flatten()


# Plot each histogram
for i, col in enumerate(main_cols):
    axes[i].hist(
        num_df[col], bins=30, 
        color='#0e6655', edgecolor='black', 
        alpha = 0.5, lw= 2)
    axes[i].set_title(f'Histogram of {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')
    

plt.tight_layout()
plt.show()


# %%
# Define the list of columns to plot
cols = [
    'Height (mm)', 'Length (mm)', 
    'Shucked weight (g)', 'Whole weight (g)', 
    'Age (y)', 'Rings']

# Set up the grid (2x2 for 4 columns)
fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.4))

# Flatten the axes array for easy iteration
axes = axes.flatten()

# Plot each histogram
for i, col in enumerate(main_cols):
    axes[i].boxplot(num_df[col])
    axes[i].set_title(f'Box plot of {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')
    

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Height: 
# 
# **> Distributions: significantly right-skewed shapes**
# - Most of species in our sample are relatively short (small in sizes), while there are two species have longer height > 0.3mm (count as 2 species which have 1417 is 0.515mm and 2051 is 1.13mm (thelongest)) (there must be existing 2 outliers falling in left-hand side so that the x-axis is spreading like plot) => the apperance of two outliners show the potential data for rare large speciments. 
# 
# - The average height of abalone shells in our sample, depending on the species, are recorded as 0.139mm
# 
# - Most of the abalone in the sample have a height ranging from <span style="color: #16a085"> 0.1mm to 0.2mm </span>, with <span style="color: #16a085">a total of 3,140 specimens recorded </span>
# 
# - Notices that there are two species have <span style="color:rgb(7, 46, 38)"> the minimum height at 0.0mm </span>>, means that, potential data entry errors or missing measurements => requires neccessarily further exploration. 

# %%
# Two species are longer than 0.3mm
Long_species = num_df[num_df['Height (mm)'] > 0.3]
print('Two species is longer than 0.3mm:')
display(Long_species)

# Number of non-height hidden species that are recored 0.0mm in length
Missing_height = num_df[num_df['Height (mm)'] == 0.0]
print('Two species cannot find height')
display(Long_species)

# %% [markdown]
# ## Whole weight
# 
# > Shape: moderately right-skewed distribution.
# 
# - Mean is approximately equal with median, recording 0.827 and 0.828 respectively, which indicating that the distribution of total weight is slightly positive skewed
#  
# - The IQR indicates that most surveyed abalones overall weigh between <span style="color: #138d75 "> 0.4g to 1.2g </span>
# 
# - Few outliers have significantly total weight greater than 2.0grams, peaking up to 2.8255 grams. 
# 

# %%
num_df['Whole weight (g)'].max()

# %% [markdown]
# ## Age: 
# 
# Shape: <span style="color:rgb(14, 60, 51) "> Right-skewed distribution </span>, few outliers on the right hand side, meaning that there are much more young abalone than the older ones. 

# %% [markdown]
# **Key findings**
# 
# - Most of physical feature of the sureveyed abalone (Heigth, Weight, Length,) are witnessed <span style= 'color: lightseagreen'> right-skewed </span>, highlighting the attraction of smaller and lighter child abalones compared to fewer larger sample species. 

# %% [markdown]
# <span style= "color: tan"> Age versus Ring </span> 
# 
# The same patterns in histogram are witnessed in the distribution of `age` and `rings` seems reasonable as people usually calculate abalones age's using their shell ring. 
# 

# %%
num_df.sort_values(by= 'Age (y)', inplace=True)
num_df[['Age (y)', 'Rings','Whole weight (g)']]

# %% [markdown]
# Extracting from two column Age Rings, We can see that the variable 'Age' in this dataset has a certain correlation with the variable 'Rings', following a linear trend: 
# 
# `age = rings + 1.5`
# 
# <span style="color: #138d75 ">  That’s also why the plots generated from Age and Rings tend to show nearly identical patterns. </span>

# %% [markdown]
# **There is clearly a distinct pattern between rings and age — it’s evident that the more rings there are, the higher the age tends to be (which makes sense as it follows the linear equation mentioned above).**

# %%
# plotting the scatter plot for rings and age 
plt.figure(figsize= (10,6))
plt.scatter(
    num_df['Rings'],num_df['Age (y)'],
    c= num_df['Whole weight (g)'], s= 50, 
    cmap= 'BrBG', marker = '^', 
    alpha= 0.3, label='Whole weight')
plt.xlabel('Rings')
plt.ylabel('Age (y)')
plt.colorbar(label='Whole weight (grams)')
plt.legend()
plt.title('Correlation between Rings and Age (y) mapping by Whole weight', color= '#0b5345')
plt.show()

# %% [markdown]
# # Ring analysis

# %%
# creating sublots for rings
plt.figure(figsize= (12,6))

# histogram for rings in the first grid
plt.subplot(1,2,1)
plt.hist(
    num_df['Rings'], edgecolor= 'black', 
    bins = 25, lw= 2, alpha = 0.7, 
    color= '#a2d9ce')
plt.title('Histogram of Rings')
plt.xlabel('Rings')
plt.ylabel('Frequency')

# box plot for ring in the second grid
plt.subplot(1,2,2)
plt.boxplot(num_df['Rings'])
plt.title('Box plot for Rings')
plt.xlabel('Rings')

plt.tight_layout()
plt.show()

# %% [markdown]
# - the most frequence observed around <span style="color: #138d75 ">  8-10 rings </span>, concluding that this might be the most common age group. 
# 
# - The median also witnessed the same data for the number of rings appears to be around <span style="color: #138d75 ">  9-10 rings </span> 
# 
# - The box plot support the evidience that most of data concentrated on the left-hand side while several outliers exist on the right side, and <span style="color:rgb(201, 59, 16) ">  there is a significantly decline in the number of abalone as the rings increased </span>

# %% [markdown]
# # TASK 5: Multi-variable plot [10 marks]
# Considering the set of variables in your dataset, select 3 or 4 variables to explore their relationship in a plot that combines multi-variable plot techniques (for example, scatter dot size and/or colour as well as position)
# 
# a. Your plot should be included in the report.pdf and:
# 
# i. Include a legend (if there is more than one curve/colour etc.), and suitable axis labels indicating the quantity and any units
# 
# ii. Uses axis scales that are set to reliably reveal any trends in the data (you may need to scale the data and adjust the unit labels if that is required to make an attractive plot)
# 
# iii. Be generally attractive with appropriate point size, line width and/or and colours.
# 
# b. Your report.pdf should include a paragraph explaining what kind of potential relationships you are choosing to investigate. Add this in a section called “Multi-variable plot part b”
# 
# c. Your report.pdf contains a short analysis (2-3 sentences) of what the plot reveals about your data and draw conclusions from your plot. Add this in a section called “Multi-variable plot part c”

# %% [markdown]
# ### GOALS: DISTRIBUTION OF AGE BY SEX 

# %%
# Take a look at distinct sex 
print('3 subtegories gender of abalone:',marine_df['Sex'].unique().sum())


#Filter age by sex
M_ages = marine_df[marine_df['Sex'] == 'M']['Age (y)']
F_ages = marine_df[marine_df['Sex'] == 'F']['Age (y)']
I_ages = marine_df[marine_df['Sex'] == 'I']['Age (y)']

# Ploting 
plt.figure(figsize=(10, 6))
plt.hist(M_ages, bins= 10, alpha=0.6, label='Male', color= 'darkslategrey', edgecolor='black')
plt.hist(F_ages, bins= 10, alpha=0.7, label='Female',color= 'teal', edgecolor='black')
plt.hist(I_ages, bins= 10, alpha=0.8, label='Infant',color= 'lightcyan', edgecolor='black')
plt.title('Age Distribution by Sex', size= 20, color= 'darkslategrey')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.legend(fontsize= 15)
plt.tight_layout()
plt.show()

# %% [markdown]
# KEY FINDINGL: 
# 
# 
# - The distribution for all gender peak between age 9-11, indicating that the target age for commercial abalone seems to fall in this range.
# 
# - Infants (showed image) is likely to be the majority in the younger age range from  4-9.
# 
# - Males (dark gray) and females (teal) peak slightly later (10–13).
# 
# - After age 13:
# 
# >> Female frequency appears more evenly spread out than male.
# 
# >> Females tend to have greater longevity or are sampled across a broader age span.
# 
# - All groups decrease in the number of abalone from 15 onward. 
# 
# 

# %% [markdown]
# ### TASK 6: Extension task [10 marks]
# Extend the analysis of your previous investigation to incorporate the use of an untaught part of a library that we have discussed in class, which demonstrates further mastery and understanding of the library.
# 
# For example, you might apply a new statistical technique or use a type of plot that we have not discussed previously.
# 
# a. Write the code demonstrating the additional feature(s).
# 
# b. Your report.pdf should include an explanation of why your approach adds value to the analysis and what further insights it reveals about the data.
# 
# c. Your report.pdf should include a description of how your analysis goes beyond what was taught in class.
# 

# %% [markdown]
# ### Goal: FIND OUT THE `Features of abalone` GROUP BY `3 DISTINCT GROUP OF GENDER`

# %%
marine_df.columns

# %%
xyz = marine_df.groupby('Sex').mean().reset_index()


# %%
# seperating each group gender
gender_group = marine_df.groupby('Sex')[['Age (y)', 'Whole weight (g)']].agg(['count','mean']).reset_index()


# rename column for clarifying


# Calculate the Relative percentage by creating new column 'Relative percentag3e for df
gender_group['Relative percentage'] = gender_group['Age (y)', 'count']/ gender_group['Age (y)', 'count'].sum() *100
gender_group.columns


# %%
gender_df.columns

# %%
#Plot bar chart of counts from matplotlib
plt.figure(figsize=(12,8))
plt.bar(
    gender_group['Sex',''], gender_group['Age (y)', 'count'], 
    color=['#d0ece7','#73c6b6','#117a65'], 
    alpha= 0.5, edgecolor= 'black')
plt.xlabel('Sex', color= 'r')
plt.ylabel('Count of sex', color= 'b')
plt.legend()
plt.title('Count abalone by sex')
plt.show()



