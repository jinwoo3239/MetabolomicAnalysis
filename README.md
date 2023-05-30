# MetabolomicAnalysis

 Metabolomics can facilitate a comprehensive understanding of cellular phenotypes by profiling the overall distribution of metabolites in biological samples such as tissues, fluids, and cells. Unlike genomics and proteomics studies, metabolomics can investigate biological endpoints that represent downstream events in a series of cellular changes. Since abnormal metabolite levels are highly associated with disease-related traits, metabolomics may provide evidence for the elucidation of disease mechanisms and the identification of diagnostic indicators. 

The raw data generated from MS or NMR analysis is processed to identify the metabolites and quantify their levels. Various statistical and bioinformatics tools are used to analyze the data and identify patterns, correlations, and biological pathways that are associated with the metabolites.

**One of the aims of this project is to facilitate preprocessing, statistical analysis, modeling, and visualization of metabolomic data**. It supports the following:


* Data Filtering
* Multivariate analysis (PCA, PLS-DA)
* Statistical analysis (Adjusted t-test, One-way ANOVA)
* Data visualization
* Receiver-Operating Characteristic curve analysis (estimator: logistic regression)
* Finding specific m/z values across multiple datasets

# Getting Started

## Prerequisites
* `python` >= 3.7
* `numpy`, `pandas`, `matplotlib`, `seaborn`, `scipy`, `statsmodels`


## Quickstart

Jupyter Notebook files are prepared as tutorials. You can simply analyze the metabolomic datasets by executing the code step by step. In metabolomics, results are checked at each stage of analysis, and the direction of analysis may be adjusted based on those previous results. Therefore, it is recommended to analyze metabolomic datasets in a step-by-step manner. Utilizing these Jupyter Notebook templates can facilitate this process and enable a more streamlined analysis of your dataset.

1. **[Data filtering and scaling](https://github.com/jinwoo3239/MetabolomicsAnalysis/1._Preprocessing.ipynb)**

    In metabolomics, data filtering is crucial for the accurate identification and quantification of metabolites (key features). Large amounts of data include both relevant and irrelevant information. The data filtering step is the process of removing unnecessary and low-quality data to improve the quality and reliability of data.

    In this module, I implemented two kinds of data filtering steps. This step is strongly recommended for datasets with large numbers of features.
    * `step_blank_substraction`
    
        A blank value is an intensity from a sample that does not contain the analyte of interest (eg, buffer, extracted solution). Subtracting the blank values from the original intensities can improve the accuracy of the analytical measurement by eliminating the background noise and interference from other sources.

        I implemented `step_blank_substraction` in the following process:


        1. Substrate the blank values from the original intensities.
        2. Replace negative values, including 0, with "ND". The ND indicates unnecessary information.
        3. Calculate the proportion of "ND" values for each feature within each group (eg, the proportion of ND of A features.. treat group: 10%, non-treated group: 20%)
           *Note, In this step, we do not consider the blank and QC (quality  control) group
        4. if the proportion of "ND" within each feature is more than 50% in all groups, the features are removed; hyperparameter: `critiera_ND_proportion` : (default = 0.5 (50%)).
    

    * `step_filter_rsd`

        RSD indicates Relative Standard Deviation (mean/standard deviation) and is a measure of the variability of precision of a set of data. A high RSD in the QC (quality control) samples indicates a problem with the reliability of the data.
        In this module, you can exclude some features that have RSD values above the threshold value (hyperparameter: `filter_ratio`) *Note, in this step, only RSD values of QC groups are considered. 



2. **[Multivariate analysis](https://github.com/jinwoo3239/MetabolomicsAnalysis/2._MultivariateAnalysis.ipynb)**

    Multivariate analysis is a set of statistical techniques used to analyze datasets that have multiple variables. In metabolomics analysis, multivariate analysis is used to analyze large and complex datasets that contain information on the abundance of many metabolites across multiple samples. The main objective of multivariate analysis in metabolomics is to identify patterns or trends in the data that can help to understand the underlying biology. 

    Some of the common multivariate analysis techniques used in metabolomics analysis include:

    `Principal component analysis (PCA)`: PCA is a technique used to reduce the dimensionality of a dataset by identifying the principal components that explain the largest amount of variation in the data. 

    `Partial least squares-discriminant analysis (PLS-DA)`: PLS-DA is a supervised multivariate analysis technique used to identify variables (metabolites) that are most responsible for the differences between groups of samples. Permutation tests or VIP score calculations are not yet implemented.


3. **[Feature selection](https://github.com/jinwoo3239/MetabolomicsAnalysis/3._Statical_analysis(features_selection).ipynb)** 

    Feature selection is to identify the key features (metabolites) associated with a particular biological condition or phenotypes. In this code (univariate feature selection), statistical tests, such as t-test and ANOVA, were performed.
    
    To compare the two groups, a Student's t-test was performed and the p-value was adjusted using Benjamini-Hochberg(BH) method to reduce the false discovery rate (FDR). It is implemented in the class `AdjustedTtestSelection`. The output file also includes the information on p-values obtained from the Shapiro-wilk test and fold-change (FC).

    To compare more than two groups, one-way ANOVA was used for multiple comparisons, followed by Bonferroni post-hoc tests. It is implemented in the class `OneWayAnova`. Other post hoc validation methods are not currently supported for `OneWayAnova`.

4. **[Comparison of key features' intensity](https://github.com/jinwoo3239/MetabolomicsAnalysis/4._Get_graph_selected_metabolite_intensity.ipynb)**

   This step simply graphs the intensity of features (metabolites) by groups. After completing the feature's annotation (ID) and grouping of selection features, this step must be performed.
   

5. **[Biomarker analysis](https://github.com/jinwoo3239/MetabolomicsAnalysis/5._Roc_curve.ipynb)** 

    This is the process of drawing the ROC curve for the selected features. A receiver operating characteristic (ROC) curve is used to evaluate the model's performance in distinguishing between groups. Only binary class (0 or 1) is supported, and the estimator used here is a logistic regression model. Rather than evaluating the performance of logistic regression, This step aimed to determine whether the selected metabolites exhibited distinct pattern differences between the two groups.


6. **[Finding the key feautres from other datatset](https://github.com/jinwoo3239/MetabolomicsAnalysis/6._Finding_key_metabolite_from_other_dataset.ipynb)**

    It is a task to check whether there are metabolites that we have identified in other datasets. Metabolites are screened based on the m/z value, and RT (retention time) is not considered. It must be necessary to recheck the results. This is simply the process of finding m/z values within the margin of error.


# Dataset
The dataset format is as follows:


* Features to be analyzed should be in columns and samples should be in rows.
* **"Name"** and **"Label"** columns must be included.
    - Name means the sample name, and Label indicated the name of a class that can group samples.
* A blank group in the label column must be named **blank** in the **Label** column, not "Blank", "B", or "b".
* Except for **Name** and **Label**, all data set values must be numeric.
* Only ".xlsx" and ".csv" files are allowed.
* example file is "example_dataset.xlsx"


Name        |Label       |*feature1*  |*feature2*  |..   .      |
:----------:|:----------:|:----------:|:----------:|:----------:| 
sample1     |Control     |501.2       |30.5        |...         |
sample2     |Control     |726.9       |20.3        |...         |
sample3     |Severe      |102.4       |600.5       |...         |
sample4     |Severe      |105.6       |750.5       |...         |
sample5     |blank       |5.0         |0.0         |...         |
sample6     |blank       |7.1         |0.0         |...         |




* Another dataset is essential to compare the intensities of selected features [(step4, Comparision of key featuers' intensity)](https://github.com/jinwoo3239/MetabolomicsAnalysis/4._Get_graph_selected_metabolite_intensity.ipynb).
* example file is "example_metabolite_Id.xlsx"


Features        |mz       |ID  |ID_group  |
:----------:|:----------:|:----------:|:----------:|
2.24_132.0663     |132.0663     |Hydroxyproline      |AA       |
2.63_132.1033    |132.1033     |Leucine       |AA        |
2.5_118.066     |118.066      |Indole    |Indole       |
2.5_146.0609     |146.0609      |Indole-3-carboxaldehyde      |Indole       |

* Another dataset is essential to find specific m/z values(maybe, biomarkers) in other multiple datasets [(step6, Finding the key feautres from other datatset)](https://github.com/jinwoo3239/MetabolomicsAnalysis/6._Finding_key_metabolite_from_other_dataset.ipynb).

* example file is "example_another_dataset_mapping.xlsx"

Features    |mz       |
:----------:|:----------:|
0.21_245.856     |245.856    |
0.22_241.8829     |241.8829    |
0.23_197.9048     |197.9048    |
0.27_189.8648    |189.8648     |
0.28_262.858     |262.858      |



# License
The example dataset must not be distributed beyond this project and must be used as toy datasets.


Source codes are distributed under the MIT License.



# Author
* Jinwoo Kim
* e-mail: jinwoo3239@gmail.com




