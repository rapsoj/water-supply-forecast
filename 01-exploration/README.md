# Exploration

Exploring different hypotheses and issues with the data for the DrivenData Water Supply Forecast Rodeo. Please track open questions, literature reviews, key hydrological terminology, and potential solutions to the challenge here.

The following Google Sheets documents are available for tracking data exploration progress:
- **Literature Review**: Academic and technical literature on current best practices and proposed methodology for streamflow forecasting
- **Hydrological Terminology**: Key terms and definitions for understanding water supply forecasting
- **Data Sources**: Descriptions for all available data sources for the Water Supply Forecast Rodeo
- **Variable Descriptions**: Summaries of all variables in the Water Supply Forecast Rodeo data sources

[Google Sheets link](https://docs.google.com/spreadsheets/d/1bqkxBPs88jt1aW8on0KQWt8KL3NOCux4M1tsnmY-UOA/edit?usp=sharing)

[Introduction to geospatial data using Python](https://developer.ibm.com/learningpaths/data-analysis-using-python/introduction-to-geospatial-data-using-python)

### Open Questions

**Data**
- Is it better to transform the output variable to be percent change from average to account for gauge-specific variation not in the data?
- Are there any open source data sources on land use/land use change that we can use?
- How are streamflow adjustment equations applied?
- Do spatio-temporal neighbourhood indicators improve predictions?

**Models**
- What is the M4 model that has been used in the past? How can we improve on its predictions and explainability?
- How do we calculate natural flow? What operations are performed to transformed observed flow into natural flow?
- What are the best physical models? Can produce a physical model forecast as a predictor?
- Can we improve performance on data outliers/extreme values by using re-sampling techniques or two-stage models?

**Explainability**
- Can data source-specific PCA improve explainability while still enabling dimensionality reductions?
- Can we apply callibration to improve forecast probabilities?
