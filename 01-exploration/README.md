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
- Why is the training data annual and the forecast format is monthly? Is this a mistake?
- Is it better to transform the output variable to be percent change from average to account for gauge-specific variation not in the data?
- Are there any open source data sources on land use/land use change that we can use?
- How are streamflow adjustment equations applied?
- How can we make weekly predictions when the ground truth data is given in annual increments?

**Models**
- What is the M4 model that has been used in the past? How can we improve on its predictions and explainability?
