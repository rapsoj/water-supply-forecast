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

## Additional thoughts:
- Explainability is still unclear. Based on NRCS still using PCR as SOTA for long timescale streamflow prediction [src](https://www.sciencedirect.com/science/article/pii/S0022169421008325?ref=pdf_download&fr=RR-2&rr=821ef5f08c5c52c0) and the M4 model, which is apparently used in practice, attempting to explain the different feature's weights, it seems that this might be what they mean by explainability. Need to ask an expert about this!
- Many works exist that are unused in practice although they allegedly improve results. Why is this? Is it because of the explainability? Why would PCR instead of a better method be used?
- If feature weights are important, why does statistical info regarding them or getting their importance through different means (eg. testing the model's sensitivity to them by taking the gradient w.r.t them) good enough?
