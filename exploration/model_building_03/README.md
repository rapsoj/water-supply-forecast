# Model Building

Different model configurations for theDrivenData Water Supply Forecast Rodeo. Please track the different model configurations and the corresponding quantile loss here.

Note that the training data provides natural streamflow at the annual scale, while the target predictions are predicitons of annual natural streamflow made on a weekly basis. To do this, the model must produce an annual forecast every week using only data that is available up to that point in time. The forecasts should get more accurate to annual natural streamflow the as more time passes.

| Model                | Cleaning | Feat. Eng. | Scaling | Feat. Selection | Dim. Red. | Param. Tuning | MQL |
|----------------------|--------------|---------------|------------|----------------------|---------------|---------------------|--------|
|  |        |         |      |                  |         |                 |   |
|||||||
