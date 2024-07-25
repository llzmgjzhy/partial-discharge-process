# Partial Discharge data Processing Paradigm

Traditional partial discharge recognition can be roughly divided into two types.One is to use different feature maps such as PRPD and TF-map to try to classify data into different clusters and identify different types of partial discharge;Another approach is to extract different data features,such as pulse kurtosis,skewness,amplitude,etc.

From above methods,it can be seen that dimensionality reduction has been performed after data feature extraction,which exposes some problems:

1. rendering the original meaningful data dimensions meaningless.
2. Discreteness.pd data itself is continuous data.However,traditional process making it cannot be analyzed in the time domain.

so this repo try to a new method to classify pd data.detail procedure and code will be refined later.