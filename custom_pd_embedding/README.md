# steps

## 1. Manmade features extract

Code is defined in the feature_extract/manmade_features.py,that can be enriched manually added.

## 2. Network_features extract

Model choices is various,example is resnet18.The network needs to be trained before working in the custom_pd_embedding.

### 2.1 fine-tune network

For the most intuitive idea:network trained in the user's data,indicates that having ability to extract useful data features.

## stage summary

add these two features up to form an vector,which is the custom pd embedding.

## 3. fine-tune transformer(VIT)

Now we have manmade and network features,adding them up to form an input vector.The vector represent the partial discharge prpd map and it has the form of vector,so it can be like a word.