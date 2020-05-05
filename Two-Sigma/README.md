# Two-Sigma
Didn't make the submission deadline, but continued to work the dataset. Created simulators for the market data using real data

Spent considerable time setting this up locally. My intent was to work offline on this. However, the submission deadline has passed and I'm going to move on. There is a lot of work in this repository. I worked to vectorize a lot of typical technical indicators and had set it up to create lag features. Basically, the daily data would come in, be appended with needed number of days (base) in order to generate the lags, then the base would update and drop off. This was doable, but I needed to really watch complexity. So I used arrays and parallel processing. I can score this with the simulator, but I'm going to move on for now. 

There is a data generator script that is decent. It takes in a list of assets and pulls the market data in order to simulate the competition. One of the sets I was testing with was over 2M samples. Tried to zip the datasets - still too large. Use the script instead. It's in the market script folder.

