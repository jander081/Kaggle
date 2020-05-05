############################################
##		MARKET DATA   		  ##
############################################

So we were able to create ema using groupby, transform, and lambda. Of course, with 40M samples, this probably won't fly. Look to vectorize the 
function and insert it instead of the lambda.



Finish RSI, and attempt in kernel PRIOR to trying to vectorize. We may not need to. Also, there may be an easier way to vectorize than the github link.


However, bear in mind that $p(t_o)$ is the price of the asset at the close of day to. For this reason, we will NOT KNOW
 that $p(t_o)>e_i(t_o)$ until the close of the trading day. Therefore, when calculating the returns of the strategy, to assume that on day $t_o$ we had a long position is an error; it is equivalent to us peaking into the future, since we only know we have to go long at the end of day to.

This should be accounted for with trading algos, not necessarily with forcasting returns


EXHAUSTED... notes..
Probably need to spend some time organizing and trying to submit what I have. 

The current technique is too slow for market data. I have to iterate through 365 days. Tacking on (growing) base dataframe for lag feats is a problem. I'd advise using the base feat approach for no more than 30 days and updating it. 

I can't imagine that the squeeze indicator is worth it. At least, not at this time.


still slow. try concatenating briefly to make the needed functions and then reduce
prior to preprocessing -- already doing that


#########
Notebooks
##########


1. market_working.ipynb: 
---------------------

This notebook basically mirrors the kernel with real data
Ran on the kernel with metric script. Score = .4. 

Score improvement with real world model = .7. Features modified. Pipelines implemented
scripts loaded onto kaggle. Run the kernel using this notebook. If score is decent, look to publish?




Notes -->
Try not 
to abstract/productionize too much. Keep it in prototype but organized. 
Implementing DFS may be a good move. 
Also, consider keeping market as a separate model from new. Different models could be used with a meta-learner. This may
also be an opportunity to use parallel processing.

One thing at a time though. Knock out the market. If you get a decent score, publish a public kernel.

2. market_feats.ipynb
--------------------

This has become the repository for feature engineering. The notebook is set up by cell block to continuely reset the dframe.
Works pretty well.

Look at BB narrowing as a signal for volatility


###########
scripts
###########

marketDataGen -> generates the real dataset for given asset names

universe_feat -> randomly assigns daily universe

ticker_ex -> separates ext 

market_trans.py
----------------
Regular transformers for pipelines