# Data Mining Cup 2022

In this repository, we store all the code related to this competition. For more information about the 2022 data mining competition [click here](https://www.data-mining-cup.com/dmc-2022/). 

## Scenario
This year's scenario is all about Pia and Philip, a married couple. They started their new e-commerce business during the pandemic in 2020 by offering convenience goods online. They began by selling an assortment of masks and disinfectants, but quickly expanded to a wider range of various everyday commodities.

Having both a background in traditional and online retail, they are aware of how distant and impersonal online shopping can feel and, at the same time, how important customer guidance and recommendations are for long-term customer loyalty.

To differentiate themselves from the many other commodity shops, they decided to put an even more significant emphasis on personalized recommendations and offers.

One key element of this strategy is a customized weekly newsletter that personally addresses each of their clients. The newsletter includes user favorites, products similar customers liked, new additions, and special offers.

However, they quickly noticed a problem: repeated recommendations of recently purchased products. One quick workaround for this issue was implementing a filter that would exclude products from the recommendation for a fixed number of days. This, however, did not meet the high standards of Pia and Philip.

They are instead looking for a model that can reliably predict the week that a returning customer might repurchase one of their frequently purchased items.

By knowing the estimated week of replenishment, products can be added to the newsletter as a reminder, thus increasing basket sizes and profits.

Since the owners are only interested in the best possible solution, they organized a contest to benchmark competing prediction approaches.

## The Task
The participating teams' goal is to predict the user-based replenishment of a product based on historical orders and item features. Individual items and user specific orders are given for the period between 01/06/2020 and 31/01/2021. The prediction period is between 01/02/2021 and 28/02/2021, which is exactly four weeks long.

For a predefined subset of user and product combinations, the participants shall predict if and when a product will be purchased during the prediction period.

The prediction column in the ''submission.csv'' file must be filled accordingly.

* 0 - no replenishment during that period
* 1 - replenishment in the first week
* 2 - replenishment in the second week
* 3 - replenishment in the third week
* 4 - replenishment in the fourth week

The different columns are separated by the ''|'' symbol. A possible example of the solution file might
look like this:

**userID|itemID|prediction**\
**12|6723|0**\
**20|8272|1**\
**28|9873|4**\
**...**

## Feature Engineering 




## Our Approach