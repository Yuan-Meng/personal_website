---
title: Causal Inference in Data Science
date: 2021-11-12
math: true
tags:
    - causal inference
    - quasi-experiment
    - statistics
categories:
- tutorials
keywords:
    - causal inference, quasi-experiment
include_toc: true

---

What do you see? 👀

{{< figure src="https://www.dropbox.com/s/gfiw187r502jb1l/1t1_redgreen_2sec.gif?raw=1" width="350" >}}

You're probably thinking, *"The red block stopped, right after which the green block started to move coincidentally."*

Nice try... I know what you're thinking, *"The red block <u>made</u> the green one move"*. 

# Why Ask Why?

## We can't help but think about causation

Your stats professor can say *"correlation is not causation"* like a broken record, but we can't help but think about causation. Even 12-month-old babies don't think the motions of the two blocks are merely correlated. They look longer, as if *surprised*, when we reverse the sequence (but not if the blocks don't abide by Newton's Third Law, in which case the event can't be causal, [Kominsky et al., 2017](http://www.jfkominsky.com/demos.html)). If you're intrigued by causal cognition, I recommend [this talk](https://youtu.be/q0HLci67Tr8) by Tobias Gerstenberg.
 
But my priors tell me you're here for data science. Just as causation is a fundamental part of human cognition, it is also the backbone of data science and, perhaps, "the cement of the universe" --- as Hume beautifully put it in [*Abstract*](http://web.mnstate.edu/gracyk/courses/web%20publishing/hume'sabstract.htm).

## We can't make things happen without knowing why

Data scientists make good things happen, not just "[conducting post mortem examinations](https://eng.lyft.com/experimentation-in-a-ridesharing-marketplace-b39db027a66e)" to conclude what a product "died of". Driving this idea home, Lyft's Sean Taylor said in a [brilliant talk](https://www.youtube.com/watch?v=2dv7NrYExzo&t=568s) that data scientists should turn user insights into the best course of actions for achieving the goal of the product and the company --- 

$$\arg \max_{\mathrm{action}} \hat{E}[\mathrm{goal|data, action}].$$

Making things happen requires a causal understanding of the problem. Mere correlations won't do. Berkeley's [Alison Gopnik](http://alisongopnik.com/) always has the nicest example: While yellow fingers and smoking are both correlated with lung cancer, washing hands doesn't prevent cancer --- we can only prevent cancer if we "intervene" on the real causes.

## Yet we can't always use A/B testing to figure out why

The gold standard for establishing causality is [A/B testing](https://en.wikipedia.org/wiki/A/B_testing). Typically, we create experiment variants that only differ in the factor we care about (e.g., old vs. new feature) and randomly assign units (e.g., users) into these variants. If we observe different outcomes (e.g., click-through rates) in different variants, then we may conclude that the treatment has a *causal effect* on the outcome ("*Because* users have feature $X_1$ rather than $X_2$, they are more likely to click on content $Y$.").

Unfortunately, random assignment is not always possible, feasible, or ethical. In the smoking example, you can't in good conscience force someone to smoke, nor can you easily have them quit smoking. More generally, experiments that subject people to higher risks than they experience in day-to-day life will have a hard time passing ethics reviews. Other times, A/B testing is plain difficult: You can't test how people like different versions of your webpage when your startup only has 500 active users, or "assign" connections to social media users, to name a few examples.

Sometimes nothing stops you from running an A/B test yet your conclusions are most likely false. Say DoorDash is worried that a lack of Dasher supplies will lead to long delivery time during peak hours and A/B tested surge pricing (which they call "SOS pricing") on selected users. They found no time difference between the treatment and the control. Is SOS pricing useless? We don't know: Customers in the same market share the same pool of Dashers; even though only the treatment "paid the price" (discouraged from placing orders), everyone in the market reaped the same benefits (increased dasher supplies + decreased order demands 👉 shorter delivery times + fewer late orders), so we of course won't see any differences between variants. This is a common problem for experimentation in marketplaces (e.g., Uber, eBay, Airbnb), social media (e.g., Facebook), and communication software (e.g., Zoom) where the treatment often "spills over" to control units via direct interactions or shared resources, resulting in an underestimation of the treatment effect. 

If you're working at or interviewing with one of those companies, it's crucial to know how else to make causal inference than A/B testing. To learn more, I suggest that you check out [Causal Inference for The Brave and True](https://matheusfacure.github.io/python-causality-handbook/landing-page.html) and [The Effect](https://theeffectbook.net/). My post is more of a bird's-eye view of this complex topic rather than a hands-on guide.

# Rethink Alternatives

## Test differently: switchbacks

First idea: Run experiments, but use more appropriate designs. To test SOS pricing properly, DoorDash can split a shared Dasher pool either across space (testing in different markets) or across time (testing at different time of the day). In practice, they do both (check out this wonderful [post](https://doordash.news/2018/02/14/switchback-tests-and-randomized-experimentation-under-network-effects-at-doordash/) on their engineering blog).


{{< figure src="https://miro.medium.com/max/1400/1*wTkgNdjmhOeMUjfGbpqzJQ.png" width="600" caption="DoorDash tests SOS pricing using switchbacks across time-market units">}}

To begin, we select multiple markets for testing and evenly divide each day in each market into small chunks, say, half an hour. Then, we randomly assign each time-market unit to having or not having SOS pricing. In the end, we compare the average delivery time across treatment units with that across control units. By contrast, randomization units in typical A/B testing are things like deliveries or customers. 

This design is called "switchbacks", in that a market is switched back and forth between algorithms over the day. The idea sounds simple but is hard to get right:

1. **How fine should the units be?** Two hours, half an hour, or 10 minutes? Too coarse, there aren't enough randomization units; too granular, the market won't have time to respond to the change and the variance would probably be high (market trends are smoother over longer periods of time and more volatile in shorter periods). We can try a bunch of window sizes to see when we can achieve a reasonably low standard error $\frac{\sigma}{\sqrt{n}}$ ($n$: the number of time-market units).

2. **Dependence between time units in the same market.** If we know how deliveries in LA are like at 11:30 AM, we can make a good guess about 12:00 PM deliveries, even though pricing algorithms may be different in those two units. To deal with the lack of independence, we can use [multilevel models](https://stats.stackexchange.com/questions/4700/what-is-the-difference-between-fixed-effect-random-effect-and-mixed-effect-model) where the pricing strategy is the "fixed effect" (captures the effect of the variable we're interested in) and time and markets are the "random effects" (capture group-level variation). Below is a toy implementation using the `lme4` package in R:

    ```r
    library(lme4)
    lmer(delivery_time ~ has_sos + (1|market/time), data=sb_data)
    ```

    Moreover, we can use the "[Huber Sandwich Estimator](https://stats.stackexchange.com/questions/50778/sandwich-estimator-intuition)" to compute variance, which doesn't make the independence assumption like ordinary least squares (OLS).

3. **Do we weight each time-market unit by the number of deliveries?** Each time-market unit's delivery time is an average across all orders. Some units (e.g., 12 PM in San Francisco) have far more orders than others (e.g., 2 AM in Anchorage) --- as vividly explained in [The Most Dangerous Equation](https://www.americanscientist.org/article/the-most-dangerous-equation), it can be misleading to use data aggregated over different sample sizes to draw conclusions. For instance, if there was a 2-hour outlier in Anchorage at 2 AM under SOS pricing, we may wrongly think this algorithm doesn't reduce delivery time as much as it did. To mitigate this problem, we can weight each unit by the delivery count or the standard error, which DoorDash data scientists didn't do [in practice](https://doordash.news/2018/02/14/switchback-tests-and-randomized-experimentation-under-network-effects-at-doordash/).

4. **Is the change visible?** Switchbacks are good for testing *invisible* algorithmic changes (e.g., ranking, pricing) but terrible for visible UI changes. Users are bound to be confused if switched back and forth between different designs.😵‍💫

## Infer from observational data?

Sometimes all we have is observational data but still wish to make causal claims, such as whether booster shots offer protection against Omicron outside of the lab. Why not just compare infection rates between people who got boosters vs. those who didn't? Since boosters are *not* randomly assigned, there may be **selection bias**:

> **Notations**: $D_i$: treated (1) or untreated (0); $Y_i$: observed outcomes; $Y_{1i}$: potential outcome of the treated; $Y_{0i}$: potential outcome of the untreated

1. **Observed difference in average infection rate**: $E[Y_i|D_i = 1] - E[Y_i|D_i = 0]$
2. **Causal effect of getting a booster**: $E[Y_{1i}|D_i = 1] - E[Y_{0i}|D_i = 1]$ 

    👉 first term: actual outcome of people who got the booster, which we know; second term: potential outcome of the same people ("clones") had they not gotten the booster, which is a counterfactual outcome we can't observe 
3. **Selection bias** ($1-2)$: $E[Y_{0i}|D_i = 1] - E[Y_{0i}|D_i = 0]$ 

    👉 second term: outcome of people who didn't get boosters, which we also know; those who chose to get a shot may have different outcomes *even without the shot*


People with health concerns (e.g., asthma), higher social economic status, etc. may be more likely to get a booster shot. The same factors may also influence how likely they will get Omicron. If we do see a difference between those who got vs. didn't get boosters, it may be due to the shot or those "confounders" that impact both the treatment and the outcome. Selection bias and confounding plague causal inference from observational data. To think clearly about confounding, or causation in general, we can draw [directed acyclic graphs](https://en.wikipedia.org/wiki/Directed_acyclic_graph) (DAGs) for the problem at hand, using nodes to represent random variables and edges their causal connections. 

{{< figure src="https://www.dropbox.com/s/xwp4oa9ncpnys12/booster.png?raw=1" width="300" caption="Judea Pearl (2000) created directed acyclic graphs (DAGs) to depict causal relationships: Nodes represent random variables and edges connections. (Yup, data engineers, it's the same 'DAG' that you see in Airflow...)">}}



# Method 1: Statistical Control 

When life gives you a confounder, you can squeeze (weak) causation out of it using statistical control. The idea is that we hold confounders at constant values and only let the treatment vary; if the outcome changes, we can be a bit more confident that the treatment has an effect. Two common methods are regression and matching.


## 1. Regression

To "regress out" all relevant confounders, we can include them in the same model as the treatment. Take linear regression for instance: 

```r
lm(infection_rate ~ got_booster + age + has_asthema + SES + ..., data=omicron)
```

After fitting the model to the data, we can extract the **partial slope** of the treatment (`get_booster`). The slope tells us, *with all else being the same*, how much getting a booster changes one's infection rate compared to not getting one. A horizontal slope ($\beta = 0$) indicates no treatment effect and $\beta <0$ means boosters may reduce infection rates (if $\beta > 0$, then the booster backfired🦠...). 

"Garbage can regression" is dangerous. Potential mistakes come in two flavors:

1. **Not controlling for the right things**: What if we miss some confounders in our model? The short answer is that, one needs to know a domain really well to understand which factors play critical roles in which problems. If your company wants to model the housing market, for instance, you really need an economist on the team or as a consultant, or you may become the next [Zillow Offers](https://ryxcommar.com/2021/11/06/zillow-prophet-time-series-and-prices/).

2. **Controlling for wrong things**: This includes ignoring functional forms or causal structures of how variables are connected ([Griffiths & Tenenbaum, 2009](https://cocosci.princeton.edu/tom/papers/tbci.pdf)). 
    - **Functional form**: Linear regression can only control for confounders that have linear relationships with the treatment and the outcome. If you know what the functional form is for a confounder, you can transform the said confounder (e.g., quadratic 👉 $x^2$). More often than not, we don't know the form, though, so we can't assume regression can control things cleanly.
    - **Causal structure**: Confounders are variables that directly or indirectly impact the treatment and the outcome --- they are what we should control for. Things may go wrong if we control for other types ([Rohrer, 2018](https://journals.sagepub.com/doi/pdf/10.1177/2515245917745629)):.  
        - **Colliders**: If $X$ is the common effect of $Y$ and $Z$ ($Y \rightarrow X \leftarrow Z$), controlling for $X$ results in spurious correlation between $Y$ and $Z$. 
          - **Example**: Warm and competent candidates tend to be successful. A job offer is a "collider" of warmth and competence. Since all of our colleagues were once successful candidates (i.e., interview results are "controlled for"), when we look around in the office, almost everyone seems warm and competent. We may jump to the conclusion that these two traits are intrinsically linked.
        - **Mediators**: If $X$ influences $Z$ through $Y$ ($X \rightarrow Y \rightarrow Z$), controlling for $Y$ leads to the false conclusion that $X$ and $Z$ are unrelated.
          - **Example**: Family wealth impacts education, which impacts future income. When we just look at PhD students (i.e, education level is controlled for) from different SES backgrounds making roughly the same amount of $$ after graduation, we may conclude that there's no such thing as generational wealth.

Since we ~~don't always~~ rarely know which variables are relevant for a problem, let alone how they are connected, statistical control may hurt unexpectedly. 

## 2. Matching

Matching cannot solve the structural problem but can help with unknown functional forms. The general idea is that we find *similar* treated and untreated units in the observational data; assuming the two "artificial comparison groups" only differ in whether or not they receive the treatment, we can attribute their outcome difference to the treatment. Below are two matching methods, among many:

1. **Exact matching**: We can find treated and untreated units that have the same exact values in all the variables we wish to control for.

    {{< figure src="https://www.dropbox.com/s/fhsknytandz400z/exact_matching.png?raw=1" width="400" caption="Toy example from Chapter 8 in [Impact Evaluation in Practice](https://openknowledge.worldbank.org/handle/10986/25030)">}}

    - **Pros**: Simple + transparent; no extra uncertainty from predictive models
    - **Cons**: Need to match as many times as there control variables + may not find enough matches ("common support") in small datasets

2. **Propensity Score Matching (PSM)**: We can take above characteristics to predict the probability each unit may receive the treatment. This probability is called the "propensity score", which we use to match treated and untreated units.

    - **Pros**: Only match once; no need to throw away units that differ on some dimensions but have similar probability of receiving the treatment
    - **Cons**: Unreliable models lead to invalid matches; matching is still tricky
        - **[Greedy vs. optimal](https://ncss-wpengine.netdna-ssl.com/wp-content/themes/ncss/pdf/Procedures/NCSS/Data_Matching-Optimal_and_Greedy.pdf)**: To be fast, for each of the $N$ units in Group 1, we can iterate through each of the $N$ units in Group 2 to find the best match (and cross out the matched pair) 👉 **time complexity**: $O(N^2)$. However, this approach doesn't guarantee optimal results over all pairs. Alternatively, we can exhaust all one-to-one pairs to find the combination that minimizes total within-pair differences 👉 this optimal approach is slow; **time complexity**: $O(N!)$
        - **Constraints**: If we force all units to be matched, we may end will dissimilar pairs. We can set a threshold ("caliper radius") on the largest propensity score difference allowed for matched pairs. Another common constraint is whether we only consider one-to-one matching or if we allow one-to-many or even many-to-many matching.


To see PSM in action, check out this [great talk](https://www.youtube.com/watch?v=gaUgW7NWai8) by HelloFresh data scientist Michael Johns. The team launched an ad campaign and wanted to know if users who clicked on the ads were more likely to purchase from HelloFresh. 

{{< figure src="https://www.dropbox.com/s/iexwhzrjzzxtokq/propensity.png?raw=1" width="400" caption="It's hard to tell if ad clicking causes HelloFresh purchases or if clicks and purchases are both driven by a third variable: Interest in cooking.">}}

Selection bias comes to haunt us: Ads are personalized, so those who are shown ads from this campaign likely differ from those who aren't; if we do observe a difference between clickers vs. non-slickers, it's hard to attribute it to the ads or to individual differences. Chances are, those interested in cooking are both inclined to click on the ads and buy food from HelloFresh.

Instead, we can use user demographics and behaviors to predict the probability that 1) they are shown an ad and 2) will click on it and match clickers vs. non-clickers based on this propensity ($p(\mathrm{expose}) \times p(\mathrm{click})$). If we see a difference in their conversation rates, we can be more confident that this ad campaign had an effect.


# Method 2: Counterfactuals

To use regression or matching, we need many treated vs. untreated units whose outcomes we can compare. It's hard to measure the causal impact of an election, a policy change, or a pandemic using statistical control. These events are special because they 1) happen *infrequently* and 2) impact *large units* such as an entire city, state, or country. As a result of *both*, treated units are few and far between. 

Why do events have to meet both criteria? We can think of some counterexamples:
- **Rare but units are small**: A particular A/B test only happens that once, but because the units are small (e.g., users or even pageviews), one experiment still results in many treated vs. untreated units.
- **Large units but not rare**: Weekends impact the entire planet, not chosen individuals (I digress: PhD students be like ["What weekends?"](https://phdcomics.com/comics.php?f=1924)), but because they happen every week, we also have many treated vs. untreated units.

<!-- > "The outside world often has a much larger effect on metrics than product changes do." Jan Overgoor, [*Experiments at Airbnb*](https://medium.com/airbnb-engineering/experiments-at-airbnb-e2db3abf39e7)
 -->
[Comparative case studies](https://en.wikipedia.org/wiki/Case_study) are traditionally used in those cases, like comparing how "similar" countries went on different trajectories because a major event (e.g., a war) happened in some but not others. The underlying philosophy is the [counterfactual view of causation](https://plato.stanford.edu/entries/causation-counterfactual/): By saying "$X$ (the event) causes $Y$" (an outcome)", we mean that "had $X$  not happened, $Y$ would not have occurred". Methods such as difference-in-differences (DD) and synthetic control put this idea into action --- they both use untreated units *similar* (I avoided the definition again, which I'll discuss later) to a treated unit to create a "parallel" universe of the treated unit where the treatment never occurred. If the treatment had no effect, the counterfactual outcome in the parallel universe would be the same as the actual outcome in the real world; if the two diverge, it provides evidence the treatment has a causal impact.

## 3. Difference-in-Differences (DD)

Let's look at some classic examples in tech. In the old days, Meta users could only like a post but not express other emotions such as anger or sadness. If someone posted about a loved one passing away, it seemed inappropriate to like it, yet many might not want to leave a comment (e.g., too effortful, not close enough). Meta data scientists hypothesized that, if more reactions were allowed, people would be more willing to engage with posts. Because of the "spillover effect" mentioned before, a regular A/B test wasn't a good idea (e.g., if a treatment user reacted to a post by control user, the latter would not be able to see it and engage back). 

To avoid bad user experience and contaminated results, Meta did a geo experiment ([talk](https://developers.facebook.com/videos/f8-2017/how-we-shipped-reactions/)). Let's they rolled out this new "Reactions" feature to Canada (treated) but not to the US (untreated) and measured the number of posts reacted to (including likes) per user before and after the rollout. Below are results I made up.

{{< figure src="https://www.dropbox.com/s/qt9aoe34zc60fv7/diff-in-diff.png?raw=1" width="400" caption="\# of posts reacted to per user before and after launching Reactions">}}

What's the treatment effect of Reactions? To answer this question, we need to assume that the outcome will *trend the same way* in different countries if the treatment had no effect. Under this "common trend" assumption, the (dashed) blue line in Canada would still be parallel to the green line in the US after the rollout. The fact that the solid blue line showing the actual outcome is higher than the dashed blue line showing the counterfactual outcome means "Reactions" drove up the outcome metric. 

Another way to understand the treatment effect is to look at the table below:

<div align="center">

|        | Canada (treated) | US (untreated) | difference |
|:------:|:----------------:|:--------------:|:----------:|
| before | 3.5              | 4.2            | -0.7       |
| after  | 5.0              | 4.9            | +0.1       |
| change | +1.5             | +0.7           | +0.8       |

</div>

- **Same time effect across units**: Under the common trend assumption, if the # of reactions per user increased by 0.7 in the US, it should also increase by 0.7 in Canada. However, it actually increased by 1.5 in Canada  — the 0.8 difference between our expectation and the reality is the treatment effect of Reactions.  
- **Same regional difference across time**: Again assuming the common trend, if an average US user reacted to 0.7 more posts compared to an average Canadian user before the launch, we should see the same difference afterwards. However, after the launch, an average US user reacted to 0.1 fewer posts than an average Canadian user. The 0.8 between these two differences is the treatment effect. 

If the common trend assumption breaks, the analysis above is invalid. One way to check is to use the unit type ($D_{treatment} = 1$: Canada; $D_{treatment} = 0$: US), covariates $X$ relevant to the outcome, and their [interaction](https://en.wikipedia.org/wiki/Interaction_(statistics)) (an interaction means the same covariates impact different units differently) to predict the outcome: 

$$Y = \beta_0 + \beta_1 D_{treatment} + \beta_2 X + \beta_3 D_{treatment} X + \epsilon.$$

If the interaction $\beta_3$ is not significant, maybe we can buy this assumption.  

But what if the trends are already different before the launch? In the research design phase, we can use *matching* to select untreated units that are similar (e.g., population, culture, user demographics and behavioral patterns, etc.) to the treated unit. The hope is that matched units are more likely to share a common trend. If you still don't want to commit to this strong assumption, then you can use methods such as synthetic control or CausalImpact to explicitly model trends over time.   

## 4. Synthetic Control

For Uber, in markets like India or South America where credit cards are not the norm, riders often pay drivers [by cash](https://techcrunch.com/2015/05/11/uber-is-testing-cash-payments-in-hyderabad-india/). Uber charges drivers \~25% of their earnings as the "service fee" so drivers receiving cash payments need to wire this fee to Uber. This may create hassles for the drivers and will take them off the road for a bit. Alternatively, drivers may prefer cash over credit card payments. It's hard to know. Uber was considering giving drivers a heads up about the trip type: For a cash trip, the app will show "CASH"; otherwise, the app doesn't say anything. One of the success metrics could be the number of cash trips per driver. 

Uber faced the same problem as DoorDash: In the same market, if treatment drivers accept more or fewer cash strops, the control will have fewer or more left to accept. Can Uber use switchbacks? That would be problematic: If a driver was in treatment and received heads up about cash trips, after being switched to control, they will interpret not seeing "CASH" as a non-cash trip rather than not knowing the type. Instead, Uber used synthetic control ([talk](https://youtu.be/j5DoJV5S2Ao)). Below is a rough sketch:

1. **Choose a donor pool**: We can roll out this feature only to, say, São Paulo (treated), and use matching to find $J$ untreated cities with similar characteristics (credit card prevalence, population, etc.), such as Rio de Janeiro and Lima. The untreated units we selected are called the "donor pool". Later they will help us create a "synthetic São Paulo" with no treatment.
2. **Find the weight of each unit**: Not all untreated units are equally important. We can use the pre-treatment data to find the optimal weights $\mathbf{W}$. To do so --- 
    - We first choose features $\mathbf{X}$ (e.g., weather, events, holiday) that can be used to predict the outcome and find their importances $\mathbf{V}$ (generally, the higher the predictive power, the higher the importance)
    - Then we build a model to predict each unit's pre-treatment outcome and optimize the weights to minimize the difference between the treated unit and the donor pool average, $\|\mathbf{X_1} - \mathbf{X_0}\mathbf{W}\| =\sqrt{\sum_{h=1}^{k}v_h(X_{h1}-\sum_{j=2}^{j+1}w_jX_{hj})^2}$ 👉 weights are non-negative ($w_j \geq 0$) and add up to 1 ($\sum_{j=2}^{J+1}w_j = 1$)
3. **Create a synthetic control**: After finding the weights (e.g., Rio: .6; Lima: .4), we can used the weighted average of the donor pool to project the post-treatment outcome in the treated unit if  the treatment didn't occur, $\sum_{j=2}^{J+1} w_j Y_{jt}$ 👉 *drum roll*... this weighted average is called the "synthetic control" and represents our best guess for how things could have been in untreated São Paulo. At any point $t$ after the launch, the difference between the actual and the synthetic São Paulo is the treatment effect at $t$, $Y_{1t}^{treated} - \sum_{j=2}^{J+1} w_j Y_{jt}$

    {{< figure src="https://www.dropbox.com/s/tebslmo8g1agtlm/synthetic_controk.png?raw=1" width="600" caption="Uber used the synthetic control method to measure how much giving drivers heads up about cash trips impacted # of cash trips per driver">}}

4. **Significance test**: As we can see, notifying drivers of cash trips seemed to have reduced the number of cash trips per driver in São Paulo. Is the decrease *statistically significant*? Synthetic control practitioners often use permutation tests to answer this question. We can "pretend" each untreated unit is treated and go through steps #1-3 to create a synthetic control and find the "treatment effect". Since the treatment didn't happen there, the effect should be small. Say we have 50 untreated units and the treated unit's effect ranks second, we can be quite confident that the treatment effect is significant. 

Compared to previous methods, synthetic control is quite complex and many things can go wrong (check out [Abadie, 2021](https://economics.mit.edu/files/17847) for more details):
1. **Donor pool**: If untreated units are dissimilar to the treated unit, we cannot create good counterfactuals (most weights are 0). Again, matching comes handy.
2. **Overfitting**: In a short pre-treatment period with too many untreated units, we're bound to "overfit" --- the synthetic control may mimic the treated unit too exactly rather than giving us a robust projection of the possible trend. 
3. **Data peaking**: When choosing the donor pool and optimizing weights, we may peek at the post-treatment data and end up finding false positive treatment effects. 
4. **Question types**: If you wish to answer questions about individual users, synthetic control is not the right method. And if you make a small UI change, synthetic control is also not great for detecting such weak signals.

Regarding #3, a newer method [CausalImpact](https://google.github.io/CausalImpact/CausalImpact.html) developed by Google can make use of data in both periods. You can see it in action in another HelloFresh [talk](https://youtu.be/KEhJNM5K73A).

# Method 3: Natural Experiments

In colloquial language, "natural experiments" often refer to once-in-a-life events, such as the Fall of the Berlin Wall. Here I'm referring to cases where "the universe randomizes something for us" ([Körding, 2021](https://youtu.be/ahyp-zox3Ks)). Below are two such examples.

## 5. Regression Discontinuity Design (RDD)

For DoorDash, late orders are the worst --- "hangery" customers may order less or churn. To make it slightly better, DoorDash automatically issues a refund to orders $\geq$ 30 minutes late. So *how does this policy impact customer lifetime values (LTVs)*? Apparently, we can't randomly assign customers into getting or not getting a refund. It's also wrong to compare LTV between those who received or didn't receive refunds, since the average lateness is different between the two groups. 

We can tweak this idea a little bit to make it work: Instead of comparing all customers who didn't get a refund vs. all customers who did, we can compare those around 30-minute cutoff. For instance, A might get a refund for a 30.1-minute order whereas B gets nothing for a 29.9-minute later order. One should not feel angrier than the other just because of the 0.2-minute difference in lateness, so if we do observe a difference in LTV between A and B, then it may well be attributed to the refund. This method is called the regression discontinuity design (RDD). 


Say DoorDash plotted the data and found a upward "jump" after the cutoff, this suggests that the refund reduced late orders' damage to LTV to some degree. 

{{< figure src="https://www.dropbox.com/s/vz60f040euu0465/rdd.png?raw=1" width="400" caption="DoorDash uses regression discontinuity design (RDD) to measure the impact of auto-refunding on customer LTVs when orders arrive late.">}}

To reach the above conclusion, we made two assumptions: 

- **Continuity**: Without any cause, the outcome should change smoothly with the "running variable" (e.g., order lateness);
- **As good as random**: Treatment assignment of "near-winners" (A) and "near-losers" (B) is as good as random 👉 this is the "natural experiment" here.

Both assumptions are reasonable in the refund case, so discontinuity can be attributed to the treatment. In other cases, we may see a naturally "jumpy" relationship around the cutoff. For instance, if GPA determines whether students can get a merit-based scholarship (eligible if GPA $\geq$ 80\%), those just below the cutoff (e.g., 79.9\%) may beg the teacher for a "mercy pass", but not those just above (80.1\%). Students who still nearly lose may have even lower grades to begin with, or didn't argue for grades, making them less comparable with the near-winners. 

This example sounds straightforward. In reality, there can be complications:
1. The cutoff isn't always deterministic, making the discontinuity less clean:
    - **Sharp (deterministic)**: Orders $\geq$ 30 minutes late definitely receive a refund and those $<$ 30 minutes late definitely don't 
    - **Fuzzy (probabilistic)**: The admission office has a suggested SAT cutoff, but students with lower scores might still get in through special programs
2. The jump may not be dramatic 👉 we can use the running variable and the treatment assignment ($D = 1$: treatment; $D = 0$: control) to predict the outcome, $y = f(X) + \beta D + \epsilon$, and check if $\beta$ is statistically significant;
3. The relationship between the running variable and the outcome may not be linear, in which case it's harder to spot the jump. 


## 6. Instrumental Variables (IVs)

I don't pretend I understand the most confusing and ingenious method of the six --- instrumental variables. The best I can do is to walk you through an iconic example ([Angrist & Krueger, 1990](https://www.nber.org/papers/w3572)): *How does years of school affect future income?*

{{< figure src="https://www.dropbox.com/s/caq8zp4b3lupbg9/iv.png?raw=1" width="400">}}

Many confounders can impact both schooling and income, making causal inference difficult. However, the US education system created a natural experiment for us: Children are required to enter school the calendar year they are 6 but can leave as soon as they turn 16. Imagine two kids Josh and Guido:

- **Birth seasons**: Josh was born on Jan. 1, 2006; Guido was born on Dec. 31, 2006 
- **Entering school**: Both would be required to go to school on Sept. 1, 2012
- **Dropping out**: However, Josh will be free to leave on Jan. 1, 2022 whereas Guido has to stay until Dec. 31st, 2022

Just because of a later birth season, Guido is required to remain in school for almost a year longer than Josh. Presumably, the birth season has nothing to do with any confounders such as family wealth or intelligence, but it has a causal effect on the treatment, years of schooling, through which it may impact income.  

{{< figure src="https://www.dropbox.com/s/rcnrj0rk0mzdljh/instruments.png?raw=1" width="400">}}


We can use the birth season as an "instrument" to estimate the causal effect of schooling on income. First, we regress years of schooling on birth season, $\hat{X} = \alpha IV$. Then, we regress income on an unbiased estimate of years of schooling obtained in the first step, $\hat{Y} = \beta \hat{X}$. This method is called "two-stage least-squares". The coefficient $\beta$ we get from the second step is the causal effect of schooling on income. IV estimation allows us to make causal inference in the presence of confounders (IVs), rather than regressing them out and wreaking havoc unbeknownst to ourselves.

The bar for what we consider as good instruments is high, which must be 

- randomly assigned to units, 
- causally impacting the treatment,
- correlated with the outcome,
- but uncorrelated with any confounders. 

Even with a ton of domain knowledge, it's still hard to find great instruments that fit the problem, which might have limited the industry use cases of this method. Weak instruments are not as useful. When we do find one, though, it really pays.

# Resources

## Books
1. [Mostly Harmless Econometrics]( https://www.amazon.com/dp/0691120358/ref=cm_sw_em_r_mt_dp_5QENZH0GT2AE3HPZDJPD) 
    - 👉 written by the OG of causal inference in econometrics, Joshua Angrist (which I learned during interviews last year)
2. [Causality: Models, Reasoning and Inference](https://www.amazon.com/dp/052189560X/ref=cm_sw_r_tw_dp_Q6HW7EVMW5NF88ZF69EA) 
    - 👉 written by the OG of causal inference in computer science, Judea Pearl (which is the tradition I follow)
3. [The Book of Why: The New Science of Cause and Effect](https://www.amazon.com/Book-Why-Science-Cause-Effect/dp/046509760X) 
    - 👉 popular science book by Judea Pearl; super fun read 
4. [Causal Inference for The Brave and True](https://matheusfacure.github.io/python-causality-handbook/landing-page.html) 
    - 👉 the author summarized common causal inference methods in econometrics and has Python implementations for each 
5. [Causal Inference: The Mixtape](https://mixtape.scunning.com/) 
    - 👉 similar to #4; also aims to cover all grounds and provides implementations (R + Stata)
6. [The Effect: An Introduction to Research Design and Causality](https://theeffectbook.net/) 
    - 👉 similar to #4 & #5, but more in depth


## Papers

7. Abadie, A. (2021). Using synthetic controls: Feasibility, data requirements, and methodological aspects. *Journal of Economic Literature, 59*(2), 391-425. ([PDF](https://inferenceproject.yale.edu/sites/default/files/jel.20191450.pdf)) 
    -  👉 review paper by creator of synthetic control; may not have all the details (e.g., how to select predictors and find feature importances...)
8. Liu, T., Ungar, L., & Kording, K. (2021). Quantifying causality in data science with quasi-experiments. *Nature Computational Science, 1*(1), 24-32. ([PDF](https://www.nature.com/articles/s43588-020-00005-8.pdf)) 
    - 👉 short review paper covering IV, RDD, and DD; has nice examples in neuroscience + industry + social issues
9. Rohrer, J. M. (2018). Thinking clearly about correlations and causation: Graphical causal models for observational data. *Advances in Methods and Practices in Psychological Science, 1*(1), 27-42. ([PDF](https://psyarxiv.com/t3qub/download?format=pdf)) 
    - 👉 my regression section is based on this lovely paper; really helps you think more clearly about causal relationships 

## Talks 
10. [When Do We Actually Need Causal Inference?](https://youtu.be/2dv7NrYExzo) by Sean Taylor
11. [Causality in Neuroscience and Beyond](https://youtu.be/ahyp-zox3Ks) by Konrad Körding
12. [Science Before Statistics: Causal Inference](https://youtu.be/KNPYUVmY3NM) by Richard McElreath 
13. [Inferring the Effect of an Event Using CausalImpact](https://www.youtube.com/watch?v=GTgZfCltMm8) by Kay Brodersen

## Blogs
14. [Lessons Learned on Experimentation @DoorDash](https://www.ai-expo.net/northamerica/wp-content/uploads/2018/11/1500-Jessica-Lachs-DoorDash-DATA-STRAT-V1.pdf)
15. [Switchback Tests and Randomized Experimentation Under Network Effects at DoorDash](https://doordash.news/2018/02/14/switchback-tests-and-randomized-experimentation-under-network-effects-at-doordash/)
16. [Under the Hood of Uber's Experimentation Platform](https://eng.uber.com/xp/)
17. Experimentation in a Ridesharing Marketplace by Lyft ([Part 1](https://eng.lyft.com/experimentation-in-a-ridesharing-marketplace-b39db027a66e#.djox1933t), [Part 2](https://eng.lyft.com/https-medium-com-adamgreenhall-simulating-a-ridesharing-marketplace-36007a8a31f2#.g9b34i3gm), [Part 3](https://eng.lyft.com/experimentation-in-a-ridesharing-marketplace-f75a9c4fcf01))
