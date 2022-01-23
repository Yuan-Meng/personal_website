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


> "Causation […] is the cement of the universe." — David Hume, *Abstract*

# Why Ask Why?

## We can't but think about causation

As a cognitive scientist, I study common sense causal reasoning. Folks say *"correlation is not causation"* like a broken record, yet we can't help but think "leaves fall *because* the wind blows", not "leaves *happen to* fall *after* the wind blows". If you're intrigued, I highly recommend [this talk](https://youtu.be/q0HLci67Tr8) by Tobias Gerstenberg at Standford.
 
{{< figure src="https://www.dropbox.com/s/gfiw187r502jb1l/1t1_redgreen_2sec.gif?raw=1" width="350" caption="[Kominsky et al. (2017)](http://www.jfkominsky.com/demos.html) showed that even 12-month-olds 'think' the red block makes the green one move if the motions abide by Newton's Third Law.">}}


But my priors tell me you're here for the data science. Just as causation is a fundamental component of human cognition, it is also the backbone of data science.

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
    sos <- lmer(delivery_time ~ has_sos + (1|market/time), data=sb_data)
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

---

**CONTENT BELOW UNDER UPDATING**

I'm rewriting this post for more clarity and accuracy. Updated version coming soon...

---

# Method 2: Counterfactuals

## 3. Difference in Differences (DiD)
## 4. Synthetic Control
## 5. CausalImpact

# Method 3: Natural Experiments

## 6. Regression Discontinuity Design (RDD)
## 7. Instrumental Variables (IVs)


<!-- ~~Facebook~~ Meta famously did a country test before [shipping "Reactions"](https://developers.facebook.com/videos/f8-2017/how-we-shipped-reactions/). It used to be the case that users could only like a post but not express other emotions such as anger or sadness. If someone posted about a loved one passing away, it seemed inappropriate to like it, yet many might not want to leave a comment (e.g., too effortful, not close enough). Meta data scientists hypothesized that, if a wider range of reactions were allowed, people would be more willing to engage with posts. They could run a regular A/B test, randomly assigning users to having or not having Reactions. However, if users in different variants are in the same friend circle, you risk creating bad user experiences: Say a treatment user reacted to a post by a control user, the latter would not be able to see it and engage back. 

To alleviate this worry, Meta tested Reactions in different countries, assuming users didn't interact with folks from other countries. In this example, country A (green) had Reactions while country B (red) didn't. The $y-$axis tracks the number of posts reacted (including likes) to per user, which Meta aimed to drive up.

{{< figure src="https://www.dropbox.com/s/s9x78epex3zw1lk/reactions.png?raw=1" width="500" caption="The # of posts reacted to per user before and after launching Reactions">}}


- **Assumptions**: 1) Differences between countries are the same at different points in time and 2) the time effect is the same across different countries. 

- **Analysis**: Under those assumptions, if Reactions had no effect at all, the green line would be parallel to the red line after feature launching. In my badly drawn illustration, **the difference between the actual green line and the dotted green line** (the counterfactual under the null hypothesis) **is the treatment effect**. This method is called **"difference in differences (DiD)"**. 

- **Formalism**: $y = \beta_0 + \beta_1 D_{post} + \beta_2 D_{treatment} + \beta_3 D_{post} D_{treatment} + \beta_4 X + \epsilon$
  - **Variables**: $y$: the value of the success metric (e.g., # of posts reacted to per user); $D_{post}$: whether an observation came from the pre-launching or the post-launching period (0: pre, 1: post); $D_{treatment}$: whether an observation was in treatment or control (1: treatment, 0: control); $X$: covariates (e.g., country/user demographics)
  - **Interpretation**: The interaction $\beta_3$ is the difference in differences

DiD is an old method that's fast but crude: Both of the assumptions above can be easily violated (e.g., between-country differences may change over time and time effects may differ by country), resulting in invalid conclusions. As we'll see in a minute, more sophisticated methods such as synthetic control and CausalImpact are developed to make causal inference without these rigid assumptions.

# Synthetic Control (Synth)

Sometimes we can't but roll out a feature to all users at once (e.g., due to urgency or the worry that treatment users might leak confidential information to the Internet) but still wish to know what would be like without this new feature.

A simple thing to do is compare metrics before and after the launch. However, changes may result from what's happening in the world besides the feature launch. 

> "The outside world often has a much larger effect on metrics than product changes do." Jan Overgoor, [*Experiments at Airbnb*](https://medium.com/airbnb-engineering/experiments-at-airbnb-e2db3abf39e7)


Synthetic control comes to the rescue: We can build a model that learns the relationship between certain predictors and the target metric in the pre-launching period and then use the learned relationship to forecast how the metric would look later if the feature wasn't launched. This method is called "*synthetic* control" in that there never was an actual control: The model's prediction simulates a "parallel universe" without the feature launch and **the difference between the synthetic control and what actually happened after launching is the treatment effect**.

Uber data scientists used synthetic control to measure the effect of giving drivers heads up about cash trips. In markets like India where not a lot of people carry credit cards, riders often pay drivers in cash. Cash trips may be inconvenient to the drivers as they need to wire commission fees to Uber later; however, some drivers may prefer to receive cash. Uber wanted to test whether showing drivers whether a trip was a cash trip in advance might affect how many cash trips they took. 

{{< youtube j5DoJV5S2Ao >}}

Uber faced the same problem as DoorDash did: A market shares the same pool of drivers and riders; if the treatment takes more or fewer cash trips, the control will have fewer or more cash trips to take. Can Uber use switchbacks like DoorDash? Nice guess, but that would be inappropriate: Imagine, a driver was in treatment and received heads up about cash trips; when they are switched to control, they'll naturally think a trip without a heads up is non-cash rather than type unknown. 

{{< figure src="https://www.dropbox.com/s/tebslmo8g1agtlm/synthetic_controk.png?raw=1" width="600" caption="Uber used synthetic control to test heads up to drivers about cash trips (to report to stakeholders, you can average the differences over time)">}}

Instead, Uber data scientists trained a time-series model to predict the number of cash trips had the feature not been rolled out in a city ("synthetic city") and compared the prediction with the actual number of cash trips after the rollout ("treatment city"). The difference between the two shows that telling drivers about the trip type encouraged them to take more cash trips.

To successfully implement synthetic control, you gotta be a good time-series modeler and are *not* tempted to tweak your model until you find treatment effects.

# DiD + Synth 👉 CausalImpact

[CausalImpact](https://google.github.io/CausalImpact/CausalImpact.html) combines ideas from DiD and synth and is the best of both worlds:

- **Unlike synth**, there is an actual control group not exposed to the new feature 👉 this means CausalImpact can use control data from both pre- and post-launching periods whereas synth only makes use of the pre-launching data; hopefully more data means higher generalizability and robustness
- **Unlike DiD**, we don't need to assume the same regional differences across time or the same time effects across regions 👉 instead, we can build a Bayesian structural time-series model using the control data to predict a "synthetic treatment" trend and compare it with the actual treatment

Google open-sourced the `CausalImpact` R package. Below is a toy example:

```r
library(CausalImpact)
impact <- CausalImpact(data, pre.period, post.period)
plot(impact)
```
{{< figure src="https://www.dropbox.com/s/cwtreolbp7h9477/causalimpact.png?raw=1" width="550" caption="Example results: The top panel shows the actual treatment (solid line) and the synthetic treatment predicted by the model (dashed line) and the middle panel shows the difference between the two before and after launching">}}

HelloFresh used CausalImpact to measure the impact of a YouTube ad campaign across geographic regions. The treatment saw the ads and the control didn't. HelloFresh data scientists used control data to predict what the treatment conversation rate would look like without the campaign and compared it with the actual observation — as mentioned, the impact of this ad campaign lies in the difference.

{{< youtube KEhJNM5K73A >}}


# Regression Discontinuity Design (RDD)

Consider another DoorDash example. Late orders anger hungry customers; the later, the worse. Frustrated customers may order less in the future or even churn, resulting in lower lifetime values (LTVs). To make things slightly better, DoorDash automatically issues a refund to orders $\geq$ 30 minutes late. How does it impact LTV? 

It's unfair to randomly assign customers into getting or not getting a refund, but it's also wrong to compare LTV between those who received or didn't receive refunds, since average lateness is likely higher in the former than in the latter.

To answer this question, we can use a regression discontinuity design (RDD), using order lateness (minutes late) and refund status (received or not) to predict LTV. As we can see in the figure below, the upward "jump" right after the 30-minute cutoff shows auto-refunding saved DoorDash's customer LTV, albeit not a lot.

{{< figure src="https://www.dropbox.com/s/vz60f040euu0465/rdd.png?raw=1" width="550" caption="DoorDash uses regression discontinuity design (RDD) to measure the impact of auto-refunding on customer LTVs when orders arrive late">}}

- **Assumption**: Trajectories of "near-winners" (orders 31 minutes late) and "near-losers" (orders 29 minutes late) would have been the same without the treatment (refunding), so discontinuity can be attributed to treatment effects. 
- **Formalism**: $y = f(X) + \beta D + \epsilon$ 👉 $y$: the outcome variable (e.g., LTV); $X$: the "running variable" that has continuous effects on the outcome (e.g., order lateness); $D$: whether treatment was assigned (0: not refunded, 1: refunded)
- **Types**: Depending on whether the cutoff is deterministic, RDD has [two types](https://scholar.princeton.edu/sites/default/files/jmummolo/files/rdd_jm.pdf)
  - **Sharp (deterministic)**: Orders $\geq$ 30 minutes late definitely receive a refund and those $<$ 30 minutes late definitely don't 
  - **Fuzzy (probabilistic)**: The admission office has a suggested SAT cutoff, but students with lower scores might still get in through special programs


If the outcome is naturally "jumpy" around the cutoff (people suddenly get hungry 30 minutes after the ETA), you may wrongly attribute discontinuity to treatment.

# "Regress It Out"

Continuing with the regression idea, we can statistically control for **confounders** (a fast and loose definition: factors that also affect the outcome) by regressing them out. Say DoorDash wanna know if paying extra money to Dashers during peak hours would increase how many hours they work; we can include **potential confounders that may lead to peak hours** in the same model, **such as bad weather or holidays**, and examine the partial slope of incentive with "all else being held the same". If it's much steeper than a horizontal line, we may claim that incentive stipulates supply.

{{< figure src="https://doordash.engineering/wp-content/uploads/2021/06/image3-1.png" width="550" caption="The relationship between Dasher incentive and expected Dasher hours during peak hours could be confounded by holidays, meal time, bad weather, etc. that can lead to peak hours (see [DoorDash engineering blog](https://doordash.engineering/2021/06/29/managing-supply-and-demand-balance-through-machine-learning/))">}}

```r
incentive <- lm(dasher_hours ~ incentive + is_holiday + is_bad_weather, data=data)
```

While simple and useful, this method is not always appropriate ([Rohrer, 2018](https://journals.sagepub.com/doi/pdf/10.1177/2515245917745629)):

- **Colliders**: If $X$ is the common effect of $Y$ and $Z$ ($Y \rightarrow X \leftarrow Z$), controlling for $X$ would result in spurious correlation between $Y$ and $Z$. 
  - **Example**: Warm and competent candidates tend to be successful. In other words, a job offer is the common effect of warmth and competence. Since all of our colleagues were once successful candidates (i.e., interview results are "controlled for"), when we look around in the office, almost everyone seems warm and competent. If not careful, we may jump to the conclusion that these two traits are intrinsically linked.
- **Mediators**: If $X$ influences $Z$ through $Y$ ($X \rightarrow Y \rightarrow Z$), controlling for $Y$ leads to the false conclusion that $X$ and $Z$ have no relationship at all. 
  - **Example**: Family wealth impacts education and education impacts future income. However, when we see PhD students (i.e, the education level is controlled for) from different socioeconomic backgrounds making roughly the same amount of $$ after graduation, we may falsely conclude that there's no such thing as generational wealth.

{{< figure src="https://www.dropbox.com/s/d9yg8h1mizd94uo/dag.png?raw=1" width="350" caption="It's desirable to control for confounders but disastrous to control for colliders (Liu et al., 2021) or mediators (Rohrer, 2018)">}}  

Since we don't always know how variables are related to one another (which we can represent using Bayesian networks), statistical control may hurt unexpectedly. 

# Instrumental Variables (IV)

Last but not least, let's revisit the education example: How does the amount of schooling affect future income? To answer this question, we need to notice something special about schooling: US children are required to enter school the calendar year they are 6 but can leave school as soon as they turn 16. So this would mean that children born earlier in the year (e.g., January) are required to stay in school almost a year longer than those born later the same year (e.g., December). 

To use econometric jargon, years of schooling is the treatment, future income the outcome, and **birth season the instrument variable (IV)**, which can only affect the outcome via the treatment. We can regress the treatment on the IV ($\hat{X} = \alpha IV$) and then regress the outcome on the treatment estimation from the first step ($\hat{Y} = \beta \hat{X}$). IV estimation allows causal inference in the presence of confounders (IVs), without us having to regress them out and thereby wreak havoc unbeknownst to ourselves.

{{< figure src="https://www.dropbox.com/s/sid3xuo4nrlshul/iv.png?raw=1" width="400" caption="Instrumental variable (IV) estimation allows causal inference without statistical control of confounders (Liu et al., 2021)">}}  

---
# Summary 

As a cognitive scientist, I'm writing a dissertation on how people think about causality intuitively. As a data scientist interested in causal inference, it's cool to see how human intuitions and formal methods converge: For both, the heart of causality is **counterfactuals** — worlds that could have been but never came to be, as well as **intervention** — what we can do to make a difference.

Not that long ago, causal inference was rather niche in data science; I'm happy to see it quickly gaining popularity in recently years. To summarize what I wrote:

- **Counterfactuals and intervention**: The common philosophical assumption behind DiD, synth, CausalImpact, and RDD is that if we don't do anything (e.g., launching a new feature), nothing will happen; since something did happen, then what we did had an impact. This falls under the [interventionist](https://plato.stanford.edu/entries/causation-mani/#Inte) view of causation: We know A causes B when *iif* doing A makes B happen. These methods differ in how they construct the counterfactual world without the treatment.

- **Confounders in the wild**: Other than what we do, lots of things in the world can make a difference to the outcome we care about. PSM and regression both hold confounders constant to see how much the treatment still impacts the outcome. We need to be extra careful how controlling for wrong variables (e.g., common effects and mediators) can lead to wrong conclusions. By contrast, IV can estimate the treatment effect even in the face of confounders but one has to be extra clever to think of useful instrumental variables.
 -->

<!-- # Resources

1. [Lessons Learned on Experimentation @DoorDash](https://www.ai-expo.net/northamerica/wp-content/uploads/2018/11/1500-Jessica-Lachs-DoorDash-DATA-STRAT-V1.pdf)
2. [Switchback Tests and Randomized Experimentation Under Network Effects at DoorDash](https://doordash.news/2018/02/14/switchback-tests-and-randomized-experimentation-under-network-effects-at-doordash/)
3. [Under the Hood of Uber's Experimentation Platform](https://eng.uber.com/xp/)
4. Experimentation in a Ridesharing Marketplace by Lyft ([Part 1](https://eng.lyft.com/experimentation-in-a-ridesharing-marketplace-b39db027a66e#.djox1933t), [Part 2](https://eng.lyft.com/https-medium-com-adamgreenhall-simulating-a-ridesharing-marketplace-36007a8a31f2#.g9b34i3gm), [Part 3](https://eng.lyft.com/experimentation-in-a-ridesharing-marketplace-f75a9c4fcf01))
5. [Causal Inference Using Synthetic Control: The Ultimate Guide](https://towardsdatascience.com/causal-inference-using-synthetic-control-the-ultimate-guide-a622ad5cf827)
6. [Inferring the Effect of an Event Using CausalImpact](https://www.youtube.com/watch?v=GTgZfCltMm8)
7. [The Book of Why: The New Science of Cause and Effect ](https://www.amazon.com/Book-Why-Science-Cause-Effect/dp/046509760X)
8. [Causal Inference for The Brave and True](https://matheusfacure.github.io/python-causality-handbook/landing-page.html)
9. [Python and the Holy Grail of Causal Inference](https://youtu.be/HPC42U9xtQY)
10. Liu, T., Ungar, L., & Kording, K. (2021). Quantifying causality in data science with quasi-experiments. *Nature Computational Science*, 1(1), 24-32. ([PDF](https://www.nature.com/articles/s43588-020-00005-8.pdf))
 -->
---

<!-- > [Causal Data Science Meeting 2021](https://www.causalscience.org/) takes place between November 15 and 16!
 -->
<!-- {{< figure src="https://www.dropbox.com/s/1dlnw8sr12ifr0m/pearl.png?raw=1" width="500">}} -->