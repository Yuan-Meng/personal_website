---
title: Quasi-Experiments for Causal Inference 
date: 2021-11-12
math: true
tags:
    - causal inference
    - quasi-experiment
categories:
- tutorials
keywords:
    - causal inference, quasi-experiment
include_toc: true

---

---
**NOTE**

- This post contains technical errors regarding synthetic control. The writing could've also been clearer. I plan on updating this article in late January. 

- None of the examples used are from actual DS interviews --- they are adapted from various company engineering blogs and PyData talks that I saw.

---


> "Causation [...] is the cement of the universe." — David Hume, *Abstract*


Data scientists help the team make good decisions and good decisions rely on causation. [Prof. Alison Gopnik](http://alisongopnik.com/) always has the nicest example: While yellow fingers and smoking are both correlated with lung cancer, washing hands doesn't prevent cancer. 

Randomized controlled experiments, or A/B tests, are the gold standard for establishing causality. Unfortunately, it's not always possible, feasible, or ethical to randomly assign people into different variants: In the example above, you can't in good conscience force someone to smoke, nor can you easily have them quit smoking. 


Worse still, even if you've run what seems like a well-controlled experiment, differences may arise from factors other than your treatment. For instance, say DoorDash A/B tested "SOS pricing" (treatment customers paid more for each order during peak hours) but *didn't* observe significant drops in delivery times in treatment vs. control. Is SOS pricing useless? Not necessarily — A given market shares the same pool of dashers and customer; only the treatment "paid the price" but the control reaped the benefits for free (SOS pricing 👉 increased dasher supplies + decreased order demands 👉 shorter delivery times + fewer late orders), making the observed difference quite likely smaller than the true difference.

Where randomized controlled experiments are tricky, quasi-experiments ("quasi" means "as if" or "almost" in Latin) may help data scientists answer questions about causality. If you're interviewing with a marketplace company such as DoorDash, Uber, Lyft, Airbnb, etc. for which regular A/B testing can be problematic, chances are you're expected to have some knowledge of those "non-traditional" methods.
# Switchbacks

For DoorDash to test SOS pricing properly, it's necessary to split the dasher and the customer pools either across space (different markets) or across time (different time of the day). In fact, they do both (DoorDash wrote a wonderful [blog post](https://doordash.news/2018/02/14/switchback-tests-and-randomized-experimentation-under-network-effects-at-doordash/)).


{{< figure src="https://miro.medium.com/max/1400/1*wTkgNdjmhOeMUjfGbpqzJQ.png" width="600" caption="DoorDash tests SOS pricing using switchbacks across time-market units">}}

We can select multiple markets for experimentation and evenly divide each day in each market into small chunks, say, half an hour. Each time-market unit is randomly assigned to having or not having SOS pricing — by contrast, randomization units in regular A/B testing would probably be deliveries. To examine whether SOS pricing is helpful, we can compare key metrics such as delivery times and order lateness between the treatment units and the control units after the switchback experiment.

To design effective switchback experiments, below are a few things to think over:
1. **How fine should the time units be?** Two hours, half an hour, or 10 minutes? Too coarse, there won't be enough randomization units; too granular, the market won't have time to respond to the change and the variance would probably be high (market trends are smoother over a longer period of time and more volatile in a shorter period). We can try a bunch of window sizes to see when we achieve a reasonably low standard error $\frac{\sigma}{\sqrt{n}}$ ($n$ is the number of time-market units).

2. **Time units in the same market are dependent**. If we know how deliveries in LA are like at 11:30 AM, we can probably take a good guess about deliveries at 12:00 PM, even though pricing strategies may differ between these two units. To address the dependence issue, we can use [**multilevel models**](https://stats.stackexchange.com/questions/4700/what-is-the-difference-between-fixed-effect-random-effect-and-mixed-effect-model) where the **pricing strategy is the "fixed effect"** (captures the effect of the variable we're interested in) and **time and markets are the "random effects"** (capture group-level variation). Below is a toy implementation using the `lme4` package in R:

    ```r
    library(lme4)
    sos <- lmer(delivery_time ~ has_sos + (1|market) + (1|time), data=sb_data)
    ```

    Moreover, we can use the "[Huber Sandwich Estimator](https://stats.stackexchange.com/questions/50778/sandwich-estimator-intuition)" to compute variance, which doesn't make the independence assumption like ordinary least squares (OLS) do.

3. **Weight each time-market unit by the number of deliveries or not?** The number of deliveries may well differ by time-market unit; it's natural to think units with more deliveries should count more towards the average metric value in a given variant ($\frac{\sum_{i}^{n}y_i w_i}{n}$; $y_i$: metric value in unit $i$; $w_i$: delivery volume in unit $i$). In practice, DoorDash data scientists use the simple, rather than weighted, average in each variant ($\frac{\sum_{i}^{n}y_i}{n}$). One justification [they gave](https://doordash.news/2018/02/14/switchback-tests-and-randomized-experimentation-under-network-effects-at-doordash/) is that randomization units are time-market units, so delivery-level information can be ignored.

4. **Is the feature change visible?** Switchbacks are great for testing invisible algorithmic changes (e.g., ranking, pricing) but terrible for visible UI changes; in the latter case, users are bound to be confused if they are switched back and forth between the old and the new designs.😵‍💫

# Difference in Differences (DiD)

~~Facebook~~ Meta famously did a country test before [shipping "Reactions"](https://developers.facebook.com/videos/f8-2017/how-we-shipped-reactions/). It used to be the case that users could only like a post but not express other emotions such as anger or sadness. If someone posted about a loved one passing away, it seemed inappropriate to like it, yet many might not want to leave a comment (e.g., too effortful, not close enough). Meta data scientists hypothesized that, if a wider range of reactions were allowed, people would be more willing to engage with posts. They could run a regular A/B test, randomly assigning users to having or not having Reactions. However, if users in different variants are in the same friend circle, you risk creating bad user experiences: Say a treatment user reacted to a post by a control user, the latter would not be able to see it and engage back. 

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

# Propensity Score Matching (PSM)

Care for another HelloFresh example? Say their data scientists wanna know if users who click on Food Network ads are more likely to buy from HelloFresh. 

Here's the problem if we directly compare clickers vs. non-clickers: Ads are usually personalized, so those who are shown Food Network ads likely differ from those who aren't; if we do observe a difference in HelloFresh purchases, should we attribute it to the ads or individual differences? Chances are, those interested in cooking are both inclined to click on Food Network ads and buy from HelloFresh.

{{< figure src="https://www.dropbox.com/s/iexwhzrjzzxtokq/propensity.png?raw=1" width="550" caption="It's hard to tell if ad clicking causes HelloFresh purchases or if clicks and purchases are both driven by a third variable: Interest in cooking">}}

Propensity score matching (PSM) is a solution: We can build a model (using user demographics or past behaviors) to predict the probability that a given user is shown any Food Network ads and match users with high exposure probability but didn't click with those who had similar probability and clicked. Presumably, matched users have the same "propensity" to click on ads, so downstream differences in HelloFresh purchases can be attributed to the act of ad clicking.

{{< youtube gaUgW7NWai8 >}}

The idea behind PSM sounds simple but it's tricky [how we can match users](https://cran.r-project.org/web/packages/MatchIt/vignettes/matching-methods.html):

1. **Greedy vs. optimal**: For each of the $N$ users in Group A, we can iterate through each of the $N$ users in Group B to find the best match 👉 time complexity: $O(N^2)$. However, this fast approach doesn't guarantee optimal results on the group level. Alternatively, we can exhaust all one-to-one match combinations to find the one that minimizes total within-pair differences 👉 this optimal approach is computationally expensive; time complexity: $O(N!)$
2. **Constraints**: We can constrain what we consider as a match. For instance, we can set a threshold on the largest propensity score difference allowed for matched pairs (caliper matching). Another common constraint is whether we only consider one-to-one matching or if we allow one-to-many or even many-to-many matching.

Matching algorithms are worth dedicating an entire book to; for data science interviews, it's enough to recognize when to use PSM and know what the "propensity" is.



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

# Resources

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

---

<!-- > [Causal Data Science Meeting 2021](https://www.causalscience.org/) takes place between November 15 and 16!
 -->
{{< figure src="https://www.dropbox.com/s/1dlnw8sr12ifr0m/pearl.png?raw=1" width="500">}}