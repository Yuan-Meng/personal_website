---
title: Quasi-Experiments for Causal Inference 
date: 2021-11-11
math: true
tags:
    - causal inference
    - quasi-experiment
categories:
- notes
keywords:
    - causal inference, quasi-experiment
---

> "Causation [...] is the cement is the universe." — David Hume, *Abstract*


Data scientists help the team make good decisions and good decisions rely on causation. [Prof. Alison Gopnik](http://alisongopnik.com/) always has the nicest example: While yellow fingers and smoking are both correlated with lung cancer, washing hands doesn't prevent cancer. 

Randomized controlled experiments, or A/B tests, are the gold standard for establishing causality. However, it's not always possible, feasible, or ethical to randomly assign people into different variants: In the example above, you can't in good conscience force someone to smoke, nor can you easily have them quit smoking. Worse still, even if you've run what seems like a well-controlled experiment, differences may still arise from factors other than your treatment. For instance, say DoorDash A/B tested SOS pricing (their version of "surge pricing": customers pay more for each order during peak hours) and didn't observe significant drops in delivery times in treatment compared to control. Is SOS pricing useless? Not necessarily — A given market shares the same pool of dashers and customer; only the treatment variant "paid the price" but the control variant reaped the benefit (e.g., increased dasher supplies + decreased order demands 👉 shorter delivery times + fewer late orders) for free, resulting in smaller observed differences than what could have been the "ground truth".

Where randomized controlled experiments are tricky, quasi-experiments ("quasi" means "as if" or "almost" in Latin) may help data scientists answer questions about causality. If you're interviewing with a marketplace company such as DoorDash, Uber, Lyft, Airbnb, etc. for which regular A/B testing can be problematic, chances are you're expected to have some knowledge of those "non-traditional" methods.

# Switchbacks

For DoorDash to test SOS pricing properly, it's necessary to split the dasher and the customer pools either across space (different markets) or across time (different time of the day). In fact, they do both (check out DoorDash's [blog](https://doordash.news/2018/02/14/switchback-tests-and-randomized-experimentation-under-network-effects-at-doordash/)).


{{< figure src="https://miro.medium.com/max/1400/1*wTkgNdjmhOeMUjfGbpqzJQ.png" width="500" caption="DoorDash tests SOS pricing using switchbacks across time-market units">}}

Multiple markets are selected for experimentation and the time of the day in each market is evenly split into small chunks, say, half an hour. Each time-market unit (rather than a delivery, for example, in regular A/B testing) is randomly assigned to having or not having SOS pricing. To examine whether SOS pricing is effective, key metrics such as delivery times and order lateness are compared between "treatment units" and "control units" after the switchback experiment.

A few things to think over:
1. **How fine should the time units be?** Two hours, half an hour, or 10 minutes? Too coarse, there won't be enough randomization units; too granular, the market won't have time to respond to the change and the variance would probably be high (market trends are smoother over a longer period of time and more volatile in a shorter period). We can try a bunch of window sizes to see when we have a reasonably low standard error $\frac{\sigma}{\sqrt{N}}$ ($N$ is the number of market-time units).

2. **Time units in the same market are dependent**. Deliveries at 11:30 AM in LA are probably highly similar to deliveries at 12 PM in LA, even though the pricing strategy might differ. To address this dependence issue, we can use mixed models where the pricing strategy is the "fixed effects" and time and markets are "random effects". Below is a toy implementation using the `lme4` package in R:

    ```r
    library(lme4)
    sos <- lmer(delivery_time ~ has_sos + (1|market) + (1|time), data=sb_data)
    ```

    Another way to correct for dependence is using the so-called "[Huber Sandwich Estimator](https://stats.stackexchange.com/questions/50778/sandwich-estimator-intuition)" (R package: `sandwich`) that relaxes the independence assumption.

3. **Weight each market-time unit by the number of deliveries or not?** Since randomization units are market-time units, we can ignore delivery-level information and use the simple, rather than weighted, average in each variant

4. **Is the change visible?** Switchbacks are great for testing invisible algorithmic changes (e.g., ranking, pricing), but not visible UI changes 👉 Users will be confused if they are switched back and forth between old and new designs😵‍💫

# Difference in Differences (DiD)

~~Facebook~~ Meta famously did a country test before [shipping "Reactions"](https://developers.facebook.com/videos/f8-2017/how-we-shipped-reactions/). It used to be the case that users could only like a post but not express other emotions such as anger or sadness. If someone posted about a loved one passing away, it seemed inappropriate to like it, yet many might not want to leave a comment (e.g., too effortful, not close enough). Meta data scientists hypothesized that, if a wider range of reactions were allowed, people would be more willing to engage with posts. They could run a regular A/B test, randomly assigning users to having or not having reactions. However, if users in different variants are in the same friend circle, you risk creating bad user experiences: Say a treatment user reacted to the post by a control user, the latter would not be able to see it and engage back. 

To alleviate this worry, Meta tested Reactions in different countries, assuming users in different countries didn't know each other. In a simplified example, country A (green) had reactions and country B (red) didn't. The $y-$axis tracks the number of posts reacted (including likes) to per user, which Meta aimed to drive up.

{{< figure src="https://www.dropbox.com/s/s9x78epex3zw1lk/reactions.png?raw=1" width="500" caption="Changes in # of posts reacted to per user after launching Reactions">}}


- **Assumptions**: To identify the treatment effect, we have to assume 1) the same country different across time and 2) the same time effect across countries. 

- **Analysis**: Under those assumptions, if Reactions had no effect, the green line would be parallel to the red line after feature launching. **The difference between the actual green line and the counterfactual under the null hypothesis is the treatment effect**. This method is called "difference in differences (DiD)". We can use a regression model to capture this analysis ($y$: success metric; $D_{post}$: 0 $\rightarrow$ pre-test, 1 $\rightarrow$ post-test; $D_{treatment}$: 1 $\rightarrow$ treatment, 0 $\rightarrow$ control) —
  - $y = \beta_0 + \beta_1 D_{post} + \beta_2 D_{treatment} + \beta_3 D_{post} D_{treatment} + \beta_4 X + \epsilon$ 
  - The interaction coefficient $\beta_3$ is the difference in differences

DiD is fast but crude: Both of the assumptions above can be easily violated, resulting in invalid conclusions. As we'll see in a minute, more sophisticated methods such as synthetic control are developed to address issues like this.

# Synthetic Control (Synth)

The main idea behind synthetic control is that we can train a model to learn the relationship between covariates and the target metric in the pre-launching period and use that relationship to predict what the metric would look like without the treatment in the post-launching period. This method is called "synthetic" control because there never was an actual control group; the model prediction served as the control and is compared with the actual trend after feature launching.

Below is a neat talk by Uber data scientists on how they tested cash trips. In markets like India where not a lot of people carry credit cards, riders often pay drivers in cash. Cash trips may be inconvenient to the drivers as they need to wire the commission fees to Uber later; however, some drivers may prefer to receive cash. Uber wants to test how giving drivers heads up about whether a trip is a cash trip might affect how many cash trips they gave. 

{{< youtube j5DoJV5S2Ao >}}

Now, Uber faces the same problem as DoorDash did: A market shares the same pool of drivers and riders; if the treatment takes more or fewer cash trips, the control would have fewer or more cash trips to take. Switchbacks are inappropriate here: Imagine, a driver was in treatment and received heads up about cash trips; later when they are switched to control, they will naturally interpret a lack of heads up as a non-cash trip rather than type unknown. 

Uber used synthetic control in this case, building a time series model to predict what would happen in a city had the feature not been rolled out ("synthetic city") and comparing it with what did happen after THE rollout ("treatment city"). The difference between the synthetic and the treatment cities are the treatment effect. To report to stakeholders, you can average differences over time.

{{< figure src="https://www.dropbox.com/s/tebslmo8g1agtlm/synthetic_controk.png?raw=1" width="550" caption="Uber used synthetic control to test heads up to drivers about cash trips">}}


As you can tell, the problem boils down to building an accurate and robust time-series model and *not* tweaking it till you find treatment effects.

# DiD + Synth 👉 CausalImpact

[CausalImpact](https://google.github.io/CausalImpact/CausalImpact.html) combines ideas from DiD and synth. Unlike synthetic control, there is an actual control group; unlike DiD, there's no need to assume invariant regional differences and time effects — instead, we build a Bayesian structural time-series models using the trend in control to predict what the trend would look like in treatment and compare this "synthetic treatment" with the actual treatment.

{{< youtube KEhJNM5K73A >}}

HelloFresh used CausalImpact to test a YouTube ad campaign across geographical regions. The treatment saw the ads and the control didn't. HelloFresh data scientists used the trend in the control conversation rate to predict what the treatment conversation rate would be without the campaign — the impact of the ad campaign lies in the difference between the predicted and the actual treatment conversion rates. 

Google open sourced the `CausalImpact` R package. Below is a toy example:

```r
library(CausalImpact)
impact <- CausalImpact(data, pre.period, post.period)
plot(impact)
```

{{< figure src="https://www.dropbox.com/s/cwtreolbp7h9477/causalimpact.png?raw=1" width="550" caption="Example results: The top panel shows the actual trend (solid) and the counterfactual (dashed) and the middle panel the difference between them">}}


# Propensity Score Matching (PSM)

Say HelloFresh wants to see if people who click on Food Network ads are more likely to buy from HelloFresh. Here's the problem: Ads are usually personalized, so those who are shown such ads likely differ from those who aren't; if we do observe a difference in HelloFresh purchases between clickers and non-clickers, should we attribute it to the ads or individual differences? It could be the case that those interested in cooking tend to click on Food Network ads and buy from HelloFresh.

{{< figure src="https://www.dropbox.com/s/iexwhzrjzzxtokq/propensity.png?raw=1" width="550" caption="In the HelloFresh example, it's hard to tell whether ad clicking causes buying or if both are driven by a third variable: Interest in cooking">}}


The solution? We can build a model using user demographics, past behaviors, etc. to predict the probability that they are shown a Food Network ad and match users with high exposure probability but didn't click on Food Network ads with those who had similar probability and clicked. This method is called "propensity score matching" (PSM). Presumably, matched users have the same "propensity" to click on ads, so we can more confidently attribute the downstream difference in their HelloFresh purchases to whether or not they have clicked on the ads.

{{< youtube gaUgW7NWai8 >}}

The idea behind PSM sounds simple but it's tricky [how we can find matches](https://cran.r-project.org/web/packages/MatchIt/vignettes/matching-methods.html):

1. **Greedy vs. optimal**: For each of the $N$ users in Group A, we can search through each of the $N$ users in Group B to find the best match 👉 time complexity: $O(N^2)$. However, this approach doesn't guarantee optimal results on the group level. Alternatively, we can exhaust all one-to-one matches to find the combination that minimizes within-pair differences 👉 time complexity: $O(N!)$. 
2. **Constraints**: We can constrain what we consider as a match. For instance, we can set a threshold on the largest propensity score difference allowed for match pairs (caliper matching). Another common constraint is whether we only consider one-to-one matching or if we allow one-to-many matching.

Matching algorithms are worth dedicating an entire book to; for data science interviews, it's enough to recognize when to use PSM and know what the "propensity" is.



# Regression Discontinuity Design (RDD)

Let's think about another DoorDash example. Late orders anger hungry customers; the later, the worse 👉 frustrated customers may churn or order less in the future, resulting in lower lifetime values (LTVs). To make things slightly better, DoorDash automatically issues a refund to orders more than 30 minutes late. The question is, how does the refund impact customer LTV? Obviously, it's unfair to randomly assign customers into getting or not getting a refund. However, it's also problematic to compare LTV between those who received or didn't receive refunds, because average lateness is likely higher in the former than the latter.

To answer this question, we can use order lateness (minutes late) and refund status (received or not) to predict LTV and look for a "jump" at the 30-minute refund cutoff. The upward jump is how much auto refunds saved DoorDash's customer LTV.

{{< figure src="https://www.dropbox.com/s/vz60f040euu0465/rdd.png?raw=1" width="550" caption="Using regression discontinuity design (RDD) to measure the impact of automatically refunds on customer LTV when orders arrive late 👉 $y = f(X) + \beta D + \epsilon$ ($y$: LTV; $X$: order lateness; $D$: refund $\rightarrow$ 1, no refund $\rightarrow$ 0)">}}

Depending on if the cutoff is deterministic or not, RDD breaks down into [two types](https://scholar.princeton.edu/sites/default/files/jmummolo/files/rdd_jm.pdf):
- **Sharp (deterministic)**: Orders $\geq$ 30 minutes late definitely receive a refund and those $<$ 30 minutes late definitely don't 
- **Fuzzy (probabilistic)**: The admission office has a suggested SAT cutoff, but students with lower scores might still get in through special programs

The core assumption behind RDD is that the trajectories of "near-winners" and "near-losers" would have been the same without the "reward", so the discontinuity (jump between orange lines) can be attributed to it. However, the relationship between the "running variable" (order lateness) and the outcome (LTV) may be naturally "jumpy" around the cutoff, rendering the conclusion invalid.

# "Regress It Out"

Continuing with the regression idea, we can statistically control for covariates by "regressing them out". Say we wanna know how gender impacts income, we can put potential confounders such as education and age in the same model and look at the slope of gender with all else being held the same. If this slope is much steeper than a horizontal line, then we can probably claim that gender affects income.

```r
bias <- lm(income ~ gender + education + age, data=data)
```

While simple and useful, this method is not always appropriate ([Rohrer, 2018](https://journals.sagepub.com/doi/pdf/10.1177/2515245917745629)):
- **Collider**: If $X$ is the common effect of $Y$ and $Z$ ($Y \rightarrow X \leftarrow Z$), controlling for $X$ would result in spurious correlation between $Y$ and $Z$. 
  - **Example**: Warm and competent candidates tend to successful. In other words, a job offer is the common effect of warmth and competence. Since all of our colleagues were once successful candidates (i.e., interview results are "controlled for"), when we look around in the office, everybody may seem both warm and competent. If not careful, we may jump to the conclusion that those two traits are intrinsically linked.
- **Mediator**: If $X$ influences $Y$ through $Z$ ($X \rightarrow Y \rightarrow Z$), controlling for $Y$ leads to the false conclusion that $X$ and $Y$ have no relationship at all. 
  - **Example**: Great musicians love what they do, love motivates them to practice, and practice makes perfect. If we look at a bunch of Curtis students, all of whom practice a ton, we may falsely conclude that someone's love of music has nothing to do achievements as a musician.

Since we don't always know how variables are related (which can be expressed by Bayesian networks), statistical control may hurt unexpectedly. 

---
# Summary 

As a cognitive scientist, I'm writing a dissertation on how people think about causality intuitively. As a data scientist interested in causal inference, it's fascinating how formal methods and human intuitions converge: For both, **the heart of causality is counterfactuals**, worlds that could have been but never came to be. 

Summarized below are common causal inference methods in data science I wrote about:

- **Intervention and counterfactuals**: The common assumption behind DiD, synth, CausalImpact, and RDD is that if we don't do anything (e.g., launching a new feature), nothing will happen; since something did happen, then what we did had an impact. This falls under the [interventionist](https://plato.stanford.edu/entries/causation-mani/#Inte) view of causation: We know A causes B when *iif* doing A makes B happen. The difference between these methods lies in how they construct the counterfactual world without the treatment.

- **Statistical control**: Apart from what we do, lots of other things in the world can make a difference to the outcome we care about. In both PSM and regression, the idea is to hold other potentially impactful variables constant and check if the variable we care about the most still changes the outcome. This is neat but be aware that controlling for wrong variables (e.g., comment effects and mediators) can lead to wrong conclusions.

Just a few years ago, causal inference was rather niche in data science; I'm happy to see those methods quickly gaining popularity in the data science community, as a cognitive scientist, data scientist, and lover of the philosophy of science.

# Resources

1. [Lessons Learned on Experimentation @DoorDash](https://www.ai-expo.net/northamerica/wp-content/uploads/2018/11/1500-Jessica-Lachs-DoorDash-DATA-STRAT-V1.pdf)
2. [Switchback Tests and Randomized Experimentation Under Network Effects at DoorDash](https://doordash.news/2018/02/14/switchback-tests-and-randomized-experimentation-under-network-effects-at-doordash/)
3. [Under the Hood of Uber's Experimentation Platform](https://eng.uber.com/xp/)
4. Experimentation in a Ridesharing Marketplace by Lyft ([Part 1](https://eng.lyft.com/experimentation-in-a-ridesharing-marketplace-b39db027a66e#.djox1933t), [Part 2](https://eng.lyft.com/https-medium-com-adamgreenhall-simulating-a-ridesharing-marketplace-36007a8a31f2#.g9b34i3gm), [Part 3](https://eng.lyft.com/experimentation-in-a-ridesharing-marketplace-f75a9c4fcf01))
5. [Causal Inference Using Synthetic Control: The Ultimate Guide](https://towardsdatascience.com/causal-inference-using-synthetic-control-the-ultimate-guide-a622ad5cf827)
6. [Inferring the Effect of an Event Using CausalImpact](https://www.youtube.com/watch?v=GTgZfCltMm8)
7. [The Book of Why: The New Science of Cause and Effect ](https://www.amazon.com/Book-Why-Science-Cause-Effect/dp/046509760X)
8. [Causal Inference for The Brave and True](https://matheusfacure.github.io/python-causality-handbook/landing-page.html)
9. Liu, T., Ungar, L., & Kording, K. (2021). Quantifying causality in data science with quasi-experiments. *Nature Computational Science*, 1(1), 24-32. ([PDF](https://www.nature.com/articles/s43588-020-00005-8.pdf))

---

> [Causal Data Science Meeting 2021](https://www.causalscience.org/) takes place between November 15 and 16!

{{< figure src="https://www.dropbox.com/s/1dlnw8sr12ifr0m/pearl.png?raw=1" width="500">}}