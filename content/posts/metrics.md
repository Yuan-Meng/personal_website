---
title: Choosing Metrics
date: 2021-11-06
tags:
    - metrics
    - product
categories:
- notes
keywords:
    - metrics, product
---


'Tis the college and job application season of the year. If only schools and companies have crystal balls to see into each candidate's future achievements, they need not interview people; since they do not have such things, SAT scores, GPA, internships, and other quantifiable metrics are used to aid decisions. Simplified metrics are by no means ideal — As Goodhart put it (often [paraphrased](https://en.wikipedia.org/wiki/Goodhart%27s_law)), "When a measure becomes a metric, it ceases to be a good measure." However, the opposite is worse: Making critical decisions based on heuristics, biases, and personal opinions.

The solution? It's *not* not using metrics but carefully designing good metrics:

> "[...] faulty intuition, untrusted partners, and complex systems can be understood via intuitive, trustworthy, simple metrics." — David Manheim, [*Goodhart's Law and Why Measurement is Hard*](https://www.ribbonfarm.com/2016/06/09/goodharts-law-and-why-measurement-is-hard/)

# Design Good Metrics

1. **Goal-oriented**: Good metrics should track the most appropriate goals given your business model and stage (think lean analytics stuff) 👉 Counterexample: For a new startup, user growth is more important than immediate revenue, so it would be unwise to bombard your site with ads ("杀鸡取卵") early, or ever
2. **Causal**: Ideally, metric movements should be caused by whether or not you're closer to your goals 👉 Counterexample: Stats 101 grades are correlated with Stats 102 grades, but inflating grades in 101 won't improve 102 grades; it's the students' prerequisites + interests + efforts that matter
3. **Not gameable**: Related to #1, good metrics should only be moved by goal attainment; it should be hard to make them look good without achieving actual goals of the product or the company 👉 Counterexample: The police may choose not to report crime if solely rewarded by low crime rate; exams are designed to measure learning, but one may get by using test-taking skills ("三长两短选一长") 
4. **Actionable:** Good metrics help make good decisions 👉 Counterexample: For a software-as-a-service (SaaS) company, by the time a user has churned, it's already too late (resurrecting a churned user is hard and acquiring a new one is costly); better to monitor engagement and usage to prevent churning
5. **Simple:** Easy to understand and compute, so people have no trouble discussing and debugging them 👉 Counterexample: weighted sum of multiple metrics 
6. **Clear:** No ambiguity in interpretation 👉 Counterexample: For a search engine, longer time per session could mean people are so engaged that they keep going down rabbit holes or that they are so confused that they can't find what they want; we simply can't tell (a better metric would be # of sessions per user)
7. **Measurable**: Can be easily measured within a reasonable time frame 👉 Counterexample: For a MOOC site, it's hard to track what people do with the knowledge later, even though that's super relevant to its mission
8. **Comparable:** Across time, products, and even companies 👉 Counterexample: Absolute changes are hard to compare, so better use relative quantities
9. **Trade off between goals:** When we have more than one goal, good metrics offer a trade-off — when driving somewhere, we want to be fast but don't want to get tickets 👉 **speed / # of ticket** is low when we drive too fast (too many tickets) or too slow (low speed) and high when we're reasonably fast

# Debug Bad Metrics
    
> "[...] metrics sometimes becomes an excuse to for doing fun math and coding instead of dealing with messy and hard-to-understand human interactions." — David Manheim, [*Overpowered Metrics Eat Underspecified Goals*](https://www.ribbonfarm.com/2016/09/29/soft-bias-of-underspecified-goals/)

1. **Think of edge/corner cases to "game" the proposed metric**
    - e.g., Udacity may achieve high conversion rate (# of conversions / # of sign-ups) by steering users away from signing up (e.g., "Hey it's too much work! You can't possibly finish...")
2. **Think of long-term goals that may be harmed by short-term metrics**
    - e.g., Quora can increase revenue by encouraging all authors to monetize their content, but if there's no restriction on how much content goes behind the paywall, soon users may leave for lack of good free content
3. **Bonus: Think of metrics that better align with company/product/feature goals**


# How to Think of Metrics?

Not so long ago, I thought of "product sense" as a misnomer: No one is born with such a sense and there's always deliberation. After thinking extensively about products for a while, I do "sense" something now: Seeing a new product or design, I can't help but wonder how it came to be, along with what metrics capture that.

For instance, when Google Maps started suggesting fuel-efficient routes, I was amazed and immediately wondered, **how do we know if this cool feature is successful**? 

{{< figure src="https://www.dropbox.com/s/jw68iz18rgjp642/metrics.jpg?raw=1" width="400" >}}

To examine whether something is successful, we need to know why it was design in the first place. Then after that, I always ask myself the following questions:
1. **Adoption and usage**: How many people use it? Do they use it much? And do they use it in ways that you care about? — However awesome, an obscure product or feature is hardly a success
2. **Company goals**: Okay, people use it, a lot. But does that serve the designer's original purposes?
3. **Counter metrics**: Are there unintended consequences?  From time to time, I think about the [value alignment problem](https://deepmind.com/research/publications/2020/Artificial-Intelligence-Values-and-Alignment) I first heard in Alison Gopnik's AI seminar: If you build an AI that maximizes the number of paperclips made, then it will wreak havoc by turning all the metal on this planet into paperclips. Make sure the new feature/product is nothing like this paperclip-making AI (e.g., product cannibalization, a more sophisticated but way less performant model)...

> (There are many popular metric frameworks out there, such as "AARRR", but above is how I like to think about the problem at hand.)

For Google Maps, fuel-efficient route suggestions hopefully give them a competitive advantage over, say, Apple Maps: iOS users might be less likely to "multi-tenant" (using both Google Maps and Apple Maps) and those who rely on Google Maps may feel free to take more trips. To test these hypotheses, we can examine changes in daily active users, trips per week per user, etc.. However, this new feature may not be all perfect  — fuel-efficient routes may be way slower than the fasted routes, so we should watch out for drastic increases in time between point A and point B (I later read an article about this on [TechCrunch](https://techcrunch.com/2021/10/06/google-maps-launches-eco-friendly-routing-in-the-u-s/); check it out if you're intrigued). 


If stuck, below are some "frameworks" to fall back on to brainstorm metrics:

- **User journey**: Most products have a conversion funnel that moves new users towards users who frequently take "core actions" (e.g., Quora: answer questions; Notion: become a paid user) of the business 👉 Track each step in this funnel to come up with relevant metrics (and diagnose problems, if any)
- **Input/output**: Input metrics are actions you take or resources you put in (e.g., DoorDash's driver incentive program pays driver extra dollars to work during peak hours) and outcome metrics are results you achieve (e.g., decreased delivery times, improvements on order lateness, higher customer retention)
- **Know the domain**: If you're working for DoorDash, you know the three sides of the market each have its own needs and wants (merchants: reach + revenue; dashers: flexibility + earnings; customers: selection + convenience), so you can think of proper metrics to track these needs and wants 👉 This is just one example; it helps a ton to read engineering blogs of the company you're interviewing with to see what matters in their specific domain

# Types of Metric Interviews

I've seen metric questions in both product and machine learning interviews. This is not surprising since no matter how you solve a business problem (experiments + hypothesis testing, machine learning), you need metrics to validate your solution.

- **Metric design**: Come up with or choose metrics that appropriately capture the goodness of a product/feature (using A/B testing to make launch decisions or measuring the success of past launches) or a model (evaluating whether an ML system is working properly, especially compared to older systems)
- **Metric evaluation**: Explain strange metric movements (e.g., Why did X dropped by Y%?) or critique wrong/misleading metrics (Does it measure what we want?)

**Regarding metric design**: [This video](https://youtu.be/nPJKFWMiIC8) said the rule of thumb is to choose 2 success metrics and 1 counter metric. However, I've been explicitly asked to provide 6+ metrics in one interview. So be sure to think thoroughly about which metrics capture what you want.

**Regarding metric evaluation**: When evaluating a metric itself, think about the principles behind good metrics mentioned earlier as well as ways to "hack" metrics; when investigating a metric movement, rule out "stupid" reasons (internal: logging errors, feature launching, outages, etc.; external: seasonality, competitors, competitor outages, regulations, etc.) first, and then break down the "haystack" (each component that makes up the metric, all parties involved, different segments, etc.) to look for the "needle" (e.g., profound changes in product/feature health) 👉 When talking to product data scientist friends, I like asking them the first thing they'd do after waking up to a jaw-dropping dashboard, for insights and for fun :)

Surprising as it may sound, I find Chip Huyen's [Design a Machine Learning System](https://huyenchip.com/machine-learning-systems-design/design-a-machine-learning-system.html) and Martin Zinkevich's [Rules of Machine Learning: Best Practices for ML Engineering](http://martin.zinkevich.org/rules_of_ml/rules_of_ml.pdf) helpful for product data scientists as well. When choosing loss functions and evaluation metrics, ML folks also need to think about how users would interact with an ML-powered feature (user actions both serve as training data and data for online evaluation after deployment) and which actions to optimize for (e.g., clicks, shares, time per session, etc.). Admittedly, I wouldn't have read these if I hadn't been preparing for an ML interview... Still, I'd recommend these two articles to product DS candidates who have extra time (😂) and wish to think about metrics even more precisely through an ML lens.


# Resources I Used

As a fresh grad with no product experience, I learned all of the above in the past 2 months from 4 product books (*Lean Analytics*, *Decode & Conquer*, *Cracking the PM Interview*, *Trustworthy Online Controlled Experiments*), dozens of company engineering blogs, TechCrunch articles, PyData and Y Combinator talks, etc. or just pondering about products. Not everything is helpful; below are what I found helpful:

1. **Lean Analytics**: Must, must read to understand which metrics are most relevant for which business models (e.g., e-commerce, user-generated content, multi-sided marketplace, media, etc.) at which stage 👉 Chances are, most tech companies you're interviewing with have one of one those business models (perhaps 23andMe, for instance, doesn't not fit into these categories...)
2. **Company research**: Many companies have great engineering blogs; public companies have earnings calls  👉 From there, you can learn what exact metrics companies use for what purposes (e.g., DoorDash uses order lateness to optimize delivery time prediction models, rather than overall accuracy — this makes a ton of sense because late orders are way worse for the customers than early orders)
2. **a16z**: Pure gold recommended by Quora — [16 Startup Metrics](https://a16z.com/2015/08/21/16-metrics/), [16 More Startup Metrics](https://a16z.com/2015/09/23/16-more-metrics/), and [16 Ways to Measure Network Effects](https://future.a16z.com/how-to-measure-network-effects/) 👉 Similar to Lean Analytics, but have some interesting additions (e.g., multi-tenanting)
3. **Emma Ding**: Check out her [metric framework](https://towardsdatascience.com/the-ultimate-guide-to-cracking-business-case-interviews-for-data-scientists-part-1-cb768c37edf4) and [product-related videos](https://www.youtube.com/watch?v=X8u6kr4fxXc&list=PLY1Fi4XflWStFs6tLQ3Gey2Aaq_U4-Xnc) 👉 I think Emma's content is a wonderful start to understanding product and metrics but you way wish to check out #1-3 to dive deeper afterwards
4. **Experiment metrics**: [Trustworthy Online Controlled Experiments: A Practical Guide to A/B Testing](https://www.amazon.com/Trustworthy-Online-Controlled-Experiments-Practical/dp/1108724264) and Udacity's [A/B testing course](https://www.udacity.com/course/ab-testing--ud257) both discuss experiment metrics in detail 👉 Experiment metrics are often shorter-term and more granular than success metrics of a product, so be sure to go over #1-4

# Final Thoughts

Since interviews tend to mimic real-life problems, I sometimes imagined, had I worked as product data scientist for a week, all of those metric shenanigans would've made much more sense to me more quickly ("纸上得来终觉浅"😬)... I'll revisit this writing after having worked on real products in the not-so-distant future. 

Meanwhile, I often hear fellow grads complaining how "unfair" it is that companies ask product or even A/B testing questions since we have no experiences — whenever someone says that, I think about law or med school students: Before working on actual court cases or patients, it is possible and desirable have the theoretical foundations. And if you find this process fun, more reasons to make it your career.