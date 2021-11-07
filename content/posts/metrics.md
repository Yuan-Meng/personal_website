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

# Design Good Metrics

'Tis the college and job application season again. If only schools and companies have crystal balls to see into each candidate's future achievements, they need not interview people; since they do not have such a thing, SAT scores, GPA, internships, and other quantifiable metrics are used to aid decisions. Simplifications are by no means ideal — As Goodhart put it ([paraphrased](https://en.wikipedia.org/wiki/Goodhart%27s_law)), "When a measure becomes a metric, it ceases to be a good measure." However, the opposite is worse: Making critical decisions based on heuristics, biases, and opinions.

The solution? It's *not* not using metrics but carefully designing good metrics:

> "[...] faulty intuition, untrusted partners, and complex systems can be understood via intuitive, trustworthy, simple metrics." — David Manheim, [*Goodhart's Law and Why Measurement is Hard*](https://www.ribbonfarm.com/2016/06/09/goodharts-law-and-why-measurement-is-hard/)

## Characteristics of Good Metrics

1. **Goal-oriented**: Good metrics should track the most appropriate goals given your business model and stage (i.e., lean analytics stuff) 👉 Counterexample: For an early startup, user growth is more important than immediate revenue
2. **Causal**: Ideally, metric movement should be caused by whether you're closer to or further from your goal 👉 Counterexample: Stats 101 grades are correlated with stats 102 gades, but grade inflation in 101 won't improve 102 grades; it's prerequisite + interest + effort that matter
3. **Not gameable**: Related to #1, good metrics should only be moved by goal attainment and it's hard to make them look good without achieving the true goals of the product or the company 👉 Counterexample: The police may not to report crime if solely rewarded by low crime rate; exams are designed to measure learning, but one may get by using test-taking skills ("三长两短选一长") 
4. **Actionable:** Good metrics help make good decisions 👉 Counterexample: For a software-as-a-service (SaaS) company, by the time a user has churned, it's already too late (resurrecting a churned user is hard and acquiring a new one is costly); better to monitor engagement and usage to prevent churning
5. **Simple:** Easy to understand and compute, so people have no trouble discussing and debugging them 👉 Counterexample: weighted sum of multiple metrics 
6. **Clear:** No ambiguity in interpretation 👉 Counterexample: For a search engine, longer time per session could mean people so engaged that they keep going down rabbit holes or that they are so confused that they can't find what they want; we simply can't tell (a better metric would be # of sessions per user)
7. **Measurable**: Can be easily measured within a reasonable time frame 👉 Counterexample: For a MOOC site, it's hard to track what people do with the knowledge later, even though that's super relevant to its mission
8. **Comparable:** Can be compared across time, products, and even companies 👉 Counterexample: Absolute changes are harder to compare than relative quantities
9. **Trade off between goals:** When we have more than one goal, good metrics offer a trade-off — when driving somewhere, we want to be fast but don't want to get tickets 👉 **speed / # of ticket** is low when we drive too fast (too many tickets) or too slow (low speed) and high when we're reasonably fast

## Debug Metrics
    
> "[...] metrics sometimes becomes an excuse to for doing fun math and coding instead of dealing with messy and hard-to-understand human interactions." — David Manheim, [*Overpowered Metrics Eat Underspecified Goals*](https://www.ribbonfarm.com/2016/09/29/soft-bias-of-underspecified-goals/)

1. **Think of edge/corner cases and find ways to "game" the proposed metric**
- e.g., Udacity may achieve high retention rate (# of conversions / # of sign-ups) by steering users away from signing up
2. **Think of long-term goals that may be harmed by short-term metrics**
- e.g., Quora can increase revenue by encouraging all authors to monetize their content, but if there's no restriction on how much content goes behind the paywall, soon users may leave for a lack of good free content
3. **Bonus: Think of some ways to better encourage true goals**


# How to Think of Metrics?

## "Product Sense"

Not so long ago, I thought of "product sense" as a misnomer: No one is born with such a sense and there's always so much deliberation. After thinking extensively about products for a while, though, I do feel it's a sense after all: Whenever I see a new product or design now, I can't help but wonder how it came to be, along with what metrics can be used to capture that.

I had my first epiphany when Google Maps started suggesting fuel-efficient routes. I was amazed by the thoughtfulness and then wondered, **how do we know if this cool new feature is successful**? 

{{< figure src="https://www.dropbox.com/s/jw68iz18rgjp642/metrics.jpg?raw=1" width="400">}}

To examine whether something is successful, we need to know why it was design in the first place. Then after that, I always ask myself the following questions:
1. **Adoption and usage**: ***How many people use it? Do they use it much?***  — However awesome, an obscure product or feature is hardly a success
2. **Company goals**: Okay, people use it. ***Does that serve the designer's purposes?*** 
3. **Counter mertrics** ***Are there unintended consequences?***  From time to time, I think about the [value alignment problem](https://deepmind.com/research/publications/2020/Artificial-Intelligence-Values-and-Alignment) I first heard in Alison Gopnik's AI seminar: If you build an AI that maximizes the number of paper clips made, then it will wreak havoc by turning all the metal on this planet into paper clips.

For Google Maps, fuel-efficient route suggestions hopefully give them a competitive advantage over, say, Apple Maps: iOS users should be less likely to "multi-tenant" (using both Google Maps and Apple Maps) and those who rely on Google Maps may feel free to take more trips. To test these hypotheses, we can examine changes in daily active users, trips per week per user, etc.. However, this new feature is not all perfect  — fuel-efficient routes may be way slower than the fasted routes, so we should watch out for drastic increases in time between the same two points.


If stuck, below are some "frameworks" to brainstorm metrics:

- **User journey**: Most products have a conversion funnel that moves new users towards users who frequently take "core actions" (e.g., Quora: answer questions; Notion: become a paid user) of the business 👉 Track each step in this funnel to come up with relevant metrics (and diagnose problems, if any)
- **Input/out put**: Input (or driver) metrics are actions you take or resources you put in (e.g., DoorDash's driver incentive program) and outcome metrics are results you achieve (e.g., delivery times, order lateness, customer retention)
- **Know the domain**: If you're working for DoorDash, you know 3 sides of the market each have its own needs and wants (merchants: reach + revenue; dashers: flexibility + earnings; customers: selection + convenience) so you can use metrics to track these needs and wants 

<!-- # Metric Questions

- **metric design**: come up with or choose metrics that appropriately capture the goodness of a specific feature/product → these metrics are used to measure the success + health of a product and hold the team accountable
- **metric evaluation**: proactively understand what might be wrong with a metric and how it may lead the team to misleading conclusions -->

<!-- 

# Readings
1. Lean Analytics -->