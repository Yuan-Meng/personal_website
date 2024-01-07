---
title: Autocompletion for Search Enginees
date: 2024-01-06
math: true
tags:
    - search
    - information retrieval
    - autocomplete
    - learning to rank
categories:
- papers
keywords:
    - search, information retrieval, autocomplete, learning to rank
include_toc: true
---

Autocompletion dates back half a century ago ([Longuet-Higgins & Ortony, 1968](https://www.doc.ic.ac.uk/~shm/MI/mi3.html)), initially designed to save keystrokes as people type and help those with physical disabilities type faster. The incomplete user input is the **"query prefix"** and suggested ways of extending the prefix into a full query are **"query completions"**. This feature is essential to modern text editors and search engines.

{{< figure src="https://www.dropbox.com/scl/fi/a112bhnp4bctso6fwn72x/Screenshot-2024-01-06-at-4.27.56-PM.png?rlkey=bphpjszgirskda9i142icb5os&raw=1" width="450" >}}


In this blog post, I summarize key ideas from the survey paper [*A Survey of Query Auto Completion in Information Retrieval*](https://www.nowpublishers.com/article/Details/INR-055), recommended by my teammate at DoorDash. 

# Why Autocomplete?
The "why" for autocompletion is rather self-evident because, if done right, it makes search more efficient for all parties involved: 

- **Users**: Saves keystrokes + speeds up typing + discovers relevant search terms not previously thought of 👉 less search friction
- **Search engines**: Reduces the probability of typos and ill-formed queries compared to if the user types the full query 👉 better query performance

Without autocompletion, Yahoo! users will spend roughly twice the time typing in the search bar, if they don't abandon search at all.

## A Note on Related Tasks
Autocompletion is a type of *query reformulation*, along with query suggestion, query expansion, and query correction. However, the goal and the usage of autocompletion are quite different from the latter three. The latter three take place *after* a query is submitted and aim to improve otherwise suboptimal query performance (e.g., the user didn't have the right "vocabulary" or typed something too specific). By contrast, autocompletion happens *before* query submission as users type and its purpose is to save keystrokes and ease the submission process.

# Build A Simple Autocompletion Engine

The "million-dollar question" is the "how" --- how do we know what the user meant to type without them typing the whole thing? 

{{< figure src="https://www.dropbox.com/scl/fi/8raqtvfpf88y71yjdc9zk/Screenshot-2024-01-06-at-6.16.17-PM.png?rlkey=vkjii3k3q8rqqaagu3ctwtp72&raw=1" width="650" >}}

## Retrieval
[Tries](https://en.wikipedia.org/wiki/Trie) ("prefix trees") are a common data structure to store associations between a prefix and its query completions. A trie can be pre-built from the past query log, which will be used to retrieve a list of query completions given a prefix. 

## Ranking
Which completions to show in what order can be determined by **heuristics-based approaches** as well as **learning to rank (LTR) models**. Most search engines only show a handful of completions --- the user's intended query should appear at the top.

### Heuristic-Based
Successful completions are ones that eventually get submitted by the user. Heuristic-based approaches rank each query completion $q_c$ by the probability that it might get submitted, given the prefix $p$, the time $t$, and the user $u$ --- $P(q_c | p, t, u)$.

- **Popularity**: Some completions appear more frequently than others. Frequency could be from past searches or future searches predicted by time-series models.
    - **Short vs. long windows**: Search may trend in the short term (i.e., a star's name trends after a movie release) or have long-term cyclic patterns (e.g., people search "NeurIPS" more in May and December, prior to the submission deadline and the conference date, respectively). Combining both trends lead to better completions than only considering either one. 
- **User**: You and I may mean very different things when typing "cat" (e.g., I'm buying "cat food" for my kids 🐱 whereas you're looking up your aunt Cathy) --- a user's past searches, previous queries in the same session, and engagement with query suggestions can be used to personalize their query completions.

### Learning to Rank
In document retrieval (DR), learning to rank (LTR) refers to learning a function $f(q, D)$ that scores a list documents $D = \\{d_1, d_2, \ldots, d_n\\}$ given a query $q$. In query autocompletion (QAC), documents are not yet in consideration; instead, we learn a function $f(p, Q)$ that scores a list of queries $D = \\{q_1, q_2, \ldots, q_n\\}$ given a prefix $p$.

Labels for QAC LTR are simpler than the usual 5-point relevance scale ("Bad", "Fair", "Good", "Excellent", "Perfect") used for DR LTR --- users show *binary* preferences towards query suggestions by "submitting (1)" vs. "not submitting (0)". The table below summarizes different features, labels, and evaluation metrics used for DR LTR vs. QAC LTR, two similar tasks each with their own peculiarities.

{{< figure src="https://www.dropbox.com/scl/fi/avkov0mnxdyw8vbpbw5y5/Screenshot-2024-01-06-at-8.36.54-PM.png?rlkey=73zcq6c6034maoq0xz9ggh6ar&raw=1" width="600" >}}

The table below shows example data for training DR LTR vs. QAC. The format is highly similar between the two. As we can see, each prefix (grouping ID) is observed with multiple query completions, each with a feature vector and a binary label.

{{< figure src="https://www.dropbox.com/scl/fi/vys6354sbbm0cdqfxhyj0/Screenshot-2024-01-06-at-9.39.03-PM.png?rlkey=cke9l5u1fer3h69e5bhicvvdb&raw=1" width="600" >}}

Below are some key ranking signals (many shared by heuristic-based approaches):
- **Popularity features**: Observed popularity from historical searches; future popularity predicted from recent trend and cyclic behavior
- **Semantic features**: Users might want query completions that are *similar* (however defined) to submitted queries in the same session. 
- **User features**: User features such as location, demographics, search history, past reformulation behavior, etc. can affect query completions.

## Evaluation
A search engine can complete queries however we want, but how do we know if the list of query completions returned to users is any good? On an abstract level, a good list is one where the user's **intended query** is ranked at the top. 

Differently from document ranking, query suggestions are mostly concerned with finding a single best solution. Moreover, all else being equal, better suggestions save more keystrokes. The metrics reflect these considerations.

- **Mean Reciprocal Rank (MRR)**: On average, how early the 1st relevant completion appears? The formula is $\frac{1}{P}\sum_{p = 1}^{P}\frac{1}{\mathrm{rank}_p}$, where $p$ is a specific prefix, $P$ is the eval prefix set, and $\mathrm{rank}_p$ is the rank of the 1st relevant query for $p$.
- **Success Rate at Top K (SR@K)**: Across the eval prefix set, \% of time the intended query can be found in top $K$ query completions.
- **_pSaved_**: The probability of using a query suggestion while submitting a query.
- **_eSaved_**: The normalized amount of keystrokes saved due to query suggestions.
- **Diversity**: Redundant query suggestions (e.g., "apple pie" and "apple pies") should be penalized. One example diversity metric is $\alpha$-nDCG, which assigns a higher gain to each additional query with more "new aspects".

# Performance Improvements
Above is the bare bone of an autocompletion engine, which can be further improved:
- **Computational efficiency**: Running BFS/DFS on vanilla tries may result in slow performance, especially given the vast query space for each prefix. We can optimize the trie data structure itself (e.g., Completion Trie, RMQ Trie, Score-Decomposed Trie, etc.) or the search algorithm (e.g., A* search).
- **Error-tolerance**: Users may have misspellings in the prefix --- spell-checking or fuzzy match is needed to ensure misspelled queries can have completions.
- **Multi-objective**: In e-commerce search, after a user submits a query completion, we ultimately want them to convert on a product. A multi-objective model also optimizes for subsequent engagement (e.g., clicks, add to cart, conversions) following query submission. This was described in an [Instacart post](https://www.instacart.com/company/how-its-made/how-instacart-uses-machine-learning-driven-autocomplete-to-help-people-fill-their-carts/).

# Presentation of Completions

Last but not least, autocompletion needs to be represented to users. The canonical way is to display completions as a vertical list below the search bar. If you're highly confident about the top completion, you may put it directly in the search bar, a method called "ghosting" ([Ramachandran \& Murthy, 2019](https://www.amazon.science/publications/ghosting-contextualized-query-auto-completion-on-amazon-search#:~:text=Ghosting%3A%20Contextualized%20query%20auto%2Dcompletion%20on%20Amazon%20Search,-By%20Lakshmi%20Ramachandran&text=Query%20auto%2Dcompletion%20presents%20a,i.e.%2C%20within%20the%20search%20box.)). In either case, the completed text is usually highlighted we can distinguish it from the user input. 

{{< figure src="https://www.dropbox.com/scl/fi/dvtlbvufymae0zn9zhggh/Screenshot-2024-01-06-at-10.25.44-PM.png?rlkey=olkouuxlhgsv39as3lvipcdos&raw=1" width="600" >}}


Compared to document retrieval, search engines have even more limited real estate to show autocomplete suggestions and a stricter requirement on the position of the intended query (i.e., even if the intended query is included in the list, it only gets clicked ~13\% of the time if ranked below top 2, [Li et al., 2015](http://www.yichang-cs.com/yahoo/SIGIR15_QAC.pdf)). In other words, precision matters even more in query suggestions than document retrieval.

Rather than simply organizing results by submission probability, sometimes it makes sense to organize by intent (see the graph below) or the nature of the prefix (e.g., if a prefix is a geolocation, completions can be ordered by region).

{{< figure src="https://www.dropbox.com/scl/fi/snj71a8tgxp1x6xy1pxor/Screenshot-2024-01-06-at-10.41.17-PM.png?rlkey=9mzmbtk4olk8950epfpaycl6c&raw=1" width="450" >}}