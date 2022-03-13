---
title: (Quirky) Roadmap for New Grad Data Scientists 
date: 2022-03-13
math: true
tags:
    - career
    - data science
    - interview 
categories:
- advice
keywords:
    - career, data science, interview 
---

# New Grad Timeline
Since last November, I regularly get questions about how to get data science jobs. I've hesitated to give advice because no advice applies to everyone. As more people ask, however, I want to write a post for those in my shoes: **(PhD) students applying to ML DS or product DS roles at tech companies through university recruiting.** 

## What's special about university recruiting
Every year around August, you'll start to see openings with "New Grad", "University Grad", "Masters/PhD Graduate", etc. in the job titles. Take advantage of them if you're a current student. University recruiting is special in several regards: 
1. **Companies don't expect you to start right away.** If you're 3+ months away from graduating, you likely won't be considered for "normal" positions. Only to new grad roles can you apply more than half a year before you start. (Data points: Recruiters at Waymo and Amex reached out to me in June 2021. When I told them I graduate in May 2022, they said they'd only consider those starting soon-ish.)
2. **Most companies don't have a new grad track for DS.** Usually, only big tech companies (e.g., Meta, Google, Uber, Microsoft, Robinhood, TikTok, DoorDash, Adobe, Pinterest, Zoom, PayPal), major hedge funds (e.g., Two Sigma), and established startups (e.g., Figma, Quora, Faire, Nuro) do. A bit to my surprise, Apple, Amazon, and Airbnb don't really hire new grad DS and Lyft seems to fill new grad roles elusively with return interns. 
3. **The hiring bar is different.** You're expected to only have school and internship experience; your resume and performance are compared to other students. So for students, university recruiting may be the path of least resistance.

## Average-case scenario for Spring grads
Say Bob graduates in Spring 2025 (mid-May at most universities) and wants to apply through university recruiting. We can reverse-engineer an "ideal" timeline:

{{< figure src="https://www.dropbox.com/s/6rxdza3h29an17n/timeline.png?raw=1" width="600" >}}


## Caveats
If you graduate in December 202X, your timeline is similar to those graduating in Summer (202X + 1), except that you miss the last Spring cycle. Some caveats:
1. **You don't have to apply through university recruiting**. If you're graduating in $<$ 2 months or already graduated, you can apply to regular entry-level jobs. You'll have far more options but may face more fierce competitions (other candidates might have worked for 1-2 years).
2. **Better late than never**: Bob's timeline is an average-case scenario (best case: got a return offer and don't need to interview at all). What if you start preparing or applying "too late", like I did for my internships? For [various reasons](https://www.yuan-meng.com/posts/nothingness/), I only started applying to Summer 2021 internships in late February 2021 and got an offer in the eleventh hour of mid-May. It makes your odds worse, no doubt, but better late than never (as that RedHook song goes, *"These bad decisions haunt me, but they make good stories"*). 
3. **Everyone's background differs**: Steps 2-4 in are pretty set, but how long does it take to build an attractive resume for internship apps (step 1)? It depends on what you already know and have done. If you're well-versed in scientific computing (stats, ML, programming) and experiment designs and have DS projects you're proud of and ready to show, you can just polish your resume and apply. If you've never trained models, run experiments, or analyzed data, it may take 1-2 years to learn and do stuff. Most cases are somewhere in between (e.g., most STEM students know stats quite well but don't use SQL on a regular basis). 

# Track #1: Machine Learning

## **TL;DR**
> If you want a career in ML, nowadays you likely need to master state-of-the-art models (Research Scientist, Applied Scientist, or Machine Learning Scientist), software engineering (Machine Learning Engineer or Software Engineer, Machine Learning), or both --- the days when data scientists could simply hand an out-of-box model in a Jupyter notebook to an ML engineer to productionize may be gone at many tech companies. 


## 500 names of ML data scientists 🦄

Master's degrees, bootcamps, and online courses prepare you to be an ML data scientist, which is stereotypically what an DS is. That may be true in tech years ago and still true in traditional industries (e.g., pharmas, banks, telecommunication, etc.). Yet, at major tech companies today, ML data scientists are unicorns. I heard a saying that this change might have begun with Meta's "Data Scientist, Analytics" track, which fully separates responsibilities of data scientists (A/B testing + SQL) and machine learning engineers (model training + deployment). Other tech companies have followed suit and hire data scientists to do pure product analytics. 

ML data scientists are rebranded into different roles at big tech companies, often with higher expectations in research or engineering. Companies like Uber, Amazon, and Microsoft have **"Applied Scientist"** roles that typically target Ph.D. grads who have strong ML skills but may come from non-ML fields (e.g., econ, EE, IEOR, cognitive science). Depending on the company and the team, applied scientists may or may not write production code. Research-oriented ML roles that don't regularly touch production code are called **"Research Scientist"** or **"Machine Learning Scientist"**. These roles typically target Ph.D. grads from relevant fields with decent publications. Prestigious AI labs (e.g., DeepMind, FAIR, OpenAI, Google Brain) may have expectations on a par with faculty hires. At companies like Lyft and sometimes Amazon, the line between research and applied scientists is somewhat blurred. 

As is more often the case, if you train models, you're also expected to deploy them. These roles are called **"Machine Learning Engineer"**  (e.g., Twitter, Adobe, Stripe) or **"Software Engineer, Machine Learning"** (e.g., Google, Waymo, Quora). ML engineers on different teams at different companies can do wildly different things, from DL research, [MLOps](https://databricks.com/glossary/mlops) (e.g., modeling training, deploying, and serving), to ML infrastructure. In all cases, you're essentially an ML data scientist and a software engineer meshed into one, just the fraction of each component may differ.

Some major tech companies still offer the **"Data Scientist, Machine Learning"** title to new grads. You're generally not expected to deploy models or do theoretic research (the [Machine Learning](https://doordash.engineering/category/data-science-and-machine-learning/) section in DoorDash's engineering blog has examples of what their ML DS do). When I applied in Fall 2021, I only saw 3: Robinhood, DoorDash, and Figma. There are of course more I missed. For instance, Airbnb has a well-known **"Data Scientist, Algorithms"** track but didn't hire new grads in my year.

## Resources/preparation for ML DS
Regardless of the title, an ML practitioner can't escape any of the following, IMO: 
1. **ML/DL foundations**: You should know how learning algorithms work in general (loss, optimization criteria, optimization routine) and have intuitions about common algorithms in classic ML (regression-, instance-, and tree-based models) and DL (multilayer perceptrons, CNN, RNN, transformers...). It's good if you understand their mathematical foundations. At a bare minimum, you should be able to use existing frameworks and libraries to train, tune, and test models. 
    - **Quick & dirty**: [The Hundred-Page Machine Learning Book](http://themlbook.com/) ([my notes](https://yuanm.notion.site/core-ml-1930f2267ce942c984b005c1bb62d429))
    - **Practical**: [Machine Learning with PyTorch and Scikit-Learn](https://www.amazon.com/gp/product/1801819319/ref=ppx_yo_dt_b_asin_title_o02_s00?ie=UTF8&psc=1), [Deep Learning for Coders](https://www.fast.ai/), 3Blue1Brown's [Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) series
    - **Foundational**: [The Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/), [Machine Learning: A Probabilistic Perspective](https://www.amazon.com/Machine-Learning-Probabilistic-Perspective-Computation/dp/0262018020), [Deep Learning](https://www.deeplearningbook.org/) by Ian Goodfellow

    If I were to start over and can only pick 2, I'd first read the 100-page book and then work my way through *Machine Learning with PyTorch and Scikit-Learn*. 

    People learn differently, but in general, you don't wanna spend 3 months just reading or, reversely, only call Scikit-Learn packages without understanding. Every time I learn a new family of models (e.g., all sorts of boosting, graph embedding, transformers), I'd read a book chapter or a seminal paper on it, watch several YouTube videos explaining the intuitions, and apply it to a toy problem I care about (end product is a notebook, an app, or a presentation). If I get stuck, I'd Google my way out (which often leads to Kaggle kernels, Medium blogs, or Analytics Vidhya articles) or ask questions on StackOverflow. Getting stuck is no pleasant feeling, but I never forget Prof. Aaron Fisher's words, 
    > *"Learning is pain. If you don't feel the pain, you're not learning."*
3. **ML system design**: You need to know how to apply ML to solving business problems (e.g., how Twitter shows feed, how Google ranks results, how Uber matches riders to drivers, how Stripe detects fraudulent transactions). Unlike software engineering system design, ML system design focuses more on the end-to-end model training process and less on the tech stack. 
    - **General approach**: [Introduction to Machine Learning Interviews Book](https://huyenchip.com/ml-interviews-book/) covers all grounds in ML interviews ([Discord](https://discord.gg/XjDNDSEYjh) for discussing answers). [Rules of Machine Learning: Best Practices for ML Engineering](http://martin.zinkevich.org/rules_of_ml/rules_of_ml.pdf) and [Design a Machine Learning System](https://huyenchip.com/machine-learning-systems-design/design-a-machine-learning-system.html) are less comprehensive but great for a quick grasp.
    - **Specific systems**: [Grokking the Machine Learning Interview](https://www.educative.io/courses/grokking-the-machine-learning-interview) 👉 This Educative course covers 6 common types of ML systems, including self-driving, ads, search ranking, recommendation, feed, etc.. I think it's great for interview preparation and guiding personal projects. 
    - **Cool tricks**: [Machine Learning Design Patterns](https://www.amazon.com/Machine-Learning-Design-Patterns-Preparation/dp/1098115783) 👉 I got this gem coincidentally (I was too early for a movie and spotted it in a bookstore) but it has guided me through many tricky situations. Examples: How do you deal with high-cardinality categorical data (e.g., airport names)? What to do if one observation can have multiple labels (e.g., hashtags of Tweet)? How can we combine tabular, text, and image data to make predictions?... 
4. **Data structures & algorithms**: As mentioned, ML roles have engineering components and therefore often require SWE-style coding interviews. The bar may differ slightly by job title (often but not always: MLE & SWE > AS > RS $\approx$ DS). 
    - **Foundations**: It doesn't matter how you learn it, but you need basic data structures and algorithms knowledge before you grind coding questions. 👉 Resources (you don't need them all): [CS Dojo](https://www.youtube.com/watch?v=bum_19loj9A&list=PLBZBJbE_rGRV8D7XZ08LK6z-4zPoWzu5H), [CS 61A](https://cs61a.org/), [Problem Solving with Python](https://problemsolvingwithpython.com/) @Berkeley, [Practical Algorithms and Data Structures](https://bradfieldcs.com/algos/)
    - **LeetCode**: I initially refrained from applying to MLE jobs because I didn't want to grind LC questions. Later, I realized several roles I started interviewing for (AS @Uber, SWE @Google, quant @Two Sigma) would require LC-style coding interviews. So I signed up for 九章算法 in October 2021, about halfway through my interview cycle, and was really burnt out. If you want to reduce stress and not limit your choices, I strongly recommend that you **seriously invest in coding interviews the summer before you apply to full-time jobs**. Upon hindsight, it wasn't that scary because on the PhD-level at least, coding interviews for MLE/AS/RS/DS are not meant to be hard. Most of my friends did about 200-300 questions when they interviewed. 

        It doesn't matter if you grind 200 or 1,000 questions --- the end goal is that you can quickly recognize patterns in new problems you got, work collaboratively with interviewers to gradually solve them, and clearly communicate your thoughts and justify your approach along the way. 

        For Mandarin speakers, taking [九章算法](https://www.jiuzhang.com/course/71/) is the most efficient way to learn patterns that apply to almost all algorithm questions. The lectures and course support are amazing. The English counterpart, [Grokking the Coding Interview](https://www.educative.io/courses/grokking-the-coding-interview), is in a text format and doesn't have teach assistants. 

        {{< figure src="https://www.dropbox.com/s/7oi1wyxnkpwnu0y/lc.png?raw=1" width="400" caption="VS Code users can use the [LeetCode plugin](https://developpaper.com/the-tutorial-of-configuring-leetcode-plug-in-in-vscode/) to grind LC locally." >}}


Between the day I first imported a Scikit-Learn model (October 2019) and when I got an offer as an ML data scientist (November 2021), 2 years had passed. In those two years, I did dozens of toy projects, deployed 4-5 personal projects, and worked on a researchy DL project in my internship. There's a reason why people joke about ML being "modern fortune-telling": From data validation + wrangling, feature engineering, to modeling selection, tuning, evaluation, and deployment, should anything go wrong, predictions would be nonsense. ML is a craft to patiently cultivate. 
 

# Track #2: Product Analytics

## **TL:DR**
> At big tech companies and established startups, if the job title "Data Scientist" is not suffixed by "Machine Learning", "Algorithms", or "Engineering", it most likely refers to product analysts. The analytics track is by far the most common way for new grads (or anyone, for that matter) to land a DS job. In this quirky roadmap, shall I offer some quirky advice: **All new grads looking for DS jobs, regardless of the track, should learn product knowledge**, because 1) you can't count on getting an ML offer given the rarity, and 2) product sense goes a long way in ML interviews as well (a model is only useful insofar as it solves a business problem and you need product sense to understand what it is). 
 

## Ubiquity and range of product DS
While writing this post, I searched "Data Scientist + New Grad" on LinkedIn and below is the first result shown to me, from Wealthfront:

{{< figure src="https://www.dropbox.com/s/cyye15hiyg4mtlx/ds.png?raw=1" width="450" >}}

This fits a typical product DS role that requires **"A/B testing + SQL + something"**. In this case, that "something" is automating their internal analysis tools. In other cases, "something" could be a bit of machine learning (e.g., Quora), causal inference (e.g., Netflix, which hires DS interns but not new grads), and so on.

## Resources/preparation for product DS
Between the day I first heard "product sense" (August 2021) and when I got my first product DS offer (October 2021), it took me about 2 months. At its core, product analytics hinges on the **scientific method**, which is ingrained into PhD training in empirical disciplines. With regard to the [timeline](https://yuan-meng.com/posts/newgrads/#new-grad-timeline) --- you may not need a whole year before applying to product DS internships if you're an experimental scientist. 

Many people ([Emma](https://www.youtube.com/c/DataInterviewPro), [课代表立正](https://www.youtube.com/c/%E8%AF%BE%E4%BB%A3%E8%A1%A8%E7%AB%8B%E6%AD%A3)) have talked about product interviews. So instead of repeating what they said, I'll mention what I may or may not have done differently.

1. **A/B testing**: I did 3 things all product DS candidates did --- reading [Kohavi's book](https://www.amazon.com/Trustworthy-Online-Controlled-Experiments-Practical/dp/1108724264/ref=sr_1_1?qid=1647057225&refinements=p_27%3ARon+Kohavi&s=books&sr=1-1), watching Emma's videos, and taking the [Udacity course](https://www.udacity.com/course/ab-testing--ud257). These are by far the best things you can do to prepare. In addition:
    - **Find a dataset to analyze**: Unlike ML, there aren't many public datasets for A/B testing. In their [NeurIPS paper](https://arxiv.org/abs/2111.10198), Liu et al. (2021) summarized a dozen online controlled experiment datasets with different levels of granularity. Read this paper, pick a dataset, and analyze the experiment. Chances are, this process may expose gaps in your knowledge (How to load, join, and aggregate data? Which assumptions to check? What test to perform?) that merely reading a book or watching videos won't do.
    - **Implement power analysis and $t$-tests**: While reading the Kohavi book, I didn't understand the sample size formula, $\frac{16\sigma^2}{\Delta^2}$, so I Googled how to derive it and [implemented]($\frac{16\sigma^2}{\Delta^2}$) different variations. For $t$-tests, I similarly implemented different variations (comparing 2 means vs. 2 proportions, pooled vs. unpooled variances) from scratch using NumPy and SciPy. 

        I know myself --- I'm awful at memorizing random stuff (e.g., I don't recall my own plate number) and have to make sense of something to remember it. IMO, it doesn't matter how you choose to learn, but you need to be honest about whether it's working or not and do whatever works for yourself.
2. **Stats**: Most product DS stats questions are packaged with the "A/B testing combo" (sample size calculations, $t-$tests, etc.). Additional questions may cover regression (linear, logistic), sampling (e.g., stratified vs. random sampling), and resampling (permutation tests, bootstrapping). Try implementing [the most used (frequentist) statistical tests](http://kordinglab.com/2016/01/02/summary-stat.html) yourself in Python (SciPy) or R.
    - **Build intuitions**: If your stats knowledge is rusty, read through [OpenIntro](https://www.openintro.org/), which covers axioms of probability, distributions, $t-$tests, OLS, etc.. StatQuest has an amazing [statistics fundamentals](https://www.youtube.com/watch?v=qBigTkBLU6g&list=PLblh5JKOoLUK0FLuzwntyYI10UQFUhsY9) series. I watched them at the beginning of the pandemic (didn't want to watch shows I used to watch with someone else...), which got me hooked by statistics. 
    - **Learn the nitty-gritty**: I'm stuck in a weird place where I'm dissatisfied by practical books that don't explain why and how methods come about but also don't feel patient enough for theoretical books filled with proofs. After watching Prof. Simon DeDeo [derive](https://youtu.be/6YEn9QRy3ks) the NYC cab waiting time distribution from the principle of maximum entropy, I was hoping to see a book that derives distributions and tests from first principles and conveys intuitions clearly. I've yet to find that book. Some good ones are [Introduction to Probability for Data Science](https://probability4datascience.com/), [Mathematical Statistics with Resampling and R](https://www.amazon.com/Mathematical-Statistics-Resampling-Laura-Chihara/dp/111941654X/ref=sr_1_1?crid=2GCVC4PMQYOEW&keywords=Mathematical+Statistics+with+Resampling+and+R&qid=1647150832&sprefix=mathematical+statistics+with+resampling+and+r%2Caps%2C106&sr=8-1), and of course, [The Elements of Statistical Learning](https://www.amazon.com/Elements-Statistical-Learning-Prediction-Statistics/dp/0387848576).
3. **Probability**: Most probability questions have to do with probability distributions and combinatorics (permutation and combination). Meta has one round of onsite interview dedicated to this kind of questions. Other companies may have one or two buried in their stats round or technical phone interviews. 
    - **Know your distributions**: Gaussian and binomial are by far the most common distributions you'll be asked about, and sometimes heavy-tailed distributions. The geometric distribution is also common, especially in churn analysis and customer lifetime value (LTV) calculations. Almost every probability theory or stats book has one chapter or two talking about distributions. I somehow particularly like the [OpenIntro](https://www.openintro.org/book/os/). 
    - **Know your combinatorics**: In the case of discrete probability, you can derive most answers from 2 rules, the **rule of product** (if it takes $n$ step to do a thing and there are $m_i$ ways to do it at step $i$, you have $\prod_{i}^{n}{m_i}$ ways to complete the whole thing) and the **rule of sum** (if there are $n$ mutually exclusive things and $m_i$ ways to do thing $i$, there are $\sum_{i}^{n} m_i$ ways to do the $n$ things). I highly enjoy this [tutorial](https://brilliant.org/wiki/combinatorics/) from Meta recruiters.
    - **Practice beforehand**: The thing is, probability questions can be confusing and unintuitive. If you know how to derive the answer slowly, that may not cut it for time-limited interviews. It's reassuring to have worked on similar problems before the interview. The "[quant green book](https://www.amazon.com/Practical-Guide-Quantitative-Finance-Interviews/dp/1438236662)" covers most probability questions under the sun, but many questions are far too hard for DS interviews. The ["DS red book"](https://www.acethedatascienceinterview.com/) may be a less overwhelming choice. When I was interviewing, I just Googled "probability questions for data science interviews" and practiced some questions. 
4. **SQL**: Even though SQL is a core skill of product DS, I somehow only did SQL interviews with one company (Meta). Someone else may offer better suggestions. 
    - **For absolute beginners**: [Select Star SQL](https://selectstarsql.com/) is a wonderful tutorial created by a former Quora data scientist. Among all the good things, I particularly appreciate how it explains the way SQL scans tables and processes queries. Later when working on head-boggling problems, I'd manually transform data on a piece of paper, the key lesson I learned from this tutorial. 
    - **Learn window functions**: If you're focusing on the analytics track, you'll run into window functions sooner or later. I attempted to get away with learning but realized that for many problems, if you don't use them, you must compensate with a sh\*t ton of JOIN's, CTE's and, God forbid, nested queries... I recommend the [window function course](https://learnsql.com/course/window-functions/)on LearnSQL. I tried a bunch of other ones, which just didn't click. Here are my [notes](https://www.notion.so/window-functions-0a792ad76e90400d9381df0931e0c990). 
    - **Practice out loud**: I know how to answer algorithm questions in front of a human but didn't know that about SQL questions. After watching Nate's [YouTube videos](https://www.youtube.com/watch?v=GeJUvdkJKEc&list=PLv6MQO1Zzdmq5w4YkdkWyW8AaWatSQ0kX), I realized there's not much of a difference --- just like in the former, you clarify the goal of the problem, confirm the input/output (especially edge and corner cases) with the interviewer, think silently/frantically for a minute or two, then verbally walk the interviewer through your step-by-step approach, and finally code it up after  the interviewer green-lights your idea. When you practice SQL questions on LeetCode or [StrataScratch](https://www.stratascratch.com/) (created by Nate?), treat it as an interview and explain your approach out loud before writing any queries. It's also a good idea to time your solutions to create realistic pressure (easy: 5 min; medium: 10 min; hard: 15 min). 

5. **Product sense**: There's so much said about product sense already and I wrote a [post](https://www.yuan-meng.com/posts/metrics/) about choosing metrics myself. To some degree, I think **product sense (at least metric investigation) is about recovering finer-grained information from highly aggregated statistics using knowledge about the business and the users**. A quick example is that, when Meta aggregates the number of posts per user across all users and all time, of course you lose information about different user segments and time periods. Then when this aggregate metric goes wrong (dropped by 20%!), can you tell why? Of course not, because the aggregation wiped out more granular details. But if you know something about your users (e.g., new users tend to post less), the context (e.g., a major event had passed), etc., then you can make an educated guess about how to "disaggregate" the data (e.g., `GROUP BY user_age`) and look again. If I were to design product interviews, I'd follow up the candidate's guesswork with a SQL question (*"Okay, you wanna look at the trend across time. How to write the query?"*)

    - **Learn the "classics"**: If you're just getting started, read [*Lean Analytics*](https://www.amazon.com/Lean-Analytics-Better-Startup-Faster/dp/1449335675) and watch Emma's [product case videos](https://www.youtube.com/watch?v=QJat_nicj9c&list=PLY1Fi4XflWSvtu963rZpfH6WeX54vSrDW) before you do anything else. In most cases, these are already enough. I skimmed *Cracking the PM Interview* and *Decode & Conquer* --- there's good stuff here and there, like competitor analysis and behavior questions, but I don't remember anything else. 
    - **Picture yourself as the investor**: Everyone says picture yourself as the user, which is great advice: You should use the product made by the company you're interviewing with if you can. But what if you're interviewing with Waymo, SpaceX, or a 2B company? You can picture yourself as the investor. If a company is public, I'd read the earnings calls to see the business' strategy and focus at the moment, how various products work together to achieve its goals, which key metrics the company chooses to report, and what concerns investors raised. Below is a screenshot from DoorDash's 2021 Q4 financial report, which is publicly available on their [investor relations page](https://ir.doordash.com/news/news-details/2022/DoorDash-Releases-Fourth-Quarter-2021-Financial-Results-4afccdf68/default.aspx). As you can see, one of their main growth strategies is to find more ways to be useful in people's lives (e.g., deliver groceries and stationery). You can also see that while DashPass helps retain customers and encourage ordering, the company needs to strike a delicate balance between order frequency and profit per active user.
    {{< figure src="https://www.dropbox.com/s/xeb151efi1ry7f3/DASH.png?raw=1" width="500" >}}
    - **Mock interviews**: Had it not been for mock interviews, I literally wouldn't have gotten a job --- my resume was initially passed by DoorDash; I joked to a mock interview buddy that I'd order from somewhere else now, was referred to the same job, and got an interview next day. "Concrete benefits" aside, it's super useful to hear how different people approach open-ended cases and verbalize your thoughts socially and interactively. 

        A perennial question is how many mock interviews one should do. It doesn't matter; some do dozens while others three --- it's the end result you strive for. Ask yourself, given any arbitrary product, can you quickly ask the right questions in order to figure out how it works, who it serves, when it succeeds, and how it might fail? Given example results, can you gauge how well it's doing? When goals clash, can you seek a good trade-off?... If these thoughts and ideas come natural to you, then doing more only gives you a diminishing return. On the other hand, if people often say you need to think more thoroughly and organize your thoughts more clearly, then you'd probably benefit from practicing more with people. Do what you need.

# Interview Preparation 

## Knowing the basics $\gg$ memorizing answers

> "If I am allowed to give only one suggestion to a candidate, it will be **'know the basics very well'**." --- Xinfeng Zhou, *The "Quant Green Book"*


I wholeheartedly agree with Xinfeng Zhou's views on interviews. I think **good interview preparation consolidates your broad knowledge base accumulated over time and brings out the logical, insightful, and methodical person that you are**, rather than helping you memorize answers to sample interview questions you don't understand or pretend to be someone you're not. Sure, even if you know the basics well and normally think logically, you may still flunk interviews if you're nervous or unprepared. However, if you're interviewing for a product role but can't think of ways to analyze a product, or interviewing for an ML role but can't choose between linear regression and neural nets, no amount of interview skills will help.

You might be interviewing with certain companies famous for re-using product analytics questions and some people do get offers just by memorizing answers. Sure, practice those questions but don't skip the basics. What if you end up having to interview with other companies? What if the questions change? Even if you got the same questions you prepared for, any lack of understanding would still show: 

> "Unless you truly understand the underlying concepts and can analyze the problems yourself, you will fail to elaborate on the solutions and will be ill-equipped to answer many other problems." 

<!-- Take this last part with the most grain of salt. Everyone has their own process and different amounts of time to spare. I was kinda lucky that my PhD advisor agreed to me taking time away from research in September and October 2021. The professor I was teaching stats with was also very understanding. So during those two months, I did nothing but interviews. I realize not everyone can commit this much (full-time jobs, family...). That said, I'll try to pick out some generalizable bits.  
 -->


## Apply early and get ready gradually

{{< figure src="https://www.dropbox.com/s/1fm8y5e5tjpj5vy/interviews.jpg?raw=1" width="550" caption="A subset of books I used to prepare for DS interviews. In case you're curious: Each folder has all the files of each company I interviewed with." >}}


If I devoured all the books above before I applied, I'd never be ready. For me and many friends, upcoming interviews are the only force to push things forward. During my last week of internship, a good friend interning at Meta said, *"I'm leaving next week so I can only refer you now!"* I was planning on taking some days off the week after but instead sent out my resume and started my interview cycle unprepared.

Back in August 2021, I had no idea what product analytics is. I'd forgotten the basic SQL I learned. I checked Glassdoor and saw that even Meta recruiter asks simple SQL and product questions, which is highly unusual for recruiters. I spent a few days going over Select Star SQL. I also asked a product DS friend how to approach product questions and got some generic advice, *"Think about the goal. Think about what data to pull. Think about when the data says you achieved the goal or not."* These might seem crude now but were enough to get me through that 15-min call. 

Right after getting Meta's phone interview, I heard from Quora. I was glad both are product roles at social media companies, so I didn't have to prepare wildly different stuff. I searched "product interviews" on YouTube and watched a bunch in a row: User growth by Meta's [Alex Schultz](https://youtu.be/URiIsrdplbo), metrics by Quora's [Adam D'Angelo](https://youtu.be/zsBjAuexPq4), and, of course, videos from [Emma's channel](https://www.youtube.com/c/DataInterviewPro). I got Kohavi's book recommended by Emma and dug out *Lean Analytics* a friend gave me. I read both in two days because I was so very intrigued: As a kid, I loved detective novels to the core; product analytics seemed just like detective work --- something went wrong with your product so you peel the onion to find the culprit. I thought of checking if Emma offered mock interviews and was happy that she did. Otherwise, my interview outcomes would've been very different (e.g., I thought it was a great idea to say I know her frameworks and was told that'd be a sure fire way to bomb an interview), and perhaps my life, too.

I passed both interviews in early September and was moved forward to onsite. Around that time, I started to hear from other places (e.g., Two Sigma, Figma, Uber, Google, Robinhood, and some I don't remember). Looking back, all of them are highly competitive, but I never thought about my odds. I was kinda "myopic", thinking I'd learn just enough to pass 45-minute phone interviews, and then I'd worry about pulling off 4-hour onsite interviews. Only much later did I learn many would check applicant counts and felt discouraged by the seeming impossibility of being chosen. My advice: Don't. Again, if you're shaky on the basics, you won't get the job "up against" only one other person; you'll get the offer if you truly understand how things work and can give crystal clear explanations, even if 2,000 applied.

In mid-September, I got a few more onsites. Another good piece of advice I got from Emma is that you don't have to schedule interviews right away if you're not ready. Recruiters get paid only when you accept an offer (usually $\propto$ your compensation), so they're motivated to help you get ready. I was worried about headcounts at Quora and Figma since there might not be as many openings as would be at larger companies. I asked both recruiters explicitly and figured out a timeline together:

> "Before I provide my availability, I wonder what would be an ideal date range on your end. Is preferable that I interview in late September, or is early to mid-October still a good time? Asking since I wish to be as prepared as possible but don't want to miss out on this opportunity (which I really appreciate) because of headcount-related issues.""

I got myself about a month's time to prepare for the onsites. Companies usually do onsite prep calls and share materials telling candidates exactly how many rounds there will be and what each round entails. Piece by piece, I prepared for each round at each place and felt ready a day or two before each onsite (4-5 rounds).

## Organize your time and learning


As interviews piled up and the content diversified, I was overwhelmed by stress (couldn't sleep or eat well; constantly irritated...). So I made a risky move, asking my PhD advisor for a month off from research, 

> "What do you think about me postponing experiments till November to focus on interviews? If I get an offer, I can squarely focus on my dissertation from then on. If I don't, I'll still come back to research right away. It's risky, but I think the expected utility is higher than if I spend several months doing full-time research and interviewing unprepared."

to which she generously agreed. My October schedule alternated between two modes:
1. **Review a topic** (70\%, $>$ 3 days from onsite): For each type of interview, I created a [Notion](https://www.notion.so/) page and spent 3-7 days filling it with content. Several were created back in September (e.g., A/B testing, SQL, product) and most in October (e.g., behavior, Pandas, ML design, probability + stats, applied data). Some of the best ones were later turned into blog posts and YouTube videos.

    {{< figure src="https://www.dropbox.com/s/sw3yngowl0l8y3i/topics.png?raw=1" width="350" >}}

    For procedural knowledge like SQL and Pandas, I usually Google around for a good problem set (e.g., Pandas: [`pandas_exercise`](https://github.com/guipsamora/pandas_exercises), SQL: [StrataScratch](https://www.stratascratch.com/)) to work through. Afterwards, I'd reflect on how I like to approach an arbitrary problem during interviews, extract patterns (e.g., when to use window functions), and analyze problems that I not only got wrong but still found challenging. 

    {{< figure src="https://www.dropbox.com/s/t1fxd5afaaq6sqg/sql.png?raw=1" width="600" caption=" A screenshot of my SQL notes. Advice was given by Nathanael Rosidi.">}}

    For probability and stats, I flipped through several textbooks (I was teaching stats) to create an interview "cheat-sheet", including properties of common distributions, formulas and assumptions of common tests (e.g., $t$, $F$, $\chi^2$), regression (OLS, $l_1$, $l_2$), and major concepts (central limit theorem, the law of large numbers, $p$-values, Bayesian HDI vs. frequentist confidence intervals...).

    For the "analytics combo" (A/B testing + product cases + metrics), I re-read *Lean Analytics*, skimmed *Cracking the PM Interview* and *Decode & Conquer*, read Medium posts by Emma and friends, and went over prep materials from recruiters (Quora and Meta both provided great articles to read and videos to watch). 

    For applied data and ML system design, I went through my [old ML notes](https://yuanm.notion.site/core-ml-1930f2267ce942c984b005c1bb62d429) (warning: I wrote it a while ago and might have said wrong things) and quickly went over [Grokking the Machine Learning Interview](https://www.educative.io/courses/grokking-the-machine-learning-interview) again. I wanted to read Chip Huyen's [ML interview book](https://huyenchip.com/ml-interviews-book/) but at that point, I chose to sleep more instead....

    Once I was done with one topic, I moved on to the next. People have different preferences; I like to focus on one small thing at a time and really nail it.


2. **Get ready for onsite** (30\%, $\leq$ 3 days from onsite): For each company, I also created a Notion page. Three days before an onsite, I read articles about special problems faced by the company (e.g., software pricing, conversion funnels, churn analysis, ETA predictions, feed ranking, content moderation and recommendation), browsed through the company's engineering blogs, watched talks on YouTube by the company's data scientists, leaderships, and investors, read earnings calls (public) or TechCrunch articles (public + private), and thought of insightful/interesting/endearing questions to ask my interviewers.  

    {{< figure src="https://www.dropbox.com/s/zty7paxbeuqzr57/companies.png?raw=1" width="250" >}}


    This was also the time I did mock interviews. Partly because I packed my schedule with too much stuff and partly because of I was pretty sure what I was doing, I only did 3 mock interviews as the candidate (for DoorDash, Quora, and Meta). Upon hindsight, I should've done more behavior mock interviews --- I was caught off guard in the Figma behavior round when asked about how I'd deal with common situations, which could've been prevented by practicing with people. 


## Optimize your timeline

As shown in the screenshot above, I withdrew from Google, Two Sigma, and Uber because there was no way I could finish. I completed the first 4 in early November and scheduled the rest from late November and early December. I did that because I wanted to leave more time for each company. That probably wasn't necessary: Even though each company has different business models and its own concerns, stats is stats and metrics are metrics --- the basics are the same everywhere, so you may only need 3-5 days for each additional place to prepare for company-specific questions. 

Knowing what I know now, below is what I should have done:
- **Finish phone interviews in early to mid-September**: Phone interviews are not meant to be hard --- they serve as smokescreens to weed out people who really don't know the basics (e.g., don't know $t-$tests). Don't drag out this step. 
- **Finish most onsite interviews in mid- to late October**: Offers commonly expire in 2 weeks (some companies are happy to extend it for you but hot startups have limited spots and no shortage of strong candidates). If you space onsite interviews too far apart, you won't have competing offers for negotiation. It's also a jarring experience waiting to hear from a place you wanna go while sitting on an soon-to-expire offer. To avoid that, the best strategy may be to study intensively for a month and finish all your interviews in two weeks.
- **Keep applying**: Different companies open university recruiting at different times. In my year, Quora held an open house as early as July and Faire sent out interview invites as late as December. As noted earlier, new grad DS positions are hard to come by, so keep applying whenever you see an opening. That way, even if you fail some interviews, you may still have things in the pipeline.

Of course, it's impossible to seize all opportunities. Some companies send out decisions in two days while others take a few weeks. Some companies have long interview cycles (famously Google and Two Sigma) while others short.  But when scheduling interviews, do take a moment to think about how things would play out (I didn't😬).

## What makes you stand out

Two candidates could've gone through exactly the same preparation yet come away with different outcomes. If I knew why, I'd switch careers to being a fortune-teller. That said, I do have some hunches about what makes a candidate stand out, based on my own actual/mock interview experience and talking to people. 

1. **Knowledge and honesty**: For things like stats, probability, SQL, ML, and algorithms, either you know it or not --- it is impossible to fake expertise. It's OK to admit you don't know something (which I did on multiple occasions), but it's frowned upon if you use buzzwords to cover what you don't understand. For instance, if you propose using deep learning to solve a business problem (e.g., friend recommendation), can you justify why it's better than regression- or tree-based methods? What architecture and loss function should you choose?... People may tell you differently, but I just think that one cannot be a good data science candidate if they cannot be a competent data scientist.
2. **Logic and clarity**: Do you have that friend who just won't get to the point, jumping from one story to another, forgetting how it all started? Don't be that friend in interviews. Your interviewer is a busy data scientist taking an hour out of their day to interview you and dying to go back to work. Make it easy. 


    {{< figure src="https://www.dropbox.com/s/8kp1o2eq33k7kzs/answers_first.jpg?raw=1" width="300" caption="This comic probably contains all the 'interview skills' you need.">}}

    Logical thinking in interviews is the same as logical thinking in everyday life. As a guitar player, I want to illustrate this point using a common headache I face: *"My guitar is buzzing. Why the heck? How can I make it stop?"*

    {{< figure src="https://www.dropbox.com/s/yysubwx2qby26pl/buzz.png?raw=1" width="300" caption="Diagram of the simplest system ('just the guitar and the amp').">}}

    1. **Clarify the problem**: *What's the frequency of the buzz? 60Hz or 120Hz?* 

        You don't ask clarification questions for the sake of asking clarification questions. You ask because you wanna figure out which beast you're dealing with. You can only ask useful questions if you know the domain. In this case, I ask the frequency question because I know 120Hz buzz is usually caused by grounding issues and 60Hz buzz some shielding or cable problems. 

    2. **Confirm information received**: Okay, the buzz is 60Hz. What now? --- *"Thanks for the information! Given the buzz is 60Hz, we can rule out grounding issues and focus on cable problems."*

        Interviews are kinda like those "Choose Your Own Adventure" games --- the problem space branches into different possibilities depending on the information you receive. This is another reason why simply remembering sample interview answers won't do, or companies feel safe to reuse questions. 

    3. **Lay out areas to investigate**: You want to thoroughly consider the problem for a few minutes (preferably silently) and then give the interviewer a "TL;DR" --- *"The problem may come from 4 places:  Two instrumental cables (guitar $\rightarrow$ amp, amp $\rightarrow$ speaker), the speaker, the amp, or the guitar itself. Are there areas I'm missing, or do you want me to investigate each?"*

        I learned this trick from Emma's videos. Giving a high-level summary at the beginning of each "chapter" in an interview not only makes your answer easy to follow but also allows the interviewer to correct your oversights and lead you towards a promising direction. 

    4. **Propose tests and fixes**: Finally, you want to walk through each hypothesis, in a descending order of priority --- *"First, let's check the cables, because they're the most common culprit and the easiest to fix." "We can first check if the cables are loose. If the buzz persists after we push them all the way in, then we can try new cables."* *"If we still hear the buzz, we can check if it's the amp, speaker, or the guitar. We can try a different one for each to see if the buzz would disappear."*

        Every hypothesis should be backed up by some distinct patterns in your data ("If $X$ is true, then we'd see $Y$"). Refrain from throwing out random guesses that you cannot tie back to (hypothetical) data somehow. Don't propose a ["Carl Sagan's dragon"](https://rationalwiki.org/wiki/The_Dragon_in_My_Garage) (an invisible, immaterial dragon🐲 that breathes heatless and also invisible fire...) that you can never empirically verify. 

    Try this for a problem that you often run into in real life (e.g., some say skincare is a multi-variate causal inference problem that requires this sort of methodical reasoning😳). If you can solve it logically, you won't have communication problems in interviews; if you can't, practice until it becomes a habit.

    
3. **Likeability and authenticity**: Think about people you enjoy working work with. What do they have in common? I find it pleasant to work with those who listen closely when I speak and respond thoughtfully and respectfully. Most people I like are polite, creative, chill, and optimistic. It makes me nervous to work with someone who got defensive upon disagreement or constructive feedback, doesn't let me finish, or complains in the face of challenges. Then, answer honestly: Would you like to work with yourself? If the answer is "meh", then work on yourself before working on interviews --- be whom you wanna work with.

    I have this unscientific theory that **emotional intelligence = problem solving**. In the last 4 years, I've taught hundreds of Berkeley undergrads and noticed the smartest students are also the most thoughtful. This may be no coincidence: It's more computationally expensive to simulate how your actions make people feel than being careless. I guess the good news is, just like practice can help you solve technical problems, it also helps you become more considerate.

    As for authenticity, whether it makes you a better candidate or not, I don't know, but it's less mentally taxing to be who you are. In the "Q&A" part of an onsite interview, I asked the interviewer for tips since I heard the hiring manager in the next around is famously sharp and scarily smart. Turned out the interviewer was nervous for the same reason when they were a new grad. I might have wasted a chance to "show my company research", but that was honestly what I wanted to ask and the interviewer's answer made me much more relaxed.

---
# Have Fun

## Fun is all you can control
Before my very first onsite, I confided in a friend, *"I've wanted to be a data scientist for two years now and it may just come true. It feels so heavy."* I didn't sleep at all that night and didn't get the job later (not causally related...). 

Because of the validation I got from my internship host and knowing what I'd learned over the years, I didn't doubt my ability to do data science. But the rejection made me worried about my arrangement with my advisor (regardless of the outcome, I must focus on research in November) and if I could find a job in time.

Then I watched a Hilary Hahn documentary where she talked about stage fright (spoiler: the violin goddess has never had stage fright):

> If you're afraid you're not gonna do well enough, you'll just replicate what you think people want you to do. Replicating is never the way to convince people you know what you're doing.
> You just have to be very well prepared, as prepared as you can be, but you're never completely ready --- you just do your best.
> I think the best thing is just to **look forward to it, expecting it to be a lot of fun and realizing the audience wants to have fun** --- they wanna enjoy the performance. If you have a satisfying experience being on stage, then I think that really helps with the unity of the performance.

Her words shifted my perspective. If I couldn't get a job in November, so what? I'd make time for job search while doing my dissertation. The only thing I could control was to give my interviewers thoughtful answers, and hopefully, an "entertaining" performance. I got offers from after the later interviews. 

## No fun without friends
In the end, I want to thank many DS friends who made this journey much more fun than it could've been. Yishu --- thank you for saying you already thought of me as a data scientist long before I even had got the chance to land an internship. Emma --- I absolutely don't recall anything we said, but musing about data science with you was a great enjoyment. My mock interview buddies --- I found interview preps hard but never dull because of you and may our friendship continue as our careers grow. 

<!-- - **Gamify the process**: While preparing for A/B testing interviews, I thought about edge cases where it fails, learned "non-traditional methods" in those cases, and later turned my notes into a causal inference [post](https://www.yuan-meng.com/posts/causality/). After my last interview, I reached out to Emma, asking if she'd talk about this topic (at that time, I didn't think I'd be the one making it). Looking back, I'd been waiting to reach out the moment I saw her videos; causal inference was an "excuse". I was dying to hear how a person making thoughtful product content thinks about products. Thinking again, that's also why I was exited about interviews --- I wanted to hear how data scientists think about their companies' products after working on them day in and day out. Another quirky piece of advice: **The new grad interview season is long and hard**, so if you can **turn it into a game you enjoy**, go for it. -->