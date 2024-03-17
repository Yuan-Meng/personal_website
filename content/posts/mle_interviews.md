---
title: (Opinionated) Guide to ML Engineer Job Hunting
date: 2024-03-16
math: true
tags:
    - career
    - machine learning
    - interview 
categories:
- advice
keywords:
    - career, machine learning, interview 
include_toc: true
---

# The Marriage Analogy

Nearly 2 years ago, I wrote a [blog post](https://www.yuan-meng.com/posts/newgrads/) on how to find jobs as a new grad data scientist (as a twist of fate, I never worked as a product data scientist but instead became an ML engineer at DoorDash). Back in 2021, I cared a ton about interview skills, answer "frameworks", and whatnot, which may still come handy at New Grad or Early Career levels. For experienced hires, however, I think of interviews as some sort of marriage proposal --- *it's something you can rehearse but can never force*.

If your resume is strong enough, you may get a date or two (e.g., recruiter call/HM chat/phone screen), but for the company to say "yes" (the offer), the expertise you bring to the table has to fit their roadmap and the person that you are must fit the company/team culture --- otherwise, the marriage will be painful (and short). 

Of course, you can and should prepare for interviews --- just like people spent days or even months preparing for the very moment to propose, but a strong foundation should've already been built over the years through thick (ML counterpart: when your models succeed) and thin (ML counterpart: when you learn from failures). 

# What to Prepare

> You need to be motivated... It isn't so difficult, but it's also not so easy. --- *[Linear Algebra: Theory, Intuition, Code](https://github.com/mikexcohen/LinAlgBook)*, Mike X. Cohen

The foreword from my favorite linear algebra book perfectly summarizes my opinion about the MLE interview. It isn't so hard that companies would ask you to implement, say, a two tower model in one hour. It isn't so easy that you can pass as an expert by memorizing a textbook without understanding. You need to be motivated because sheer power of will is sometimes needed to go over the vast knowledge repertoire of an MLE, especially when you also need to perform well at your current job while on the hunt. Rest assured that no time ever goes to waste even if you don't get an offer --- reading and thinking about ML foundations help you make more principled or creative decisions at work, or re-appreciate the sheer beauty of AI/ML. 

A typical MLE interview loop consists of 6 rounds, as shown in the hexagon. Some companies use a subset and some companies have their unique rounds. Generally, a successful candidate is a good coder with solid ML foundations who has delivered high-impact ML projects via challenging cross-functional collaborations and follows the latest industry/academic trends. As Josh Tobin put it, good MLEs are unicorns. 

{{< figure src="https://www.dropbox.com/scl/fi/jtwcup2ms2nt3wondqp04/Screenshot-2024-03-15-at-9.23.48-PM.png?rlkey=5e042y4cxlwo71rgajyyxkzfn&raw=1" width="1000" caption="In a talk at Berkeley, Josh Tobin likened ML engineers to 'unicorns'" >}}

1. **Coding**: For MLE candidates, coding interviews are like the GRE for Ph.D. candidates --- good performance by no means guarantees acceptance, but poor performance easily disqualifies you (unless you have an uncannily matching background). There are 3 styles of MLE coding questions:
    - **LeetCode** (*extremely common*): Classic data structures & algorithms problems; usually medium or hard -- sometimes really hard
    - **ML coding** (*increasingly more common*): Use NumPy to code up a classic ML algorithm (e.g., KNN, K-means, logistic regression, linear regression, perceptrons, etc.) or use PyTorch to code up a component of a SOTA model (e.g., multi-headed attention, positional encoding)
    - **Object-oriented programming** (*somewhat are*): Write a series of classes to implement a system (e.g., an employee registry, a non-ML movie recommender) --- kinda like a mini backend system design question
2. **ML System Design**: Your resume may be packed with impressive projects, but it's impossible to tell if you thought of the solutions yourself or someone else told you what to do. The MLSD round is a window into your first-principle thinking of how to design a scalable ML system for a business problem at hand. 
    - **Content:** In 45 min to 1 hour, you need to translate a vague business problem into an end-to-end ML system, from collecting system requirements, understanding business objectives, framing the problem as an ML task, to data generation, feature engineering, model architecture \& loss function choices, and monitoring, deployment, and A/B testing. 
    - **Focus:** Some companies focus on "ML" (e.g., SOTA model details) and some on "E" (backend implementations of feature store, model store, and data/feature/prediction pipelines, etc.). It's good to understand both.
3. **ML Breadth**: Most companies have a rapid-fire round of questions about ML foundations ("八股文快问快答"), such as the bias-variance trade-off, detecting overfitting and under-fitting, L1/L2 regularization, feature/model selection... Noways DL basics are also a necessity, such as how backpropagation works, choices of optimizers and schedulers, vanishing/exploding gradients and remedies, regularization methods that apply specifically to neural nets, etc.. 
4. **ML Domain Knowledge**: This round ensures you're an expert of your domain (e.g., NLP/ranking) who deeply understands key ideas from seminal works.
    - **NLP**: The original transformer architecture, the classic BERT/GPT/BART/T5, how Llama 2/Mistral 7B/etc. innovated from the original transformer, optimization techniques for pre-training and fine-tuning (e.g., flash attention, sliding window attention, LoRA), RLHF, LLM eval, etc.
    - **Search/rec/ads**: Ranking metrics (MRR, nDCG, MAP...), embeddings (created via contrastive representation learning such as two-tower models), candidate sampling, multi-task \& multi-objective learning, popular rec model architectures (e.g., Wide \& Deep, DCN, DCN V2, DLRM, DIN, DIEN, etc.)...
5. **Project Deep Dive**: Your past success is a predictor of your future. By walking through how you delivered impactful ML projects by choosing the right business problem to solve, carefully considering all alternatives and their trade-offs, aligning the team on key decisions, and executing against decided milestones, you give the hiring manager some confidence that you could do the same there. 
    - **A big red flag**: Nothing is worse than making a random decision without any justifications (e.g., long-term tech debt vs. short-term velocity, accuracy vs. latency, buy vs. build, etc.), because it suggests you're unable/unwilling to make difficult decisions when the moment comes, but it is those moments that make or break a project.  
6. **Behavior Interview**: A rapid-fire round of questions about how you solve problems, work with others, deliver under pressure and limited resources, iterate on feedback, mentor team members, learn from successes/failures… 

The million-dollar question is, how can one prepare for so much in so little time? As is true for most good things in life, it's *devotion* and *motivation*, and a *foundation* built over the years. I often think about the lyrics of *Remember the Name*: 

> This is ten percent luck <br/> Twenty percent skill <br/> Fifteen percent concentrated power of will <br/> Five percent pleasure <br/> Fifty percent pain <br/> And a hundred percent reason to remember the name <br/> --- Fort Minor, [*Remember the Name*](https://www.youtube.com/watch?v=7HfjKUYiumA) (2005)


# How to Prepare

## Meta Tip: Track Your Learnings

I don't know about you, but I'm the kind of person who won't remember dinner plans unless I write down with whom am I having dinner at which restaurant (or I might stand you up 😂). For a much bigger undertaking like the MLE interview prep, I track what I've learned in Notion Pages and when I learn them in Notion Calendar.


### One Topic per Notion Page 

I used to open a Google doc for each company I interviewed with and quickly noticed the redundancy --- many companies have the same rounds, so instead of writing new notes, I could've reviewed previous ones. I switched to Notion (have been a big fan since 2018) to consolidate my notes. As can be seen from the screenshot, I created a page for each interview round. Whenever I learn something new about a topic, I add it to the corresponding page. When reviewing a topic, I open the page to skim. 

{{< figure src="https://www.dropbox.com/scl/fi/gabmdr5f1oey5skstxkh6/Screenshot-2024-03-15-at-11.29.21-PM.png?rlkey=cf4m9kphy4b80w4prppzp5u92&raw=1" width="450" caption="I created Notion pages to collect knowledge for each MLE interview round" >}}

I can't share my notes since they contain NDA content --- you should create your own notes anyways, because deep learning (no puns intended) doesn't come from reading someone else's notes; rather, notes are a means to consolidate your learning from the source (e.g., textbooks, SOTA papers, talks, engineering blogs, code, etc.).

### Block Time on Your Calendar

I practice LC-style coding for one hour before work and 2 hours after work. I read NLP + ranking papers/books/blogs/code/etc. for 2-3 hours in the evenings. I sometimes take days off, but adhere to the schedule if I can. Having this schedule in Notion Calendar reminds me to focus on work during the day (so I don't think about interviews) and enjoy learning after work (rather than stretching "busy work"). 

You should find a schedule that suits your lifestyle. Or perhaps, you thrive without schedules, like some of my friends do. The bottom line is, ML interview prep takes a ton of time and you need to learn how to fit it into your normal work/life. 

{{< figure src="https://www.dropbox.com/scl/fi/g1m9j7z63j51hbfa2q9fi/Screenshot-2024-03-15-at-11.39.04-PM.png?rlkey=zg36o6hys6yhpnd4qi8i2xd6j&raw=1" width="1200" caption="I blocked time for LC practice and NLP/ranking readings in Notion Calendar" >}}


## Coding

### Classic DS\&A


Before grinding LC, take [NeetCode's](https://neetcode.io/courses) beginner and advanced courses, where the incredible [Navdeep Singh](https://www.youtube.com/c/neetcode/about) explains all the data structures and algorithms you'll ever need for coding interviews. Each lesson typically lasts between 10 to 25 minutes and includes 3 to 5 practice questions. The courses are paid, but for the cost of 3-4 take-out orders, you get the best way to organize your learning and practice.

 {{< figure src="https://www.dropbox.com/scl/fi/b59qf2upsoz3pu4rvsi6s/Screenshot-2024-03-16-at-12.18.14-AM.png?rlkey=gjprpbxezvo4uzjt6d53w2qpp&raw=1" width="600" >}}

 The table below summarizes data structures and algorithms in the two courses. 

 <table>
<tr>
<th>Common data structures</th>
<th>Common algorithms</th>
</tr>
<tr>
<td>

- Strings
- Arrays
- Matrices (can represent graphs)
- Hash tables
- Hash sets
- Heaps (priority queues)
- Linked list
- Trees
  - Binary tree
  - Binary search tree
  - Tries
  - Segment trees
- Graphs
  - Union Find
  - Directed acyclic graphs (DAGs)

</td>
<td>

- Binary search
- Depth-first search (DFS)
- Breadth-first search (BFS)
- Backtracking 
  - Subsets
  - Combinations
  - Permutations
- Sliding windows
- Two pointers
- Prefix sum (e.g., Kadane's)
- Dynamic programming (top-down, bottom-up)
- Graph algorithms
  - Topological sort
  - Dijkstra's
  - Prim's 
  - Kruskal's
- Sorting
- Bit manipulation

</td>
</tr>
</table>

After that, I recommend doing Blind 75 and NeetCode 150, two problem collections that cover classic, high-frequency questions from interviews. NeetCode All (400+ problems) may be an overkill --- if you have interviews lined up, then move on to practice company-tag questions (URL pattern: `leetcode.com/company/<company name>`).

 {{< figure src="https://www.dropbox.com/scl/fi/rxbnz4jpmzd8q22a7ccyx/Screenshot-2024-03-16-at-12.41.08-AM.png?rlkey=dpmx3aseb5ijp84eo61z64o26&raw=1" width="450" >}}


I was terrible when doing my first 100-200 questions, often having no clues or struggling to translate my thoughts into code. After 300-400 questions, I started to get the hang of it, often having a strong sense of the solution by recognizing patterns in the prompt/test cases. When solving the problem, I also have an intuition about whether I'm going in the right direction or not (e.g., adding more and more auxiliary data structures or special cases is a strong sign of going awry). Think of LC problems as training data for the AI that is you --- too few, you don't get the chance to observe representative patterns and learn useful tricks. 

However, the "learning algorithm" also matters. I like how Chris Jereza [put it](https://www.youtube.com/watch?v=lDTKnzrX6qU) --- many people have seen so many LC questions yet still fail to recognize common tricks and patterns, just like many go to the gym so often yet still fail to get fit --- *mastery is not just determined by how many questions you did, but also by intensity and reflection*. I track problems I solved in Notion, mark my level of mastery (so I can re-do problems for which I had "no clue" or "bugs"), and summarize key insights for each problem in one or two sentences. For each company I interviewed with, I manually added questions from LC --- the long-term benefits far outweigh the extra time in the beginning. You can modify my [template](https://www.notion.so/yuanm/7667ceebb8dd4f3f840d96e43c9733c1?v=a9389f5990ed4cc691e326f92488624f&pvs=4) to your liking. 

 {{< figure src="https://www.dropbox.com/scl/fi/ofj185vw66fucdyqb5sd5/Screenshot-2024-03-16-at-12.53.20-AM.png?rlkey=2if7axp0v8wpwbothdo6onuix&raw=1" width="1200" >}}


I highly recommend doing a few mock interviews before the real interview, because it prepares you for solving problems while being watched. That said, I'm not a big fan of excessive mock interviews --- after two or three, I already know I can think under stress; it's just a matter of whether I know the algorithm or not. For me, practicing a wide variety of problems on my own is more efficient than spending hours scheduling mock interviews with friends, but if your bottleneck is nerves rather than knowledge, then it may be worth investing more time in mock interviews.

### ML Coding

Some companies ask ML candidates to code an ML algorithm from scratch. The original motivation is to see if the candidate can translate their understanding into code; in practice, it's a memorization game --- you can prepare for common algorithms in advance. You can find my implementations of classic ML (e.g., KNN, K-means, linear regression, and logistic regression) in my [blog post](https://www.yuan-meng.com/posts/md_coding/). Candidates may also be asked to implement a component from a neural net. For instance, NLP candidates may be asked to implement multi-headed attention, which you can find in my other [post](https://www.yuan-meng.com/posts/attention_as_dict/). 

It's rarer but you may also be asked to implement eval metric calculations given true labels and predicted labels, so prepare for the common ones (e.g., AUC, nDCG).

## ML System Design

At work, writing an RFC ("Request For Comments") for a new model can take more than a week: I need to understand the business problem the model tries to solve or technical issues it aims to improve, the status quo, in what product surface/UI will the model be called under what latency requirements, what training data we have or need to request, how to frame the ML task, and the path towards productionization (e.g., data/feature/training/serving/monitoring pipelines, offline eval, A/B...). 

In 45 min to 1 hour, you'll be asked to replicate the whole process. You gotta structure your answers well to keep the interview moving on track. Some books (Alex Xu's [*Machine Learning System Design Intervie*w](https://www.amazon.com/Machine-Learning-System-Design-Interview/dp/1736049127)) and courses (Educative: [*Grokking the Machine Learning Interview*](https://www.educative.io/courses/grokking-the-machine-learning-interview) and [*Machine Learning System Design*](https://www.educative.io/courses/machine-learning-system-design)) teach you exactly how to structure your answers. I summarized lessons from these sources in my [notes](https://yuanm.notion.site/ml-system-design-cd12d1028ed640e199e64ed0ed461851?pvs=4). 

 {{< figure src="https://www.dropbox.com/scl/fi/bld9zng0xz2r5grp6fp9i/Screenshot-2024-03-16-at-10.55.08-AM.png?rlkey=jfd9g8bi2h1dv6k2j1xlqwjz0&raw=1" width="1200" >}}

At the end of the day, however, **structure exists to support substance**. When it comes to actual label/model designs and the production system, you'll be asked detailed questions about industry gold standards and your best practices at work --- *this is the substance that shines through*. As some with a Search/NLP background, can I design an autonomous driving system for an interview? Perhaps, if I memorize some image data annotation and augmentation techniques and popular CV models, but my lack of CV experience will be seen through and, if offered the job, I'll most definitely run the system to the ground... So my advice for preparation is, sure, read Alex's book to master the structure, but the real preparation is being a good ML engineer who designs thoughtful, practical, reliable, and scalable ML systems and going to design reviews to learn how other ML engineers think (or critique). 


## ML Breadth

In this new wave of AI/ML craze, many non-ML friends and strangers asked me how to transition into an MLE. As someone who went through this transition back in school, I hate to discourage people, but it's also no use being dishonest: Grasping machine learning theory and applying it to production systems takes years of study and experience. Anyone can copy \& paste PyTorch code online, but only seasoned ML engineers can trouble-shoot a model that's not performing as intended (see my [post](https://www.yuan-meng.com/posts/ltr/) on Airbnb's long-winded journey into DL). ML engineering is somewhat like medicine in that neural networks are as much an enigma as the human body: Highly specialized knowledge is needed to diagnose a non-working system and prescribe remedies. 

This post is aimed at helping current ML engineers seeking new ML opportunities. I don't have first-hand advice on transitioning from DS/SWE to MLE for experienced hires, but someone else may have, so I encourage you to keep looking for theirs. 

{{< figure src="https://www.dropbox.com/scl/fi/92naxalcd6i78zx1ztyzb/0F7631FE-DE40-47A8-A9E3-2DF4923C4C95_1_201_a.jpg?rlkey=ek4tn1eiu7ttg8lsd624wqanh&raw=1" width="500" >}}

All supervised learning models share the 3 following components ([Burkov, 2019](https://github.com/Chrisackerman1/The-Hundred-Page-Machine-Learning-Book)):
- **Loss function**: The discrepancy between the predicted value and the target value
- **Optimization criterion**: Objectives the training process seeks to optimize for
- **Optimization routine**: Approaches (e.g., gradient descent, SGD, momentum, Adam) to leveraging training data to find a solution to the optimization criterion  

The so-called "ML breadth" interview usually revolves around these concepts. For instance, how do we know if model training has converged (by looking at train/valid losses)? What is the so-called bias-variance trade-off? How do we tell if a model is under-fitting or over-fitting? If a model is over-fitting, how can we regularize it (e.g., L1/L2 regularization, early stopping, dropout...)? When training very deep neural nets, how do we prevent vanishing or exploding gradients? How do we choose the right optimizers and schedulers? How do we properly initialize model weights?... 

You can find the answers to these questions in countless textbooks, YouTube videos, tech blogs, interview guides, etc.. I like learning things systemically (as opposed to in a piecemeal manner) and below are books and online resources I like the best. 

1. **Math foundations**: Once in a while when I scroll through my LinkedIn/X feed, I'd see ML "gurus" fighting over whether math is necessary for getting into ML in 202X. My sense is that it greatly helps to know/review basic linear algebra and multi-variate calculus, or ML papers will read like gibberish --- and ML engineers do need to catch up on seminal/new papers in their fields.
    - **Linear algebra**: [Essence of Linear Algebra](https://www.youtube.com/watch?v=fNk_zzaMoSs&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) by 3Blue1Brown and [Linear Algebra: Theory, Intuition, Code](https://github.com/mikexcohen/LinAlgBook) by Mike X Cohen 👉 **Why**: Some joke that ML is glorified linear algebra, which is quite true. Many ML methods are directly taken from linear algebra (e.g., OLS, matrix factorization, LoRA), many ML concepts hinge on linear algebra (e.g., embedding: projection), and linear algebra offers a compact way to represent weights \& biases in each neural net layer and operations on them... 
    - **Calculus**: [Essence of Calculus](https://www.youtube.com/watch?v=WUvTyaaNkzM&list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr) by 3Blue1Brown 👉 **Why**: So you understand how optimization routines work and vanishing/exploding gradient problems
2. **Combine theory + code**: I recommended [Machine Learning with PyTorch and Scikit-Learn](https://www.amazon.com/Machine-Learning-PyTorch-Scikit-Learn-learning-ebook/dp/B09NW48MR1) by Sebastian Raschka 2 years ago and my recommendation stands today. It's the best foundational ML book that combines theory with code, covering everything from traditional ML to SOTA, implementing both using libraries and from scratch. I recommend reading it from cover to cover and run the [code](https://github.com/rasbt/machine-learning-book). 
3. **Go deep into theory**: Last but not least, Kevin Murphy wrote 3 epic books on machine learning theory, the most recent ones being [Probabilistic Machine Learning: An Introduction (2022)](https://probml.github.io/pml-book/book1.html) and [Probabilistic Machine Learning: Advanced Topics (2023)](https://probml.github.io/pml-book/book2.html). Even for concepts you and I can't be more familiar with such attention, he still shed fresh light on them every now and then (see my [post](https://www.yuan-meng.com/posts/attention_as_dict/) on the "attention as soft dictionary lookup" analogy). An an ML who wishes to connect the dots scattered over the years, I think Kevin's books are the dream. 
    {{< figure src="https://www.dropbox.com/scl/fi/f2624w8e5dxe77cko9791/Screenshot-2024-03-16-at-2.57.10-PM.png?rlkey=8a51dwek7aiqdwtbn8a6ix0ao&raw=1" width="500" caption="[Table of Content](https://probml.github.io/pml-book/book2.html#toc) of Probabilistic Machine Learning: Advanced Topics" >}}

<!-- I don't specifically prepare for ML Breadth interviews or keep Notion notes --- I did that 2 years ago (see my [old notes](https://www.notion.so/yuanm/core-ml-1930f2267ce942c984b005c1bb62d429?pvs=4)).  -->

Chip Huyen curated some [questions](https://huyenchip.com/ml-interviews-book/contents/8.1.2-questions.html) in her ML interview book and a Roblox Principal MLE also wrote a [nice article](https://medium.com/@reachpriyaa/how-to-crack-machine-learning-interviews-at-faang-78a2882a05c5) that contains many actual questions I've encountered, which you can go over before an ML Breadth interview to refresh your memories.

## ML Depth 

If you interview with Meta or the likes, you'll get the chance to choose a team later, but in today's market, most MLE candidates interview with specific teams that look for specific talents. As a domain expert, you need to know the gold standard in your chosen field(s) and the new bells and whistles. For me, it's ranking and NLP.

### Paper Tracker


It would take many lifetimes to read all papers that ever got published in your field. That said, good ML engineers should be familiar with the seminal works and "new classics" in their domain and make paper reading a regular habit. I used to save paper PDFs in random folders and would often forget which ones I had read, let alone what I learned from them. Then, inspired by the coding tracker, I created a paper tracker to keep track of papers/books/blog posts I read on different topics. 

{{< figure src="https://www.dropbox.com/scl/fi/m5nh0fyhaxe4f282kvkyo/Screenshot-2024-03-16-at-3.15.57-PM.png?rlkey=twfh7pw2n1c0mlwelmtil6wjc&raw=1" width="1500" >}}

For each entry, I write a one- or two-sentence TL;DR, and after clicking on the paper title, I jot down detailed notes. The example below is for the [Llama 2](https://arxiv.org/abs/2307.09288) paper. 

{{< figure src="https://www.dropbox.com/scl/fi/g8qics73srz7jbexo3994/Screenshot-2024-03-16-at-3.26.40-PM.png?rlkey=3vcyjsh8wj3xwdkqj22muzw93&raw=1" width="1000" >}}


### Ranking

Ranking is the revenue-generator of consumer-facing companies --- without search/discovery/ads, companies can't make money. This is why ranking is one of the most impactful and exciting domains and has the most job openings (arguably tied with NLP). 

I recently summarized classic search ranking papers in a blog post, [An Evolution of Learning to Rank](https://www.yuan-meng.com/posts/ltr/). You can find the original papers in the ["Learn More"](https://www.yuan-meng.com/posts/ltr/#learn-more) section. 

Recommendation is the sibling of search, where users don't submit a query yet we still read their minds and show appealing contents to them. To gain a deep understanding of this domain, my friend recommended [深度学习推荐系统](https://www.amazon.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%EF%BC%88%E5%85%A8%E5%BD%A9%EF%BC%89-%E5%8D%9A%E6%96%87%E8%A7%86%E7%82%B9%E5%87%BA%E5%93%81-%E7%8E%8B%E5%96%86/dp/7121384647/ref=sr_1_1?crid=XZSJ6ZW1282A&dib=eyJ2IjoiMSJ9.BxuHruDCtEpyjInRfwvRFn7pyNHfnLqM9I7MhiVO4QjP5NwIPzDAXhfcdM4N-9cc.-Amg_ZFv605zkwdYZmfQJIg4d3GGoacM4rlgWcwz29c&dib_tag=se&keywords=%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F&qid=1710629348&sprefix=%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%2Caps%2C141&sr=8-1) by Zhe Wang, who used to be the Head of Engineering at TikTok. This book not only reviews the evolution of recommendation models from traditional collaborative filtering to emerging deep learning architectures, but also speaks from practical experience of productionizing such models. Unfortunately for non-Chinese speakers, this book is only written in Chinese. I also find Gaurav Chakravorty's [blog](https://substack.com/@recsysml) and [repo](https://github.com/gauravchak) a gold mine of practical rec models knowledge. For instance, in this [two-tower model repo](https://github.com/gauravchak/two_tower_models), Gaurav started with a bare bone structure and gradually introduced components such as user history embeddings, position debiasing, and a light ranker, which is highly educational.

I don't know much about ads, which has extra concerns (e.g., calibration, bidding) on top of search/recs. I've turned down recruiters reaching out for ads positions.


### NLP 

As a Cognitive Science student, I was interested in all aspects of human intelligence but language😅, thinking that language is an indirect, inaccurate byproduct of thought. Back in the summer of 2021 when I was driving to LA, I listened to a Gradient Dissent [episode](https://www.youtube.com/watch?v=VaxNN3YRhBA) where Emily Bender argued that to interact with language models as we do with humans, we must treat them as equal linguistic partners rather than as mechanistic pattern recognizers. Back then, that day seemed far away. Now with the success of ChatGPT and ever more powerful new LLMs, NLP has become the fastest growing field in AI/ML, with far more active research, new startups, and headcount than any other ML fields. I got into NLP only because I was assigned to work on Query Understanding when I joined DoorDash as a new grad. If ranking has always fascinated me, then NLP is a required taste I've come to appreciate.

To get into NLP, first read [Natural Language Processing With Transformers](https://www.amazon.com/Natural-Language-Processing-Transformers-Applications/dp/1098103246) by Hugging Face --- it's the best textbook on transformer-based models, with intuitive explanations and practical code examples, covering a wide range of NLP tasks including sequence classification, named entity recognition, question answering, summarization, generation, etc.. It also talks about model compression, training without enough data, and pretraining LLMs from scratch, which come handy in production. 

While awesome, the HF book leaves you wanting for details from the original papers. To learn more, check out Sebastian Raschka's [LLM reading list](https://magazine.sebastianraschka.com/p/understanding-large-language-models), which starts with the original transformer in *Attention is All You Need* and its predecessors, moves on to the post child of each iconic model architecture (e.g., encoder-only: BERT; decoder-only: GPT; encoder-decoder: BART), and ends on SOTA models, training techniques (e.g., [LoRA and QLoRA](https://lightning.ai/pages/community/lora-insights/)), and architectural innovations (e.g., FlashAttention, Rotary Positional Embeddings). To understand more deeply how each component of the original transformer can be optimized, I recommend Lilian Weng's blog post [The Transformer Family Version 2.0](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/). To get an in-depth understanding of latest models (e.g., Llama 2, Mistral, Mamba) and fundamental techniques (e.g., quantization, RAG, distributed training), I suggest watching Umar Jamil's amazing YouTube [videos](https://www.youtube.com/@umarjamilai) and reading his neat implementations ([GitHub](https://github.com/hkproj)). Andrej Karpathy has also created a great course [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) and legendary educational repos (e.g., [miniGPT](https://github.com/karpathy/minGPT), [nanoGPT](https://github.com/karpathy/nanoGPT), [minbpe](https://github.com/karpathy/minbpe)), which use minimalist code to train minimalist models.

For important topics not covered, you can read representative blog posts and literature review to catch up, such as [prompt engineering](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/), [RLHF](https://huggingface.co/blog/rlhf), and [LLM evaluation](https://arxiv.org/abs/2307.03109).

There will never be a day when you think you know enough about NLP. Neither does your interviewer. Most NLP interviews focus on the basics, such as the original transformer, common tokenizers, popular models (e.g., BERT/GPT/BART/T5/Llama 2), and widely used techniques (e.g., RLHF, RAG, LoRA). Focus on these in your interview prep and don't worry about being asked about papers that came out yesterday.

## Behavior Interview

### Tell Me A Time When...

I used to stammer my way through behavior interviews because I didn't prepare for certain questions I was asked. Then Amazon recommended [Behavioral Interviews for Software Engineers](https://www.amazon.com/Behavioral-Interviews-Software-Engineers-Strategies/dp/B0C1JFQYCR/ref=tmm_pap_swatch_0?_encoding=UTF8&qid=&sr=) to me, which conveniently lists 30+ behavior questions one will ever get asked. I don't find the answers particularly applicable to MLE candidates --- the question list is the treasure. Just like I did for coding and reading, I created a tracker for BQ questions, coming up with the best example for each question and putting my answer in the STAR (situation, task, action, result) format. 

{{< figure src="https://www.dropbox.com/scl/fi/5fe9oqb3bufcyonokjlg2/Screenshot-2024-03-16-at-3.32.07-PM.png?rlkey=3azakxu2vyiplwoiw7n9kkevg&raw=1" width="1000" >}}

Any example you provide should be a true reflection of your character, but think carefully about which examples to provide --- don't paint yourself in a bad light 😂. Reading Merih Taze's [Engineers Survival Guide](https://www.amazon.com/Engineers-Survival-Guide-Facebook-Microsoft/dp/B09MBZBGFK) helped me understand what behavior makes a good engineer --- model not your answers but your words and deeds after it. I wish I read it in my first year at DoorDash, which would've saved me many sweat and tears from reaching alignment, getting visibility, making impact, helping  managers to help you, resolving conflicts, and whatnot. Gergely Orosz's [The Software Engineer's Guidebook](https://www.amazon.com/Software-Engineers-Guidebook-Navigating-positions/dp/908338182X) is another great book to read which you have the time.

### Project Deep Dive

In some sense, the project deep dive is like an extended version of "tell me about the project you're most proud of", but goes into all technical details imaginable ---

- **Background**: What business problem was the model trying to solve? 
- **Solution \& alternatives**: How did you solve the problem? Which alternatives did you consider? What were the trade-offs? How did you get the team onboard? 
- **Challenges**: Which technical challenges did you face? Cross-functionally?
- **Impact**: In the end, what impact did you create? How did you measure it?
- **Reflection**: What were you most proud of and what would you change?

Choose a successful project you drove from end to end, or the lack of ownership or impact will backfire. Not only that, choose a project for which you made wise decisions by carefully considering all alternatives and thoughtfully balancing all sides (e.g., business goals, urgency, engineering excellence, infra...). If you made a big mistake (e.g., only involving a function until it was too late, making hasty decisions without thinking ahead) in a project, talk about other projects --- this is your chance to showcase your technical + XFN collab prowess, not a post mortem.

To prepare for this round, I re-read my self-review from perf cycles and made slides for each significant project I've done. I highly recommend finding a friend not even in your field and practicing the delivery of your stories.  

# The "Moment of Truth"

Back in school, I often lost sleep the night before the interview, fearing any wrong word I said could cost me the job. Nowadays I work, sleep, and play with cats as usual before interviews, knowing results are determined long before then. If I am "the one", then all I need is to chat with the interviewer about topics we both love and feel passionate about. For instance, I can spot a ranking expert from a brief conversation --- even if they occasionally stammer or forget one or two minute details --- so can my interviewer. If I make grave mistakes or can't recover what I forget from first-principle thinking, then I'm not the one and not hiring me is for the best. It's kinda like being rejected after a few dates isn't pleasant, but it's orders of magnitude better than being married to someone profoundly incompatible.

If you know me by now, you might've guessed that I created an interview progress tracker, using one table to track the stage (**scheduling**: sent availability but coordinator hasn't finalized the schedule; **in progress**: interview scheduled for a future date; **done**: completed the given interview round) of each round (e.g., recruiter/HM chat, phone screen, onsite) with each company and another table to track what each round entails. When interviewing with a specific company, I filter down to the rounds asked by that company and review the corresponding Notion Pages. 

{{< figure src="https://www.dropbox.com/scl/fi/bp78z6q27pa5cvotn342k/Screenshot-2024-03-16-at-11.44.33-PM.png?rlkey=wgymx513hyfu6t1wbo2zutvr1&raw=1" width="1000" >}}

