---
title: (Opinionated) Guide to ML Engineer Job Hunting
date: 2024-10-15
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

Over 2 years ago, I wrote a [blog post](https://www.yuan-meng.com/posts/newgrads/) on how to find jobs as a new grad data scientist (as a twist of fate, I never worked as a product data scientist but instead became an ML engineer at DoorDash). Back in 2021, I cared a ton about interview skills, answer "frameworks", and whatnot, which may still come handy at New Grad or Early Career levels. For experienced hires, however, I think of interviews as some sort of marriage proposal --- *it's something you can rehearse but can never force*.

If your resume is strong enough, you may get a date or two (e.g., recruiter call/HM chat/phone screen), but for the company to say "yes" (the offer), the expertise you bring to the table has to fit their roadmap and the person that you are must fit the company/team culture --- otherwise, the marriage will be painful (and short). 

Of course, you can and should prepare for interviews --- just like people spent days or even months preparing for the very moment to propose, but a strong foundation should've already been built over the years through thick (ML counterpart: when your models succeed) and thin (ML counterpart: when you learn from failures). 

# What to Prepare

> You need to be motivated... It isn't so difficult, but it's also not so easy. --- *[Linear Algebra: Theory, Intuition, Code](https://github.com/mikexcohen/LinAlgBook)*, Mike X. Cohen

The foreword from my favorite linear algebra book perfectly summarizes my opinion about the MLE interview. It isn't so hard that companies would ask you to implement, say, a two-tower model in one hour. It isn't so easy that you can pass as an expert by memorizing a textbook without understanding. You need to be motivated because sheer power of will is sometimes needed to go over the vast knowledge repertoire of an MLE, especially when you also need to perform well at your current job while on the hunt. Rest assured that no time ever goes to waste --- even if you don't get an offer --- reading and thinking about ML foundations help you make more principled or creative decisions at work, or re-appreciate the sheer beauty of AI/ML. 

A typical MLE interview loop consists of 6 rounds, as shown in the hexagon. Some companies use a subset and some companies have their unique rounds. Generally, a successful candidate is a good coder with solid ML foundations who has delivered high-impact ML projects via challenging cross-functional collaborations and follows the latest industry/academic trends. As Josh Tobin put it, good MLEs are unicorns. 

{{< figure src="https://www.dropbox.com/scl/fi/jtwcup2ms2nt3wondqp04/Screenshot-2024-03-15-at-9.23.48-PM.png?rlkey=5e042y4cxlwo71rgajyyxkzfn&raw=1" width="1000" caption="In a talk at Berkeley, Josh Tobin likened ML engineers to 'unicorns'" >}}

1. **Coding**: For MLE candidates, coding interviews are like the GRE for Ph.D. candidates --- good performance by no means guarantees acceptance, but poor performance easily disqualifies you (unless you have an uncannily matching background). There are 3 styles of MLE coding questions:
    - **LeetCode** (*extremely common*): Classic data structures & algorithms problems 👉 <span style="background-color: #FDB515">same as the bar for SWE</span>: Meta asks 2 Mediums in 45 min; most places ask 1 Hard + 1 Medium in 1 hour, or 1 Hard with some follow-ups in 45 min.
    - **ML Coding** (*traditional or researchy positions*):  Code up a simple model (e.g., K-means, logistic/linear regression), a SOTA building block (e.g., multi-headed attention), or an ML concept (e.g., AUC, vector/matrix multiplication) 👉 these were popular a few years ago, and some companies still keep them; researchy positions may also probe into ML coding.
    - **Object-Oriented Programming** (*backend-heavy positions*): In places where MLEs handle heavy backend work (e.g., Netflix, Reddit), candidates are often asked to implement some sort of CRUD (create, read, update, delete) API 👉 class methods get increasingly more complex with each follow-up.
2. **ML System Design**: Your resume may be packed with impressive projects, but it's impossible to tell if you thought of the solutions yourself or someone else told you what to do. The MLSD round is a window into your first-principle thinking of how to design a scalable ML system for a business problem at hand. 
    - **Content:** In 45 min to 1 hour, you translate a vague business problem into an end-to-end ML system, from collecting requirements, understanding business objectives, framing the problem as an ML task, to data generation, feature engineering, label design, model architecture \& loss function choices, and eventually monitoring, deployment, scaling, and A/B. 
    - **Focus:** Some companies focus on the "ML" (SOTA model details) and some on the "E" (the end-to-end ML system at training and serving time, including feature store, model store, data/feature/prediction/monitoring pipelines, etc.). Some companies have 2 design rounds covering both. It's necessary to deeply understand both industry-standard models and the ML infra.
3. **ML Breadth**: Many companies have a rapid-fire round asking questions about ML foundations ("八股文快问快答"), such as the bias-variance trade-off, detecting overfitting and under-fitting, L1/L2 regularization, feature/model selection... Noways DL basics are also a necessity, such as how backpropagation works, choices of optimizers and schedulers, vanishing/exploding gradients and remedies, regularization methods that apply specifically to neural nets, etc.. 
4. **ML Domain Knowledge**: This round ensures you're an expert of your domain (e.g., NLP/ranking) who deeply understands key ideas from seminal works.
    - **NLP**: The original Transformer in *Attention Is All You Need*, popular Transformers (BERT/GPT/BART/T5/Llama), efficient Transformers (e.g., FlashAttention, Sliding Window Attention, LoRA, QLoRA, PEFT), RLHF, eval...
    - **Search/Rec/Ads**: Multi-stage ranking design (candidate generation 👉 first-pass ranking 👉 second-pass ranking 👉 re-ranking), ranking metrics (MRR@K, nDCG@K, MAP@k, Recall@k...), popular DL architectures (e.g., Wide \& Deep, DCN, DCN V2, DIN, DIEN, DHEN, etc.), multi-objective learning and value models, solutions for cold-start/position bias/diversity...
5. **Project Deep Dive**: Your past success is a predictor of your future. By walking through how you delivered impactful ML projects by choosing the right business problem to solve, carefully considering all alternatives and their trade-offs, aligning the team on key decisions, and executing against decided milestones, you give the hiring manager some confidence that you could do the same there. 
    - **A Big Red Flag**: Nothing is worse than making a random decision without any justifications (e.g., long-term tech debt vs. short-term velocity, accuracy vs. latency, buy vs. build, etc.), because it suggests you're unable/unwilling to make difficult decisions when the moment comes, but it is those moments that make or break a project.  
6. **Behavior Interview**: A rapid-fire round of questions asking how you solve problems, work with others, deliver under pressure and limited resources, iterate on feedback, mentor team members, learn from successes/failures… 

I joke with my friends that <span style="background-color: #FDB515">project management is the only real skill behind ML interview preparation</span> 😆. Finding the time is already challenging --- it takes *devotion* , *motivation*, and *discipline* to make consistent progress toward your goals, be them learning the state of the art or enhancing the foundational knowledge that you've built over the years. I sometimes listen to *Remember the Name* to remind myself:

> This is ten percent luck <br/> Twenty percent skill <br/> Fifteen percent concentrated power of will <br/> Five percent pleasure <br/> Fifty percent pain <br/> And a hundred percent reason to remember the name <br/> --- Fort Minor, [*Remember the Name*](https://www.youtube.com/watch?v=7HfjKUYiumA) (2005)

# How to Prepare

## Meta Tip: Track Your Learnings

I'm your stereotypical INTJ who plans everything and tracks everything 🤣. For each interview topic, I have a dedicated Notion page; I block daily recurring times on my Notion Calendar to study a given topic at a give time; when the time comes, I track each interview round as a task in a centralized table with all my interviews.

### One Topic per Notion Page 

I used to open one Google doc for each company I interviewed with and quickly noticed the redundancy --- many companies have the same rounds, so instead of writing new notes, I could've reviewed previous ones. I switched to Notion (a big fan since 2018) to consolidate my notes. As you can see from the screenshot, I created one page for each interview round. Whenever I learn something new about a topic, I add it to the corresponding page. When reviewing a topic, I open the page to skim. 

{{< figure src="https://www.dropbox.com/scl/fi/v27sq1fb7ma34rq05a83y/Screenshot-2024-10-15-at-7.27.07-PM.png?rlkey=o1u0pz016xxzsdql3cqsv5052&st=b890it5j&raw=1" width="400" caption="I created Notion pages to collect knowledge for each MLE interview round" >}}

I can't share my notes since they contain NDA content --- you should create your own notes anyways, because deep learning (no puns intended) doesn't come from reading someone else's notes; rather, notes are a means to consolidate your learning from the source (e.g., textbooks, SOTA papers, talks, engineering blogs, code, etc.).

### Block Time on Your Calendar

I practice LC-style coding for one hour before work and 2 hours after work. I read NLP + ranking papers/books/blogs/code/etc. for 2-3 hours in the evenings. I do take days off, but try to adhere to this schedule if I can. Having a schedule in Notion Calendar reminds me to focus on work during the day (so I don't think about interviews) and enjoy learning after work (rather than stretching "busy work"). 

You should find a schedule that suits your lifestyle. Or perhaps, you thrive without schedules, like some of my friends do. The bottom line is, ML interview prep takes a ton of time and you need to learn how to fit it into your normal work/life. 

{{< figure src="https://www.dropbox.com/scl/fi/g1m9j7z63j51hbfa2q9fi/Screenshot-2024-03-15-at-11.39.04-PM.png?rlkey=zg36o6hys6yhpnd4qi8i2xd6j&raw=1" width="1200" caption="I blocked time for LC practice and NLP/ranking readings in Notion Calendar" >}}

### Track Interview Progress

 When the time comes for interviewing, I track each interview as a task in a centralized table. Before an interview is scheduled, I place it under "scheduling." Once it's confirmed, I move it to "in progress," and after it's completed, I drag it to "done." Each interview comes with its own Notion page, where I note the date and time for each round, what to prepare, and the time I decide to allocate (e.g., 2 weeks for coding, 1 week for ML design, and 3 days for behavior).

{{< figure src="https://www.dropbox.com/scl/fi/h6f0r7m4g8pmjyig53jra/Screenshot-2024-10-15-at-8.01.27-PM.png?rlkey=4laxswoi84nkf94o6mkpvs4wp&st=5nmw4hz2&raw=1" width="1200" caption="I track each round as a task, with a detailed page for the prep agenda." >}}

For me personally, this tracking system relieves my stress and helps me focus:

1. **Tunnel vision**: When I prepare for a phone interview, I focus only on what's necessary for that phone interview and don't worry about whether I'll pass the onsite; when I prepare for an onsite at one company, I don't worry about preparing for another company's interview later.
2. **Get into the zone**: When I open an interview-specific page, I get into the zone.

## Coding

### Classic DS \& A

As the NeetCode guy [put it](https://www.youtube.com/watch?v=2V7yPrxJ8Ck), <span style="background-color: #FDB515">the only way to solve LC problems is to have seen the algorithms beforehand</span>; it's unrealistic to derive neat algorithms on the spot. That said, I do think coding interviews are valuable for weeding out "bad" engineers: If someone is a great engineer, it would be out of character for them to write a solution with no regard for time or memory efficiency, keep adding "patches" to save a flawed logic, or get frustrated when their code doesn't work the first time. Of course, the reverse isn't true: Just because someone solves a LC problem doesn't mean they're a great engineer -- they might've just seen the solution yesterday.

So, how do you get started with LC? If you start with the 1312-page [CLRS](https://en.wikipedia.org/wiki/Introduction_to_Algorithms) book, you may still not have got your hands dirty a year later. If you just grind random questions, you may never identify common patterns. I find [NeetCode's](https://neetcode.io/courses) beginner and advanced courses a great middle ground. Navdeep Singh explains almost all the data structures and algorithms you'll ever need for coding interviews. Each lesson lasts between 10 to 25 minutes and comes with 3 to 5 practice questions, so you'll learn the theory and get hands-on practice immediately. The courses are paid, but for the price of 3-4 take-out orders, you get the best way to organize your LC prep.

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

After the courses, I recommend doing Blind 75 and NeetCode 150, two problem collections that cover classic, high-frequency interview questions. NeetCode All (500+ problems) is an overkill --- if you have interviews lined up, then move on to practicing LC problems with company tags (URL: `leetcode.com/company/<company name>`).

{{< figure src="https://www.dropbox.com/scl/fi/rxbnz4jpmzd8q22a7ccyx/Screenshot-2024-03-16-at-12.41.08-AM.png?rlkey=dpmx3aseb5ijp84eo61z64o26&raw=1" width="500" >}}

The final test is solving the problem on the spot. Around the time I began getting offers, including from companies notorious for asking hard coding questions, I started feeling calm and composed with new problems. Below is how I approach them:

{{< figure src="https://www.dropbox.com/scl/fi/jtg4r456ve2gdfzat7ku6/Screenshot-2024-10-15-at-8.20.41-PM.png?rlkey=0xwvfs2j657ji1kw47iazn56k&st=v85wynqt&raw=1" width="1200" caption="My approach to solving (almost) all LC problems in coding interviews." >}}

1. **Understand**: Interviewers usually paste the prompt in the coding interface (e.g., CoderPad, HackerRank, CodeSignal). Read both the prompt and test cases carefully. Most problems are clear, but ask if anything is unclear.
2. **Brainstorm**: In the first minute or two, let your intuitions flow. The prompt or test cases often hint at solutions. For example, words like "most" or "least" might suggest a greedy approach, or using DFS/DP to explore all possibilities.
3. **Narrow down**: If you have a clear intuition, go with it. Otherwise, weigh the optimal solution for the problem against what you can realistically implement given the time you have. For instance, even if Dijkstra's is optimal, I might opt for regular DFS since it's easier for me to explain and implement.
4. **Communicate**: Once you find a direction, explain it to the interviewer. If they agree, proceed; if they suggest a different approach, pivot.
5. **Implement**: Only start coding if the interviewer agrees on your approach. I find it helpful to outline the steps as comments, then flesh out each step, moving comments down as I code. Walk through your solution with a simple example.

It took me a while (3-4 months) to reach the above state. During my first 100-200 problems, I struggled even with Medium problems, often adding unnecessary data structures or special case handling as things went awry. Think of LC problems as training data for the AI that is you --- too few training examples (< 400 problems), you may miss common patterns and useful tricks. However, the "learning algorithm" matters too. As Chris Jereza [puts it](https://www.youtube.com/watch?v=lDTKnzrX6qU), many people solve thousands of problems but fail to recognize patterns, just like many go to the gym regularly but never get fit. Mastery comes not just from the number of problems you solve but from *intensity* and *reflection*. I track all problems I solved in Notion, rate my mastery of each ("no clue", "brute force", "has bugs", "bug free"), revisit problems for which I got "no clue" or had bugs, and summarize a one-sentence gist for each solution. For each company I interviewed with, I manually added company tags from LC --- this upfront effort paid off long-term. You can modify my [template](https://www.notion.so/yuanm/7667ceebb8dd4f3f840d96e43c9733c1?v=a9389f5990ed4cc691e326f92488624f&pvs=4) to fit your needs.

{{< figure src="https://www.dropbox.com/scl/fi/psk5siwwikk29kt62tq6j/Screenshot-2024-10-16-at-3.30.46-PM.png?rlkey=lkkwhf57wywsdnkredqsn4viq&st=4jq5ey6f&raw=1" width="1500" >}}

I highly recommend doing a few mock interviews before the real one, as it prepares you to solve problems while being watched. Companies like Google and Meta even provide a mock interview service where you can meet with one of their interviewers before your actual interview. That said, I'm not a big fan of excessive mock interviews --- after two or three, I already know I can think under pressure; it's just a matter of whether I know the algorithm or not. For me, practicing a wide variety of algorithms on my own is more efficient. However, if your bottleneck is nerves rather than knowledge, it might be worth investing more time in mock interviews.

### ML Coding

Some companies may ask you to code simple ML algorithms from scratch and run them on toy data, examples including K-means (K-means++), logistic/linear regression (gradient descent), and KNN. You can find example implementations in my [blog post](https://www.yuan-meng.com/posts/md_coding/). You can ask your recruiter if you're expecting LC-style coding, ML coding, or both. 

Sometimes, you may not write a full algorithm, but rather a model component (e.g., multi-headed attention --- see my other [post](https://www.yuan-meng.com/posts/attention_as_dict/) for implementation), an operation (e.g., dot product, matrix multiplication), or an eval metric (e.g,, ROC-AUC, PR-AUC, nDCG). For a complete list of ML coding problems, check out [ML Code Challenges](https://www.deep-ml.com/)!

{{< figure src="https://www.dropbox.com/scl/fi/va6t574yiv8d6ugouuj9n/Screenshot-2024-10-16-at-3.26.06-PM.png?rlkey=cwl2u4wmszqk7wkcm3vikj36i&st=1r4jf2ty&raw=1" width="1500" caption="https://www.deep-ml.com/ has a comprehensive list of ML coding problems." >}}

### Breadth-First vs. Depth-First

I'm naturally a depth-first person: For every company I interview with, I work through its LC tags twice --- once to solve them and again to review and reproduce the solution. This strategy doesn't generalize well --- with each new company, I put in loads of incremental effort. Companies like Google are notorious for asking novel questions, so if I were to grind LC again, I'd take a breadth-first approach. I'd focus on data structures and algorithms where I often make mistakes and solve representative problems for each to ensure I fully grasp the solutions.

{{< figure src="https://www.dropbox.com/scl/fi/dx1iz5hgp0g8qwyf996hh/Screenshot-2024-10-17-at-5.46.03-PM.png?rlkey=cmhjgynzmfrh9tgf5wdxbz0h1&st=wz3sp2qw&raw=1" width="1200" >}}

## ML System Design

> <span style="background-color: #FDB515">"Project" is the word many organizations use to coordinate efforts which move the needle of a specific business goal.</span> --- Gergely Orosz (2023), <em>The Software Engineer's Guidebook</em>.


### Mini Design Review

I think of the ML system design interview as a mini design review for an ML project. You first need to understand how the model moves the needle of a business goal (and push back if it doesn't), how success is measured, which users will see the model's predictions on which surfaces, what the system requirements are (e.g., latency, throughput, SLA), and how to translate the business problem into an ML problem with the right objective(s). Then, dive deep into how to develop (training data + label generation, features), evaluate (offline, online), and serve the model (e.g., data + feature + training + serving + monitoring pipelines) at web scale.

You need to drive the interview in an organized manner to cover all bases in 45 minutes or 1 hour, while also attuning to the interviewer's preferences --- some focus on model architectures or feature ideas, others on ML infra and the end-to-end ML system. Think of them as design reviewers and meet their requests to get buy-in: Discuss modeling ideas from papers and your experience if they want model details, or draw a system diagram if they ask for the end-to-end architecture.

When I was interviewing for New Grad roles, I took ML system design courses (Educative: [*Grokking the Machine Learning Interview*](https://www.educative.io/courses/grokking-the-machine-learning-interview) and read Alex Xu's [*Machine Learning System Design Interview*](https://www.amazon.com/Machine-Learning-System-Design-Interview/dp/1736049127). However, these resources don't cover enough ML depth in any given domain, be it ranking, trust & safety, or NLP. As an interviewer myself, I've seen many candidates who memorize the structure from these sources, but it's really only substance that matters. You gotta know the industry gold standard, state-of-the-art models, and best practices for ML production in your domain.

<!-- summarized learning in my [notes](https://yuanm.notion.site/ml-system-design-cd12d1028ed640e199e64ed0ed461851?pvs=4). -->

<!-- 
{{< figure src="https://www.dropbox.com/scl/fi/bld9zng0xz2r5grp6fp9i/Screenshot-2024-03-16-at-10.55.08-AM.png?rlkey=jfd9g8bi2h1dv6k2j1xlqwjz0&raw=1" width="1200" >}}

As a counterexample --- can I design an autonomous driving system in an interview, from a ranking/NLP background?  Perhaps, if I memorize some CV techniques, but my lack of experience will show, and, if offered the job, I'll run the system to the ground... My advice: Sure, read Alex's book for the structure, but the real preparation is being a good MLE who designs thoughtful, practical, reliable, and scalable ML systems and goes to design reviews to learn how others think (or critique). 
 -->

### Resources

Below are resources I still find useful for ML system design: 
1. **General advice**: [ML Eng Interview Guide](http://patrickhalina.com/posts/ml-eng-interview-guide/) by Pinterest's [Patrick Halina](https://www.linkedin.com/in/patrick-halina/?originalSubdomain=ca)
2. **Feature ideas**: [Features in Recommendation Systems (那些年，我们追过的 Feature)](https://pyemma.github.io/Features-in-Recommendation-System/)
3. **Data pipelines**: [Data Management Challenges in Production Machine Learning](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/46178.pdf)
4. **Evolution of an ML system**: [Best Practices for ML Engineering](https://developers.google.com/machine-learning/guides/rules-of-ml)
5. **State of the art**: Engineering blogs + papers in your domain (e.g., general: NeurIPS/ICLR/ICML; information retrieval: KDD/RecSys/SIGIR) 

## ML Breadth

In this new wave of AI/ML craze, many DS/SWE friends and strangers asked me how to transition into an MLE. As someone who went through this transition back in school, I hate to discourage people, but it's also no use being dishonest: Grasping machine learning theory and applying it to production systems takes years of study and experience. Anyone can copy \& paste PyTorch code online, but only seasoned ML engineers can trouble-shoot a model that's not performing as intended (see my [post](https://www.yuan-meng.com/posts/ltr/) on Airbnb's long-winded journey into DL). ML engineering is somewhat like medicine in that neural networks are as much an enigma as the human body: Highly specialized knowledge is needed to diagnose a non-working system and prescribe remedies. 

This post aims to help current ML engineers find new ML opportunities. I do not have first-hand advice on transitioning from DS/SWE to MLE for experienced hires, but someone else may have, so I encourage you to keep looking for their advice. 

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
2. **Combine theory + code**: I recommended [Machine Learning with PyTorch and Scikit-Learn](https://www.amazon.com/Machine-Learning-PyTorch-Scikit-Learn-learning-ebook/dp/B09NW48MR1) by Sebastian Raschka 2 years ago and my recommendation stands today. It's the best foundational ML book that combines theory with code, covering everything from traditional ML to SOTA, implementing both using libraries and from scratch. I recommend reading it from cover to cover and running the [code](https://github.com/rasbt/machine-learning-book). 
3. **Go deep into theory**: Last but not least, Kevin Murphy wrote 3 epic books on machine learning theory, the most recent ones being [Probabilistic Machine Learning: An Introduction (2022)](https://probml.github.io/pml-book/book1.html) and [Probabilistic Machine Learning: Advanced Topics (2023)](https://probml.github.io/pml-book/book2.html). Even for concepts you and I can't be more familiar with such as attention, he still shed fresh light on them every now and then (see my [post](https://www.yuan-meng.com/posts/attention_as_dict/) on the "attention as soft dictionary lookup" analogy). As an MLE who wishes to connect the dots scattered over the years, I think Kevin's books are the dream. 
    {{< figure src="https://www.dropbox.com/scl/fi/f2624w8e5dxe77cko9791/Screenshot-2024-03-16-at-2.57.10-PM.png?rlkey=8a51dwek7aiqdwtbn8a6ix0ao&raw=1" width="500" caption="[Table of Content](https://probml.github.io/pml-book/book2.html#toc) of Probabilistic Machine Learning: Advanced Topics" >}}

<!-- I don't specifically prepare for ML Breadth interviews or keep Notion notes --- I did that 2 years ago (see my [old notes](https://www.notion.so/yuanm/core-ml-1930f2267ce942c984b005c1bb62d429?pvs=4)).  -->

Chip Huyen curated some [questions](https://huyenchip.com/ml-interviews-book/contents/8.1.2-questions.html) in her ML interview book and a Roblox Principal MLE also wrote a [nice article](https://medium.com/@reachpriyaa/how-to-crack-machine-learning-interviews-at-faang-78a2882a05c5) that contains many actual questions I've encountered, which you can go over before an ML breadth interview to refresh your memories.

## ML Depth 

If you interview with companies such as Meta, LinkedIn, or Pinterest, you'll choose a team later. At most companies (e.g., Netflix, Apple, Uber, and even Google now), MLE candidates interview with specific teams looking for domain experts who are familiar with the gold standards in their fields and the latest advancements.

### Paper Tracker

It would take many lifetimes to read all papers that ever got published in your field. That said, good ML engineers should be familiar with the seminal works and "new classics" in their domain and make paper reading a regular habit. I used to save paper PDFs in random folders and would often forget which ones I had read, let alone what I learned from them. Then, inspired by the coding tracker, I created a paper tracker to keep track of papers/books/blog posts I read on different topics. 

{{< figure src="https://www.dropbox.com/scl/fi/m5nh0fyhaxe4f282kvkyo/Screenshot-2024-03-16-at-3.15.57-PM.png?rlkey=twfh7pw2n1c0mlwelmtil6wjc&raw=1" width="1500" >}}

For each entry, I write a one- or two-sentence TL;DR, and after clicking on the paper title, I jot down detailed notes. The example below is for the [Llama 2](https://arxiv.org/abs/2307.09288) paper. 

{{< figure src="https://www.dropbox.com/scl/fi/g8qics73srz7jbexo3994/Screenshot-2024-03-16-at-3.26.40-PM.png?rlkey=3vcyjsh8wj3xwdkqj22muzw93&raw=1" width="1000" >}}

### Ranking

Ranking is a revenue generator for consumer-facing companies --- without search, recommendations, or ads, they can't make money. This is why ranking is one of the most impactful and exciting domains, with the most job openings nowadays (arguably tied with NLP). I collect the ranking papers (search/rec/ads) I read in this [repo](https://github.com/Yuan-Meng/search_rec_ads_papers).

Earlier this year, I summarized classic search ranking papers in a blog post, [*An Evolution of Learning to Rank*](https://www.yuan-meng.com/posts/ltr/). You can find the original papers under ["Learn More"](https://www.yuan-meng.com/posts/ltr/#learn-more).

Recommendation is the sibling of search, where users don't submit a query, yet we still read their minds and show appealing content. To gain a general understanding of this domain, I recommend [深度学习推荐系统](https://www.amazon.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%EF%BC%88%E5%85%A8%E5%BD%A9%EF%BC%89-%E5%8D%9A%E6%96%87%E8%A7%86%E7%82%B9%E5%87%BA%E5%93%81-%E7%8E%8B%E5%96%86/dp/7121384647/ref=sr_1_1?crid=XZSJ6ZW1282A&dib=eyJ2IjoiMSJ9.BxuHruDCtEpyjInRfwvRFn7pyNHfnLqM9I7MhiVO4QjP5NwIPzDAXhfcdM4N-9cc.-Amg_ZFv605zkwdYZmfQJIg4d3GGoacM4rlgWcwz29c&dib_tag=se&keywords=%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F&qid=1710629348&sprefix=%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%2Caps%2C141&sr=8-1) by Zhe Wang, who was the Head of Engineering at TikTok. This book not only reviews the evolution of recommendation models, from collaborative filtering to deep learning architectures, but also offers practical insights into productionizing such models. Unfortunately, for non-Chinese speakers, this book is only available in Chinese; the book content is slightly outdated, too, as models published after 2020 (e.g., [DCN V2](https://arxiv.org/abs/2008.13535), [DHEN](https://arxiv.org/abs/2203.11014)) are not covered. I also find Gaurav Chakravorty's [blog](https://substack.com/@recsysml) and [repo](https://github.com/gauravchak) to be gold mines of practical rec models knowledge. For instance, in this [two-tower model repo](https://github.com/gauravchak/two_tower_models), Gaurav starts with a barebones structure and gradually introduces components like user history embeddings, position debiasing, and a light ranker, which is highly educational.

Ads have extra concerns (e.g., bidding, auction, calibration) on top of organic ranking. I've become increasingly more fascinated by ads, which is arguably the most technically challenging ranking domain and directly impacts company revenues.

### NLP 

As a Cognitive Science Ph.D. student, I was interested in all aspects of the human intelligence except for language, thinking that language is an indirect, inaccurate byproduct of thought 😅. Back in Summer 2021 when I was driving to LA, I listened to a Gradient Dissent [episode](https://www.youtube.com/watch?v=VaxNN3YRhBA) where Emily Bender argued that to interact with language models as we do with humans, we must treat them as equal linguistic partners rather than as mechanistic pattern recognizers. Back then, that day seemed far away. Now with the success of ChatGPT and ever more powerful new LLMs, NLP has become the fastest growing field in AI/ML, with far more active research, new startups, and headcount than any other ML fields. I got into NLP only because I was assigned to work on Query Understanding after joining DoorDash as a new grad. If ranking has always fascinated me, then NLP is a required taste I've come to appreciate.

To get into NLP, first read [*Natural Language Processing With Transformers*](https://www.amazon.com/Natural-Language-Processing-Transformers-Applications/dp/1098103246) by Hugging Face --- it's the best textbook on transformer-based models, with intuitive explanations and practical code examples, covering a wide range of NLP tasks including sequence classification, named entity recognition, question answering, summarization, generation, etc.. It also talks about model compression, training without enough data, and pretraining LLMs from scratch, which come handy in production. 

While awesome, the HF book leaves you wanting for details from the original papers. To learn more, check out Sebastian Raschka's [LLM reading list](https://magazine.sebastianraschka.com/p/understanding-large-language-models), which starts with the original transformer in *Attention is All You Need* and its predecessors, moves on to the post child of each iconic model architecture (e.g., encoder-only: BERT; decoder-only: GPT; encoder-decoder: BART), and ends on SOTA models, training techniques (e.g., [LoRA and QLoRA](https://lightning.ai/pages/community/lora-insights/)), and architectural innovations (e.g., FlashAttention, Rotary Positional Embeddings). To understand more deeply how each component of the original transformer can be optimized, I recommend Lilian Weng's blog post [*The Transformer Family Version 2.0*](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/). To get an in-depth understanding of latest models (e.g., Llama 2, Mistral, Mamba) and fundamental techniques (e.g., quantization, RAG, distributed training), I suggest watching Umar Jamil's amazing YouTube [videos](https://www.youtube.com/@umarjamilai) and reading his neat implementations ([GitHub](https://github.com/hkproj)). Andrej Karpathy has also created a great course [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) and legendary educational repos (e.g., [miniGPT](https://github.com/karpathy/minGPT), [nanoGPT](https://github.com/karpathy/nanoGPT), [minbpe](https://github.com/karpathy/minbpe)), which use minimalist code to train minimalist models.

For important topics not covered, you can read representative blog posts and literature review to catch up, such as [prompt engineering](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/), [RLHF](https://huggingface.co/blog/rlhf), and [LLM evaluation](https://arxiv.org/abs/2307.03109).

There will never be a day when you think you know enough about NLP. Neither does your interviewer. Most NLP interviews focus on the basics, such as the original transformer, common tokenizers, popular models (e.g., BERT/GPT/BART/T5/Llama 3.2), and widely used techniques (e.g., RLHF, RAG, LoRA). Focus on these in your interview prep and don't worry about being asked about papers that came out yesterday.

## Behavior Interview

### Tell Me A Time When...

I used to stammer my way through behavior interviews because I didn't prepare for some questions I was asked. Then Amazon recommended the [*Behavioral Interviews for Software Engineers*](https://www.amazon.com/Behavioral-Interviews-Software-Engineers-Strategies/dp/B0C1JFQYCR/ref=tmm_pap_swatch_0?_encoding=UTF8&qid=&sr=) book to me, which has 30+ behavior questions one will ever get asked. I don't find the answers particularly applicable to MLE candidates --- the question list itself is the treasure. Just like I did for coding and reading, I created a tracker for BQ questions, coming up with the best example for each question and putting my answer in the STAR (situation, task, action, result) format. 

{{< figure src="https://www.dropbox.com/scl/fi/ex0xgnv4noz5x24y44fuu/Screenshot-2024-10-17-at-5.19.13-PM.png?rlkey=sgyodjvv5brws04495avew5qh&st=gigx7p1m&raw=1" width="1500" >}}

Any example you provide should be a true reflection of your character, but think carefully about which examples to provide --- don't paint yourself in a bad light 😂. Reading Merih Taze's [*Engineers Survival Guide*](https://www.amazon.com/Engineers-Survival-Guide-Facebook-Microsoft/dp/B09MBZBGFK) helped me understand what behavior makes a good engineer --- model not your answers but your words and deeds after it. I wish I read it in my first year at DoorDash, which would've saved me many sweat and tears from reaching alignment, getting visibility, making impact, helping my managers to help me, resolving conflicts, and whatnot. Gergely Orosz's [*The Software Engineer's Guidebook*](https://www.amazon.com/Software-Engineers-Guidebook-Navigating-positions/dp/908338182X) is another great book to read if you have the time.

### Project Deep Dive

In some sense, the project deep dive is like an extended version of "tell me about the project you're most proud of", but goes into all technical details imaginable ---

- **Background**: What business problem was the model trying to solve? 
- **Solution \& alternatives**: How did you solve the problem? Which alternatives did you consider? What were the trade-offs? How did you align the team? 
- **Challenges**: Which technical and XFN collaboration challenges did you face?
- **Impact**: In the end, what impact did you make? How did you measure it?
- **Reflection**: What were you most proud of and what would you change?

Choose a successful project you drove from end to end, or the lack of ownership or impact will backfire. Not only that, choose a project for which you made wise decisions by carefully considering all alternatives and thoughtfully balancing all sides (e.g., business goals/priorities, eng excellence, infra limits...). If you made a big mistake (e.g., not involving a function until it was too late, making hasty decisions without thinking ahead) in a project, talk about other projects --- this is your chance to showcase your technical + XFN collab prowess, not a post mortem.

When preparing for this round, I re-read my annual and mid-year perf reviews and made slides for each significant project I delivered. I highly recommend finding a friend not even in your field and practicing getting your stories across to them.

# The "Moment of Truth"

## Don't Sweat It

Back in school, I often lost sleep the night before an interview, fearing that any wrong word could cost me the job. Nowadays, I work, sleep, play with cats, and have hotpot as usual before interviews, knowing results are determined long before. If I'm "the one", all I need is to chat with the interviewer about topics we're both passionate about. For instance, I can spot a ranking expert from a brief conversation, even if they occasionally stammer or forget a few minor details --- so can my interviewer. If I make grave mistakes and can't recover what I forget through first-principle thinking, then I'm not "the one", and not hiring me is for the best. It's like being rejected after a few dates --- it's unpleasant, but it's far better than marrying someone profoundly incompatible and divorcing them later.

<!-- If you know me by now, you might've guessed that I created an interview progress tracker, using one table to track the stage (**scheduling**: sent availability but coordinator hasn't finalized the schedule; **in progress**: interview scheduled for a future date; **done**: completed the given interview round) of each round (e.g., recruiter/HM chat, phone screen, onsite) with each company and another table to track what each round entails. When interviewing with a specific company, I filter down to the rounds asked by that company and review the corresponding Notion Pages.  -->

## "Imposter Syndrome"

> <span style="background-color: #FDB515">However, maybe you're working somewhere where you haven't dealt with large scale data or they aren't using any type of sophisticated ML techniques. This is a great motivation to get a new job!</span> --- *[Systems Design Interview Guide](http://patrickhalina.com/posts/systems-design-interview-guide/)*, Patrick Halina, Staff MLE @Pinterest

When I first started my interview prep journey, I struggled with imposter syndrome: If I'm already building the most sophisticated models from SOTA papers, then why would I need to look for new opportunities? But if I haven't worked on such models, then why would anyone take a chance on me? If I talk about new ML techniques I've only learned from papers to land a job, am I letting my words speak louder than my actions, which goes against the kind of ML engineer I aspire to be?

Pinterest ML engineer Patrick Halina's words has brought me great solace: It is *because* you desire new challenges and knowledge that you dream of going somewhere new --- if you stay put, nothing will ever change. The tears, sweat, discipline, drive, and passion you've put into pursuing your dream will fuel your new journey.
