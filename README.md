# Project_3: The Paradox between Autism and Giftedness
### Overview

In this notebook, I use a simple logistic regression classifier on a dataset of approximately 2000 reddit posts to predict whether the post was written by an individual with Asperger's, or whether the post is written by an individual from the Gifted community. I focus on the interpretability of simple classification models and what that means for text data.

Disclaimer: The steps I've taken here constitute a good baseline to get started on an NLP project, though by no means are they comprehensive.

The project is also based on a few key assumptions. They are as follows: 
1. For simplicity, I assume the posts in r/aspergers and r/Gifted subreddits were written by individuals who have been clinically diagnosed with Asperger's and officially tested as Gifted respectively. 
2. Asperger's and Autism are used interchangeably in the context of this project. 
3. While the posts in the r/aspergers subreddit were written in first person, and largely by individuals who have been clinically diagnosed with Autism, the same cannot be said about the posts in the r/Gifted subreddit. In the r/Gifted subreddit, moderators do not discriminate againsts posts that are written by parents who suspect their children are gifted and inividuals who suppose they are gifted. Upon further inspection, there is a significant number of posts that were not written by individuals who have been officially tested as gifted. This means that the actual number of posts in the Gifted corpus should be less than the total number of posts we have scraped from the r/Gifted subreddit (approximately 1000). For the sake of making sure that there is balanced classification, I have chosen to keep these posts. If these posts were removed, there would be the issue of imbalanced classification - which can be corrected by tuning the classification threshold. 

---

### Problem Statement

The provision of healthcare in Singapore has become more challenging due to a couple of reasons. One salient challenge is the poor design of systems and operational inefficiencies lead to significant waste in Healthcare. Healthcare waste is incurred any time a patient, or healthcare professional engages in unnecessary medical activity - ranging from preventable mistakes in medical care, to misdiagnoses, provision of unnecessary treatments, and procedural inconsistencies. Research has shown that waste in OECD countries amount to as much as one-fifth of the country’s healthcare expediture and this can amount to a staggering sum.

In Singapore, part of the costs (and risks) are first borne by the state (tax-payers), and/or private health insurers. The remaining costs (and risks) are borne by individuals and their families. There is incentive for the stakeholders mentioned above to eliminate waste to achieve cost-savings. The Singapore government has begun to incorporate technology into various care models to overcome the various cost-based challenges in the healthcare sector, without compromising on the quality of care. Most of these technologies are procured from private healthtech companies and start-ups.

You work in the Research and Development (R&D) department of a healthtech startup in Singapore. The company has been enlisted by the Ministry of Health Holdings (MOHH) to create a simple diagnostic tool to rule out conditions that are commonly misdiagnosed. After the development of a differential diagnosis, this tool will be a core feature in the series of additional tests that will conducted by healthcare professionals to rule out either Autism or Giftedness. Healthcare professionals will be able to come to a final diagnosis that is more precise, reducing the likelihood of misdiagnosis produced by the existing slew of subjective tests. A digital health intervention pilot study can be implemented to evaluated the cost-effectiveness of this technology.

Through this simple NLP-model, you will also be able to supplement the current diagnostic criteria in the DSM-5 for Autism with additional information. This makes for a more robust litmus test in the diagnosis process. 

**Citations:**
- Ooi, Low & Koh, Gerald & Tan, Lawrence & Yap, Jason & Chew, Samuel & Jih, Chin & Fung, Daniel & Sing, Lee & Lee, Patricia & Boon, Lim & Lim, Ruth & Low, James & Sachdev, Ravinder & Seah, Daren & Yeng, Siaw & Chiu, Tan & Teo, David & Tiwari, Satyaprakash & Tym, Wong & Scott, Richard. (2015). National Telemedicine Guidelines of Singapore.
- Lim, P. (2018). "Specific Language Impairment in Children with High-Functioning Autism Spectrum Disorder." Inquiries Journal, 10(05).
- Nakhooda, F. (2021). The Bottom Line (Healthcare): Cutting Healthcare Waste: A Win-Win for Providers, Payers, Patients. The Business Times, Opinion & Features.

---

### Datasets

There are 3 datasets included in the [`data`](./data/) folder for this project. These correspond to subreddit posts from r/aspergers, and r/Gifted. The 3rd dataset is the combined dataframe of posts from r/aspergers and r/Gifted for Exploratory Data Analysis and Preprocessing & Modeling purposes. 

* [`aspergers_df.csv`](./data/aspergers_df.csv): Records of posts from r/aspergers subreddit, consisting of the title, body, and 10 top level comments.
* [`gifted_df.csv`](./data/gifted_df.csv): Records of posts from r/Gifted subreddit, consisting of the title, body, and 10 top level comments.
* [`df.csv`](./data/df.csv): Combined records from r/aspergers and r/Gifted subreddits. 
---
### General Findings from EDA
WordCloud generated from Asperger's Corpus before Data Cleaning and Pre-Processing: <br><br>
![wordcloud_asp](https://user-images.githubusercontent.com/126059186/226282460-c56b248c-2c86-4788-afbe-e7c984d978c6.png)

WordCloud generated from Gifted Corpus before Data Cleaning and Pre-Processing: <br><br>
![wordcloud_gifted](https://user-images.githubusercontent.com/126059186/226282078-98c7ced6-426c-4c8c-9b78-fccc67c548e1.png)

* WordClouds after Data Cleaning and Pre-Processing should be created to visualise the salience of these processes.

---

### Data Dictionary after Preprocessing, for Modeling

| Feature | Type | Dataset | Description |
|---|---|---|---|
| **text_feature** | *object* | *df* | *orginal posts from each subreddit* |
| **diagnosis** | *int64* | *df* | *ground truth (either 1 for 'asperger's' and 0 for 'gifted')* | 
| **text_clean** | *object* | *df* | *corpus of cleaned text, after removal of embellishments such as capitalisation, special/non-roman characters, new linews, punctuation, links, username, numbers, double spacing* |
| **no_stop_words** | *object* | *df* | *list of cleaned text after removal of stop words and word tokenization* |
| **lemma** | *object* | *df* | *corpus of cleaned text after lemmatization* |

---

### Brief Summary of Analysis
**Problem Statement: What other information can we extract using NLP-models to create a more robust diagnosis criteria for Asperger's?** 

#### Key Takeaways
- Based on exisitng research, a gifted child may present an extensive and advanced vocabulary with a rich verbal style. On the other hand, a child on the autism spectrum may have an advanced use of vocabulary, but they may not have full comprehension of the language they use. They may also have a less inviting verbal style that lacks the engagement of others. 

- Additional differences gleaned from the NLP-task include:
1. Characteristics of Individuals with Asperger's
> * Individuals who have been clinically diagnosed with Asperger's can be preoccupied with social skills in life and at work. 
> * They tend to have special interests that they fixate on. 
2. Characteristics of Gifted Individuals
> * Gifted individuals who have been tested as gifted can be concerned over the neglect of their spatial ability. 
> * They tend to value Justice.

---

### Conclusion and Business Recommendation
Maximize the AUC-ROC by choosing the best performing model, set of transformations and hyperparameters. This minimizes the number of misdiagnosed patients (Type I Errors / False Positive or Type II errors / False Negative), resulting in ↑ waste in healthcare. 

Tune the classification threshold to an optimal cut-off to achieve business objective. 
- Typically, you move the threshold away from a 50% cut-off in one of the two following cases: 
1. High Recall, Low Precision - 25% Threshold
> * In this scenario, you want High Recall, which means that all individuals who potentially have Aspergers will be notified to go for further screening and interventions (True Positive Rate ↑ ), even if it means falsely classifying some Gifted individuals as 'Aspergers' (False Positive Rate ↑), risking Low Precision. 
2. Low Recall, High Precision - 75% Threshold
> * In this scenario, you want High Precision, which means that you only want individuals who truly have Aspergers to be notified for further screening and interventions (↓ True Positive Rate and ↓ False Positive Rate), even if it means missing out on / leaving out some individuals who could potentially have Aspergers (False Negative Rate ↑), risking Low Recall. 
- Since cost-savings (elimination of waste) is the goal, you should be looking to ↑ optimal classification threshold to 0.6 based on the evaluation. I would argue that it is also unethical to have an individual's unique intellectual abilities be eclipsed by an Autism diagnosis (False Positive / Type I error. Hence, there is a strong case to be made for ↓ False Positive Rate (Type I Error) at the expense of ↑ False Negative Rate (Type II Error) - scenario 2. 

Conduct a Digital Health Intervention Pilot Study to evaluate the cost-effectiveness of this intervention. Healthcare KPI should be a cost-based metric as the objective is the elimination of waste and cost-reduction.
> * Intervention group : to be diagnosed using existing subjective diagnostic criteria (DSM-5)
> * Non-intervention group : to go through a diagnostic process assisted by NLP-based diagnostic tool

Note that the optimal classification threshold should differ across industries. 
> * If you are providing this technology to stakeholders in the healthcare sector looking to acheive cost-savings / minimize waste, focus on ↓ False Positive Rate (Type I Error), even if it is at the expense of ↑ False Negative Rate (Type II Error). 
> * If you are providing this technology to stakeholders in the education sector looking to acheive cost-savings / minimize waste, focus on ↑  False Positive Rate (type I Error), even if it is at the expense of ↓ False Negative Rate (Type II Error).

---

### Limitations and Areas of Future Research
1. This model is unable to classify or account for individuals with dual diagnoses. The solution could be to create a multiclass classifier instead.
2. Most of the posts seem to be based in other geographical regions, hence the samples might not be an accurate representation of the Asperger's and Gifted communities in Singapore. The solution could be to identify local sources of text data.
3. Intepretability of Logistic Regression Models. For a more robust analysis, use LIME and/or SHAP to explain the coefficients and weights of the individual text features.
4. The Area Under Curve is currently 0.94. Even though it is a good score, there are still overlapping regions indicative of the presence of False Positive (Type I Errors) and False Negative (Type II Errors) classifications. Possible solutions to enhance AUC-ROC: 
> * Increase rigour of data cleaning process.
> * Regularization to reduce the weights of the less important key words and increase the weights of the key words which are more salient.
> * Feature Engineering: Consider adding other quantitative features for a more robust model. 
5. Oversimplification of complex NLP-task attributed to implicit assumptions made - refer to overview for list of assumptions.
