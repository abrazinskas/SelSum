# AmaSum: abstractive opinion summarization dataset

This folder contains the **AmaSum** dataset with **33,324** abstractive summaries and their associated customer reviews for **31,483** Amazon products. 
Each product has **verdict**, **pros**, and **cons** as the summary; see the example below. 


**Verdict:**

*A go-to study guide that project managers can use as a reference long after passing the PMP exam.*


**Pros:**

* *Goes beyond the basics to give good insight into the PMI*
* *Well edited with few to no errors in spelling or sample problems*
* *Sample questions are on par with actual exam questions*
* *Plenty of exercises and practice questions*

**Cons:**

* *Not a standalone resource, but works well in tandem with other prep resources*
* *Rita’s process chart and guide are confusing and not useful to some*


## Dataset formats

The dataset is available in its raw format (JSON) and pre-processed for FAIRSEQ models. JSON raw files contain not only reviews and summaries but also additional meta information, which can be useful for summarization. FAIRSEQ formatted files can be used to train FAIRSEQ models if they are binarized beforehand. For binarization instructions, please refer to the [preprocessing folder](../preprocessing).
Please **read the license** before downloading the dataset.


1. [JSON raw 1](https://abrazinskas.s3.eu-west-1.amazonaws.com/downloads/projects/selsum/data/raw_min_10_revs.zip): minimum 10 reviews in each ASIN file. Min_len=10, max_len=120;
2. [JSON raw 2](https://abrazinskas.s3.eu-west-1.amazonaws.com/downloads/projects/selsum/data/raw_min_10_max_100_revs.zip): minimum 10 reviews in each ASIN file. Min_len=10, max_len=120, max_revs=100;
3. [FAIRSEQ formatted](https://abrazinskas.s3.eu-west-1.amazonaws.com/downloads/projects/selsum/data/form_min_10_max_100_revs.zip): minimum 10 reviews in each ASIN file. Min_len=10, max_len=120, max_revs=100.


## License

The dataset can be used for **non-commercial** and **educational purposes** only.
See `LICENSE.txt` for more details.