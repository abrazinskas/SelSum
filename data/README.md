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

The dataset is available in its raw format (JSON) and pre-processed for FAIRSEQ models. In the latter case, sources (input reviews) and targets (summaries) are in separate files. Reviews in the source files are separated by a special symbol. JSON raw files contain not only reviews and summaries but also additional meta information, which can be useful for summarization. FAIRSEQ formatted files can be used to train FAIRSEQ models if they are binarized beforehand. For binarization instructions, please refer to the [preprocessing folder](../preprocessing).


| Format | Size | Min revs | Avg revs | Max revs | Link |
| :---:| :---: | :---:| :---: | :---: | :---: |
| JSON  | 374.8 MB | 10  | 76 | 100 | [Download](https://abrazinskas.s3.eu-west-1.amazonaws.com/downloads/projects/selsum/data/raw_min_10_max_100_revs.zip) |
| JSON | 1.1 GB | 10 | 326 | 2361 |[Download](https://abrazinskas.s3.eu-west-1.amazonaws.com/downloads/projects/selsum/data/raw_min_10_revs.zip) |
| FAIRSEQ | 323.8 MB | 10 | 76 | 100 | [Download](https://abrazinskas.s3.eu-west-1.amazonaws.com/downloads/projects/selsum/data/form_min_10_max_100_revs.zip) |
| FAIRSEQ | 807.1 MB | 10 | 326 | 2361 |[Download](https://abrazinskas.s3.eu-west-1.amazonaws.com/downloads/projects/selsum/data/form_min_10_revs.zip) |

Reviews were also filtered, with **10** and **120** minimum and maximum lengths, respectively.

## License

The dataset can be used for **non-commercial** and **educational purposes** only.
See `LICENSE.txt` for more details.