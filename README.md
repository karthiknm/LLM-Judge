# LLM-Judge for Summarization

This repository contains the code and resources for distilling a large language model (LLM) judge into a smaller open-weight model for scalable summarization evaluation.

Recent work shows that strong LLMs can evaluate summaries more reliably than traditional lexical metrics such as ROUGE or BERTScore. However, using large proprietary models as judges can be expensive and difficult to deploy at scale. This project investigates whether the evaluation behavior of a strong LLM judge can be transferred to smaller models through distillation.

The system trains compact student models to imitate rubric-based judgments produced by a stronger teacher model.


## GPT Annotated Dataset
- The GPT 5 Mini annotated dataset for this work can be found at `https://drive.google.com/file/d/1EWf4DZtYFDdldjel4sT24LaFfa5_t4Ru/view?usp=sharing`

## Model weights
- The model weights for the Qwen 3B model is available at `https://drive.google.com/file/d/1C-OtxrzitICP9Dxzb9vdzQvxkIZ5A9Wt/view?usp=sharing`
- The model weights for the Qwen 1.5B model is available at `https://drive.google.com/drive/folders/1kA74LaTS1PGdgGfPD7ICdbCCNSu4MvQS?usp=sharing`

## Synthetic Summary Generation
We use the following summarization models to generate 5000 synthetic summaries before passing them to the teacher model (GPT 5 Mini) for further annotation.
- `facebook/bart-large-cnn`
- `google/pegasus-cnn_dailymail` or `google/pegasus-xsum`
- `Sachin21112004/distilbart-news-summarizer` 


## Teacher Annotation 
We use the GPT-5 Mini model for annotation of the generated summaries.
For each `(source, generated_summary)` the annotation is done for the following:
- `coherence` (1-5)
- `consistency` (1-5)
- `fluency` (1-5)
- `relevance` (1-5)
- `reasoning` (concise evidence-grounded explanation)

## Student Model Training

Student models are trained to reproduce the teacher outputs using supervised fine-tuning.

**Base models**

- Qwen2.5-1.5B  
- Qwen2.5-3B  

**Training setup**

- Parameter-efficient fine-tuning using LoRA  
- Autoregressive generation of structured JSON outputs  
- Objective: token-level cross-entropy on teacher outputs  

The model learns to predict both:

- rubric scores  
- accompanying reasoning traces  

This allows the student judge to approximate not only the scoring behavior of the teacher but also the explanatory patterns behind its judgments.

---

## Demo

A small demo application was developed to showcase the distilled evaluation system.  
The application allows users to input a document and a candidate summary and receive rubric-based evaluation scores along with explanations.

**Demo video**

https://drive.google.com/file/d/1_drGImqm8EdT04yQHVzoHYFL0S7LRzKV/view

The code for the demo application is included in this repository.

