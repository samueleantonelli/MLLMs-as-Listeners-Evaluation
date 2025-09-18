# Referring-Expression-Comprehension
This work focuses on the Referring Expression Comprehension (REC) of models with Models Vs Humans Speakers.
It is built on two main streams of experiments:
1. **Evaluation of MLLMs-as-Listeners:** MLLMs are evaluated in the role of Listeners in a Referring Expression task, understanding whether such models perform better with human- or model-geneated descriptions of targets inside images.
2. **Implicit evaluation of MLLMs:** internal states of MLLMs are evaluated using probing, to understand whether such models encode the pragmatic concept of discriminativity, central to solve Referring Expression tasks. 

---

# 1. Speaker Comparison with Qwen Listener on RefOI Dataset

This repository provides a codebase and analysis pipeline to **compare the performance of different Speaker models** (i.e., description generators) using **VLMs** as Listeners for grounding tasks on the [RefOI dataset](https://huggingface.co/datasets/Seed42Lab/RefOI).
It is built using QWEN2.5-VL as principal listener, but modifying the proper line it is adaptable to any other VLM.

## Background
* **Dataset:** [RefOI](https://huggingface.co/datasets/Seed42Lab/RefOI), designed for referring expression grounding with bounding boxes.
* **Task:** Given an image and a description (from either a human or a model Speaker), Qwen predicts the bounding box for the referred object.

Each sample contains:
* Plain image
* Target description
* Speaker identity (model/human)
* Type of utterance (brief/default for models, spoken/written for humans)
* Ground-truth bounding box
* Token statistics (e.g., redundancy)
* Co-occurrency
_for more detailed info visit the dataset's HuggingFace._

## Evaluation Pipeline
1. **The Listener receives** the plain image and a description.
2. **It outputs predicted bounding box coordinates.**
3. **The code evaluates the Intersection over Union (IoU)** between predicted and gold standard box (`IoU > 0.5` is considerated correct, so positive accuracy).
4. If IoU < 0.5, the sample is saved in a folder with both the generated and golden bounding box for qualitative error analysis.


## Key Results (using QWEN2.5-VL as Listener)

| Speaker    |  % Brief Correct | % Default Correct | % Total Correct |
| :--------- | :--------------: | :---------------: | :-------------: |
| LLava7B    |        86%       |        91%        |       89%       |
| LLava13B   |        93%       |        90%        |       91%       |
| LLava34B   |        90%       |        88%        |       89%       |
| CogVLM     |        89%       |        88%        |       88%       |
| GPT-4o     |        89%       |        89%        |       89%       |
| XCompose   |        84%       |        89%        |       87%       |
| GLaMM      |        59%       |        43%        |       52%       |
| Mini CPM‑V |        89%       |        91%        |       90%       |
| **Average**|      **88%**     |      **87%**      |      **88%**    |

| Speaker    |  % Spoken Correct| % Written Correct | % Total Correct |
| :--------- | :--------------: | :---------------: | :-------------: |
| **Humans** |     **88%**      |      **87%**      |     **88%**     |

_*Brief= the Speaker is prompted to output short descriptions;
*Default= the Speaker is simply asked to output a description;
*Spoken= the human gave the description by talking;
*Written= the human gave the description by writing._

## Insights
* **Moderl VLMs are robust** to the source of the description and achieves very high accuracy.
* **Human speakers** naturally adapt their utterances to task complexity; models generally do not.
* **Description redundancy** does not significantly impact SOTA VLMs performance: models are not sensitive to longer or more verbose utterances as long as they are informative.

## Usage
1. **Clone the repository.**
2. **Install requirements** (see `requirements.txt`).
3. **Run the evaluation pipeline**:

   ```bash
   python evaluate_listener.py 
   ```

4. **See the results and analysis** in the generated report or use the provided notebooks for visualization.

## References
* Ma et al. (RefOI Dataset): [https://huggingface.co/datasets/Seed42Lab/RefOI](https://huggingface.co/datasets/Seed42Lab/RefOI)
* [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen-VL)
* For more details on the evaluation and pipeline, see `docs/`.

---

**Note:**
The README above is focused ONLY on comparisons with Qwen as the Listener. It is possible to adapt the code to use also different VLMs as listeners.



----


# 2. Probing Analysis and Auxiliary Experiments

In addition to the explicit evaluation with Qwen as Listener, we conducted **implicit probing experiments** and **auxiliary analyses** to investigate whether pragmatic discriminativity is encoded in MLLMs and what factors shape model performance.

## Probing Setup
- **Goal:** test if QWEN2.5-VL internally represents the notion of *discriminativity* (distinguishing between discriminative and underdiscriminative utterances).
- **Method:** embeddings from each model layer were extracted and used to train a linear probe (logistic regression).
- **Data:** pairs of image + description from RefOI, labeled by human annotators. Such data was shared by refOI authors (https://huggingface.co/datasets/Seed42Lab/RefOI) based on the human performances in the referring expression task.
- **Evaluation:** Accuracy, F1, AUROC across human vs model sources.

## Key Results (Implicit Evaluation via Probing)

The probing analysis tested whether **QWEN2.5-VL embeddings** encode the distinction
between **discriminative** and **underdiscriminative** utterances.

### Overall Performance

| Investigation type                     | ACC  | F1   | AUROC |
|----------------------------------------|------|------|-------|
| Normal image + caption                 | 0.80 | 0.78 | 0.90  |
| Boxed image + caption                  | 0.79 | 0.77 | 0.88  |
| Normal image + caption (no box ref)    | 0.79 | 0.78 | 0.88  |
| **Text Only**                          | 0.74 | 0.73 | 0.84  |
| **Random labels (control)**            | 0.49 | 0.43 | 0.50  |

* Performance was best in the **Image + Text** setting (ACC ≈ 0.80, AUROC ≈ 0.90).  
* **Text Only** also gave reasonably high results (AUROC 0.84), showing that the probe
  could largely rely on linguistic features.  
* The **Random Labels** control collapsed to chance, ruling out dataset artifacts.  
* Differences between conditions with or without bounding boxes, or with stripped
  references, were minimal: the probe did not really exploit visual cues.  

---
# 3. Further Analyses
Further analysis on the probe were operated:

### Breakdown by Speaker Source

| Source          | ACC  | F1   | AUROC | N test |
|-----------------|------|------|-------|--------|
| CogVLM          | 0.88 | 0.69 | 0.95  | 162    |
| GLaMM           | 0.90 | 0.70 | 0.80  | 41     |
| GPT-4o          | 0.72 | 0.76 | 0.95  | 143    |
| LLaVA-13B       | 0.72 | 0.51 | 0.84  | 144    |
| LLaVA-34B       | 0.68 | 0.57 | 0.92  | 139    |
| LLaVA-7B        | 0.80 | 0.49 | 0.87  | 141    |
| MiniCPM-V       | 0.64 | 0.64 | 0.89  | 146    |
| XCompose        | 0.74 | 0.78 | 0.84  | 134    |
| **Humans (spoken)** | 0.98 | 0.99 |  /   | 91     |
| **Humans (written)**| 0.95 | 0.98 |  /   | 87     |
| **Models (avg.)**   | 0.75 | 0.61 | 0.89  | 1050   |
| **Humans (avg.)**   | 0.97 | 0.96 |   /   | 177    |

* **AUROC values** above 0.90 for several models indicate that some discriminative signal is present, but the **F1 scores** remain modest, reflecting noisy boundaries.  
* AUROC could not be computed for humans due to the lack of negative examples in the dataset.  

**Takeaway:** Probing shows that pragmatic discriminativity is partially encoded in
MLLMs, but unevenly across Speaker sources and strongly dependent on textual rather
than visual cues. 

## Layer-Wise Analysis
- Discriminativity signal appears early (AUROC > 0.80 by layer 2).
- Performance increases through mid layers, peaking around AUROC ≈ 0.90.
- Text Only and Image + Text conditions follow similar patterns, showing that **probes rely mostly on textual cues**, not multimodal integration.

## Linguistic Analysis
To test whether probes captured pragmatic competence or linguistic regularities:
- **Spatial terms, definites, cardinals, colors** are more frequent in discriminative utterances (+256%).
- **Sentence length**: discriminative descriptions are often longer (+19%).
- **POS profiles** correlate with discriminativity.

Conclusion: probing separability is largely driven by **surface linguistic markers**, not genuine pragmatic encoding.

## Single-Case Analysis
Comparing Text Only vs Image + Text probe outputs:
- The **image acts as a compensatory channel**, helping when text cues are insufficient.
- Sometimes the image corrects over-signaling in text-only predictions.
- Visual information plays a **secondary corrective role**, not a deeply integrated one.
- 
----

# General Insights
The combination of explicit and implicit evaluations paints a consistent picture of the current state of MLLMs in pragmatic reasoning:

1. **No clear benefit from human-like brevity.**  
   In explicit evaluation, models as Listeners did not perform better with concise, discriminative human utterances than with verbose, underspecified model-generated ones. This indicates that they do not exploit Gricean maxims such as Quantity or Relevance when interpreting references.

2. **Probing suggests shallow discriminativity encoding.**  
   Linear probes were able to separate discriminative from underdiscriminative descriptions, but this effect was mostly explained by surface-level linguistic cues (e.g. presence of spatial markers, definite articles, cardinal numbers). There was no evidence of a robust multimodal representation of pragmatic competence.

3. **Visual information is secondary, not integrative.**  
   The image helped the probe only in cases where the textual description was ambiguous or misleading, acting as a corrective signal. However, overall trends showed that textual embeddings alone already provided most of the separability, suggesting that models do not truly integrate vision and language for pragmatic reasoning.

4. **Models are robust but pragmatically naïve.**  
   While state-of-the-art MLLMs achieve high accuracy in reference tasks and show resilience to redundancy or verbosity, they remain indifferent to pragmatic efficiency. Unlike humans, they do not adapt their behavior depending on task difficulty or communicative context.

5. **Implications for future research.**  
   These findings highlight a gap between human and model pragmatic competence: MLLMs can succeed in grounding tasks, but their success is not driven by human-like reasoning about informativeness. Bridging this gap will likely require training regimes or architectures that explicitly incorporate communicative goals, common ground, and interlocutor modeling, rather than relying only on distributional learning.

In summary, current MLLMs can be described as *effective but pragmatically naïve*: they perform well in reference comprehension, yet they do not leverage pragmatic principles in either behavior or representation. This limits their ability to participate in truly cooperative communication, a crucial step toward human-like dialogue systems.




