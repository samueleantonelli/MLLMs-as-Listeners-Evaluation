# Referring-Expression-Comprehension
This work focuses on the Referring Expression Comprehension (REC) of models with Models Vs Humans Speakers

---

# Speaker Comparison with Qwen Listener on RefOI Dataset

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

