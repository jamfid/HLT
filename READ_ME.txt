SemEval-2024 Task 8: Black-Box Detection of Machine-Generated Text via Prompt-Tuning
Author: Giacomo Fidone, g.fidone1@studenti.unipi.it
------------------------------------------------------------------------------------

This folder contains:

- 6 python notebooks:

    - 'data.ipynb', for data understanding and data splitting;  
    - 'zero_shot_classification.ipynb', for the zero-shot classifier;
    - 'lora_A.ipynb', for LoRA-RoBERTa (subtask A);
    - 'lora_B.ipynb', for LoRA-RoBERTa (subtask B);
    - 'prompt_tuning_A.ipynb', for PT-Mistral-7B (subtask A);
    - 'prompt_tuning_B.ipynb', for PT-Mistral-7B (subtask B).

- 1 py script: 'utils.py', for utilities employed across multiple notebooks;
- 1 pdf file: 'report.pdf', which contains a report of the project;
- 2 txt file: 
  
    - 'requirements.txt', containing requirements;
    - the current 'READ_ME.txt' file.

Data is not included due to size constraints, but is available for download:

- For subtask A, 'subtaskA_train_monolingual.jsonl' and 'subtaskA_dev_monolingual.jsonl' in https://drive.google.com/drive/folders/1CAbb3DjrOPBNm0ozVBfhvrEh9P9rAppc
- For subtask B, 'subtaskB_train.jsonl' and 'subtaskB_dev.jsonl' in https://drive.google.com/drive/folders/11YeloR2eTXcTzdwI04Z-M2QVvIeQAU6-

NB: before running any other notebook, data must be processed via 'data.ipynb'. For reproducibility, see section 3.2 in 'report.pdf'.


