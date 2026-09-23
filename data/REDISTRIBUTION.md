# Benchmark redistribution review

Reviewed 2026-09-15. This inventory records upstream statements, not a legal
clearance. The repository's MIT license covers its software, not third-party
benchmark text. Input checksums establish identity, not redistribution rights.

| Input family | Upstream statement | Release consideration |
|---|---|---|
| GPQA Diamond subjects | [GPQA](https://huggingface.co/datasets/Idavidrein/gpqa) lists CC BY 4.0 but also requires agreement not to expose examples online as plaintext or images. | Do not publish plaintext inputs without resolving this condition. An authorized local preparation path or additional permission is needed. |
| HMMT | [MathArena](https://huggingface.co/datasets/MathArena/hmmt_feb_2025) specifies CC BY-NC-SA 4.0. | Preserve attribution and applicable noncommercial/share-alike terms; do not apply the software license to these questions. |
| AIME 2025/2026 | [AIME25](https://huggingface.co/datasets/math-ai/aime25) and [AIME26](https://huggingface.co/datasets/math-ai/aime26) carry Apache-2.0 metadata. | Check notices and scope of rights to underlying contest questions before redistribution. |
| AIME 2024 | [AIME24](https://huggingface.co/datasets/HuggingFaceH4/aime_2024) points to [AI-MO](https://huggingface.co/datasets/AI-MO/aimo-validation-aime), which identifies Apache-2.0. | Confirm the license scope for original competition questions and retain applicable notices. |
| MATH-100 | [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) points to [PRM800K](https://github.com/openai/prm800k), with MIT terms; the original MATH code repository also has MIT terms. | Confirm coverage of the dataset text rather than assuming a code license resolves original problem rights. |
| LiveCodeBench subsets | [Code generation lite](https://huggingface.co/datasets/livecodebench/code_generation_lite) has an unspecified `cc` license tag. | Identify the applicable CC variant and terms for third-party contest statements. |
| SuperGPQA Science | [SuperGPQA](https://huggingface.co/datasets/m-a-p/SuperGPQA) specifies ODC-BY for the database and disclaims ownership of underlying third-party content. | Database attribution alone does not settle permissions for every selected question. |
| MMLU Science | [MMLU](https://huggingface.co/datasets/cais/mmlu) identifies MIT terms. | Preserve upstream copyright/license notices and verify scope for included text. |
| HLE text-only | [HLE](https://huggingface.co/datasets/cais/hle) identifies MIT terms, requires access approval, and requests no public sharing, re-uploading or distribution. | Clarify the relationship between access conditions, this request, and MIT before redistributing text; obtain data through personal authorized access. |

Benchmark JSONL files are generated locally rather than included in the source
distribution or wheel. The preparation script does not accept agreements on a
user's behalf and does not remove restrictions on local use or redistribution.
Statistical profiles and measured result tables do not require bundling
the question text to run replay or aggregate Tables 1--4; real-generation runs
do require the corresponding prompts.
