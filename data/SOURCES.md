# Benchmark sources and access

Benchmark tasks and prompts are prepared locally with `make benchmarks`.
The pinned source revisions and file lists are recorded in
[`benchmark_sources.json`](benchmark_sources.json); preparation instructions
are in [`README.md`](README.md).

Third-party benchmark text remains subject to its upstream licenses, notices,
and access conditions. The repository's MIT license applies to its software
and does not relicense benchmark text. The source pages below provide the
applicable terms and access instructions.

| Benchmark family | Source and usage notes |
|---|---|
| GPQA Diamond subjects | [GPQA](https://huggingface.co/datasets/Idavidrein/gpqa) lists CC BY 4.0 and requires an access agreement that restricts exposing examples online as plaintext or images. Obtain access through your own account. |
| HMMT | [MathArena HMMT](https://huggingface.co/datasets/MathArena/hmmt_feb_2025) specifies CC BY-NC-SA 4.0, including attribution, noncommercial, and share-alike terms. |
| AIME 2025/2026 | [AIME25](https://huggingface.co/datasets/math-ai/aime25) and [AIME26](https://huggingface.co/datasets/math-ai/aime26) list Apache-2.0 metadata. These sources contain third-party competition questions. |
| AIME 2024 | [AIME24](https://huggingface.co/datasets/HuggingFaceH4/aime_2024) points to [AI-MO](https://huggingface.co/datasets/AI-MO/aimo-validation-aime), which lists Apache-2.0. These sources contain third-party competition questions. |
| MATH-100 | [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) points to [PRM800K](https://github.com/openai/prm800k), which provides MIT terms. The source includes third-party mathematical problems. |
| LiveCodeBench subsets | [Code generation lite](https://huggingface.co/datasets/livecodebench/code_generation_lite) lists an unspecified `cc` license tag and contains third-party contest statements. Refer to the upstream notices for their terms. |
| SuperGPQA Science | [SuperGPQA](https://huggingface.co/datasets/m-a-p/SuperGPQA) specifies ODC-BY for the database and disclaims ownership of underlying third-party content. |
| MMLU Science | [MMLU](https://huggingface.co/datasets/cais/mmlu) identifies MIT terms; see the source for copyright and license notices. |
| HLE text-only | [HLE](https://huggingface.co/datasets/cais/hle) identifies MIT terms, requires access approval, and requests no public sharing, re-uploading, or distribution. Obtain access through your own account. |

Benchmark JSONL files are generated locally and are not bundled in the source
distribution or wheel. The preparation script uses your existing access and
does not accept upstream agreements on your behalf. Dataset metadata alone
does not establish redistribution rights for underlying third-party content.

The included statistical profiles and measured result tables support replay
and aggregation of Tables 1--4 without question text. Real-generation runs
also require the corresponding locally prepared prompts.
