---
title: 'GEN5 - A Binary Container Format For Reproducible AI Image Generation Artifacts'
tags:
  - Python
  - Machine Learning
  - File format
  - Reproducibility
authors:
  - name: Anuroop V J
    orcid: 0009-0002-3982-9724
    equal-contrib: true
date: 1 January 2026
bibliography: paper.bib
---

# Summary

'gen5' is an open-source Python package for storing AI-generated images together with their generation context to improve reproducibility. It provides a simple and structured way to persist images and theor associated context using a binary container representation.
# Statement of need

Reproducibility in  AI-generated images is an active area of research today, yet apart from a handful of ad-hoc metadata embedding mechanisms, there is a gap for a standard and coupled way of storing and utilising generated context concerning AI generated images. Although EXIF, XMP, and similar formats allow for the embedding of custom metadata, they fail to consider the context relating to AI-generated images as first-class, validated artifacts. Sidecar files, although suitable for large-scale datasets, introduces unneeded complexity and fragility for small to medium-scale analysis and experimentation as it requires maintaining separate files and synchronization systems. As a result, critical information such as generation parameters, environment details and intermediate representations are often fragmented, loosely coupled, or lost altogether. In addition to this, existing standards are not reliable for storing anything more than limited metadata like prompts, model name, etc. This points to a need for a structured and integrated way of dealing with such context.

The 'gen5' library addresses this gap by providing a simple and structured way of storing and interacting with such context directly alongside the binary of the generated image in a binary container format. This offers a simple solution for experimenters to compare and study generated images.

Concurrent work by [@Gao2024TowardsDA] proposes AIGIF (AI-Generated Image Format), a lightweight container that serializes only the generation syntax, including prompt, seed, model identifier, and inference parameters, and reconstructs the image via re-execution of the generative pipeline. AIGIF achieves extreme compression by forgoing storage of pixels or intermediate representations. In contrast, gen5 adopts a fidelity-first design: it embeds the final image, the initial latent tensor, a structured metadata record, and a cryptographic fingerprint of the runtime environment (PyTorch version, CUDA, OS, and GPU driver). This enables bitwise reproducibility without requiring model availability or re-generation, at the cost of a larger file size. While AIGIF excels in low-bandwidth sharing scenarios where reconstruction is feasible, gen5 targets scientific reproducibility, downstream editing (latent-space manipulation), and forensic traceability. In addition, 'gen5' has explicit storage of latent states and environment hashes providing stronger guarantees for auditability and analysis. At the time of writting, there is no publically available, easy-to-use implementation for producing or consuming AIGIF images.

Separately, IPTC has recently proposed separate fields for storing AI related metadata (i.e, AI System Used, AI System Version Used, AI Prompt Information, and AI Prompt Writer Name) [@iptc2025]. While this improves transparency for editorial and provenance workflows, these fields are not designed to support ML reproducibility. They do not capture critical technical details such as model checkpoint hashes, sampling configurations, seeds/initial noise, or hardware/runtime information, which are required in typical ML pipelines.

Work by [@Guo_2024_CVPR] demonstrates that optimizing the initial noise before sampling leads to images that are more semantically aligned with the input prompt. Importantly, once an optimized noise tensor is obtained, it can be reused to regenerate the same aligned image under deterministic sampling conditions. Thus, storing the optimized initial noise offers a practical mechanism for improving reproducibility and consistency in text-to-image generation, while still benefiting from the alignment gains introduced by INITNO.

A review of current AI-native formats, including AIGIF [@Gao2024TowardsDA], IPTC AI metadata [@iptc2025], and diffusion model checkpoints, reveals no comprehensive solution that embeds latent states directly alongside the image in a unified container. Gen5 fills this niche by ensuring that all components necessary for analysis, comparison, and reuse remain inseparable during sharing, archiving, or processing.

# References
