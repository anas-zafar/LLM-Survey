# Large Language Models: A Comprehensive Survey of its Applications, Challenges, Limitations, And Future Prospects (Updated 2025)

The Large Language Models Survey repository is a comprehensive compendium dedicated to the exploration and understanding of Large Language Models (LLMs). It houses an assortment of resources including research papers, blog posts, tutorials, code examples, and more to provide an in-depth look at the progression, methodologies, and applications of LLMs. This repo is an invaluable resource for AI researchers, data scientists, or enthusiasts interested in the advancements and inner workings of LLMs. We encourage contributions from the wider community to promote collaborative learning and continue pushing the boundaries of LLM research.

## Timeline of LLMs
![evolutionv1 1](https://github.com/anas-zafar/LLM-Survey/blob/main/Images/evolutionv1.2.PNG)

## List of LLMs (Updated July 2025)

| Language Model | Organization | Release Date | Checkpoints | Paper/Blog | Params (B) | Context Length | Licence | Try it |
|---|---|---|---|---|---|---|---|---|
| **2025 Latest Models** |
| Grok 3 / Grok 3 Mini | xAI | 2025/02 | [Grok 3](https://x.ai/), [Grok 3 Mini](https://x.ai/) | [Grok 3 Beta — The Age of Reasoning Agents](https://x.ai/news/grok-3) | 314 active (1M+ total) / Smaller variant | 1M tokens | Proprietary | [xAI Platform](https://grok.com) |
| Llama 4 Scout | Meta | 2025/04 | [Llama 4 Scout](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E) | [The Llama 4 herd: The beginning of a new era](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) | 17B active (109B total) | 10M tokens | [Llama 4 Community License](https://github.com/meta-llama/llama-models/blob/main/models/llama4/LICENSE) | [HuggingFace](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E) |
| Llama 4 Maverick | Meta | 2025/04 | [Llama 4 Maverick](https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E) | [The Llama 4 herd](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) | 17B active (400B total) | 1M tokens | [Llama 4 Community License](https://github.com/meta-llama/llama-models/blob/main/models/llama4/LICENSE) | [HuggingFace](https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E) |
| Llama 4 Behemoth | Meta | 2025/04 (Training) | [In Training](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) | [The Llama 4 herd](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) | 288B active (~2T total) | TBD | TBD | TBD |
| Qwen 3 Family | Alibaba | 2025/04 | [Qwen 3 Family](https://huggingface.co/collections/Qwen/qwen3-6633ebacb5c0a53ce76e1089) | [Alibaba unveils Qwen3](https://techcrunch.com/2025/04/28/alibaba-unveils-qwen-3-a-family-of-hybrid-ai-reasoning-models/) | 0.6B - 235B (22B active) | 32K - 131K tokens | Apache 2.0 | [Qwen Chat](https://chat.qwen.ai/) |
| DeepSeek-R1 Family | DeepSeek | 2025/01-05 | [DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1), [R1-Zero](https://huggingface.co/deepseek-ai/DeepSeek-R1-Zero), [R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528) | [DeepSeek-R1: Incentivizing Reasoning Capability](https://arxiv.org/abs/2501.12948) | 37B active (671B total) | 128K tokens | MIT | [DeepSeek Platform](https://chat.deepseek.com/) |
| o3 / o3-mini / o4-mini | OpenAI | 2025/01-04 | [o3](https://platform.openai.com/docs/models/o3), [o3-mini](https://platform.openai.com/docs/models/o3-mini), [o4-mini](https://platform.openai.com/docs/models/o4-mini) | [Introducing OpenAI o3 and o4-mini](https://openai.com/index/introducing-o3-and-o4-mini/) | Undisclosed | 200K tokens | Proprietary | [ChatGPT](https://chat.openai.com/) |
| Claude 4 (Sonnet & Opus) | Anthropic | 2025/05 | [Claude Sonnet 4](https://docs.anthropic.com/en/docs/about-claude/models/overview), [Claude Opus 4](https://docs.anthropic.com/en/docs/about-claude/models/overview) | [Introducing Claude 4](https://www.anthropic.com/news/claude-4) | Undisclosed | 200K tokens | Proprietary | [Claude.ai](https://claude.ai/) |
| Gemini 2.5 Family | Google | 2025/03-06 | [Gemini 2.5 Pro](https://ai.google.dev/gemini-api/docs/models), [2.5 Flash](https://ai.google.dev/gemini-api/docs/models), [2.5 Flash-Lite](https://ai.google.dev/gemini-api/docs/models) | [Gemini 2.5: Our newest Gemini model with thinking](https://blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/) | Undisclosed | 1M tokens | Proprietary | [Gemini](https://gemini.google.com/) |
| **Major 2024 Models** |
| GPT-4o / GPT-4o mini | OpenAI | 2024/05-07 | [GPT-4o](https://platform.openai.com/docs/models/gpt-4o), [GPT-4o mini](https://platform.openai.com/docs/models/gpt-4o-mini) | [Hello GPT-4o](https://openai.com/index/hello-gpt-4o/), [GPT-4o mini: advancing cost-efficient intelligence](https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/) | Undisclosed | 128K tokens | Proprietary | [ChatGPT](https://chat.openai.com/) |
| o1 / o1-mini | OpenAI | 2024/09 | [o1](https://platform.openai.com/docs/models/o1), [o1-mini](https://platform.openai.com/docs/models/o1) | [Learning to Reason with LLMs](https://openai.com/index/learning-to-reason-with-llms/) | Undisclosed | 200K / 128K tokens | Proprietary | [ChatGPT](https://chat.openai.com/) |
| Claude 3 Family | Anthropic | 2024/03 | [Claude 3 Haiku](https://docs.anthropic.com/en/docs/about-claude/models), [Claude 3 Sonnet](https://docs.anthropic.com/en/docs/about-claude/models), [Claude 3 Opus](https://docs.anthropic.com/en/docs/about-claude/models) | [Introducing the next generation of Claude](https://www.anthropic.com/news/claude-3-family) | Undisclosed | 200K tokens | Proprietary | [Claude.ai](https://claude.ai/) |
| Claude 3.5 Sonnet | Anthropic | 2024/06 | [Claude 3.5 Sonnet](https://docs.anthropic.com/en/docs/about-claude/models) | [Claude 3.5 Sonnet](https://www.anthropic.com/news/claude-3-5-sonnet) | Undisclosed | 200K tokens | Proprietary | [Claude.ai](https://claude.ai/) |
| Claude 3.7 Sonnet | Anthropic | 2024/10 | [Claude 3.7 Sonnet](https://docs.anthropic.com/en/docs/about-claude/models) | [Claude 3.7 Sonnet](https://www.anthropic.com/news/claude-3-7-sonnet) | Undisclosed | 200K tokens | Proprietary | [Claude.ai](https://claude.ai/) |
| Gemini 1.5 Pro / Flash | Google | 2024/02-05 | [Gemini 1.5 Pro](https://ai.google.dev/models/gemini), [Gemini 1.5 Flash](https://ai.google.dev/models/gemini) | [Our next-generation model: Gemini 1.5](https://blog.google/technology/ai/google-gemini-next-generation-model-february-2024/) | Undisclosed | 1M-2M / 1M tokens | Proprietary | [Gemini](https://gemini.google.com/) |
| Gemini 2.0 Flash | Google | 2024/12 | [Gemini 2.0 Flash](https://ai.google.dev/gemini-api/docs/models) | [Gemini 2.0 Flash](https://blog.google/technology/google-deepmind/gemini-2-0-flash-multimodal/) | Undisclosed | 1M tokens | Proprietary | [Gemini](https://gemini.google.com/) |
| Gemma 2 | Google | 2024/06 | [Gemma 2 Family](https://huggingface.co/collections/google/gemma-2-release-667d6600fd5220e7b967f315) | [Gemma 2: Improving Open Language Models at a Practical Size](https://blog.google/technology/developers/google-gemma-2/) | 9B, 27B | 8K tokens | Apache 2.0 | [HuggingFace](https://huggingface.co/google/gemma-2-9b) |
| Llama 3 Family | Meta | 2024/04 | [Llama 3 Weights](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) | [Introducing Meta Llama 3](https://ai.meta.com/blog/meta-llama-3/) | 8B, 70B | 8K tokens | [Custom](https://github.com/facebookresearch/llama/blob/main/LICENSE) | [HuggingChat](https://huggingface.co/chat/) |
| Llama 3.1 | Meta | 2024/07 | [Llama 3.1 Weights](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) | [The Llama 3 Herd of Models](https://ai.meta.com/research/publications/the-llama-3-herd-of-models/) | 8B, 70B, 405B | 128K tokens | [Custom](https://github.com/facebookresearch/llama/blob/main/LICENSE) | [HuggingChat](https://huggingface.co/chat/) |
| Llama 3.2 | Meta | 2024/09 | [Llama 3.2 Models](https://huggingface.co/collections/meta-llama/llama-32-66f448ffc8c32f949b04c8cf) | [Llama 3.2: Revolutionizing edge AI and vision with open, customizable models](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/) | 1B, 3B, 11B, 90B | 128K tokens | [Custom](https://github.com/facebookresearch/llama/blob/main/LICENSE) | [HuggingChat](https://huggingface.co/chat/) |
| Llama 3.3 | Meta | 2024/12 | [Llama 3.3 70B](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) | [Llama 3.3 70B](https://ai.meta.com/blog/llama-3-3-70b/) | 70B | 128K tokens | [Custom](https://github.com/facebookresearch/llama/blob/main/LICENSE) | [HuggingChat](https://huggingface.co/chat/) |
| Phi-3 Family | Microsoft | 2024/04-08 | [Phi-3 Mini](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct), [Phi-3 Small](https://huggingface.co/microsoft/Phi-3-small-128k-instruct), [Phi-3 Medium](https://huggingface.co/microsoft/Phi-3-medium-128k-instruct), [Phi-3.5 Mini](https://huggingface.co/microsoft/Phi-3.5-mini-instruct) | [Phi-3 Technical Report](https://arxiv.org/abs/2404.14219) | 3.8B - 14B | 4K-128K tokens | MIT | [Azure AI Studio](https://azure.microsoft.com/en-us/products/ai-studio/) |
| IBM Granite 3.0 / 3.1 | IBM | 2024/10-12 | [Granite 3.0](https://huggingface.co/collections/ibm-granite/granite-3-0-language-models-6752fa54b3b2f429c4c3be6c), [Granite 3.1](https://huggingface.co/collections/ibm-granite/granite-3-1-language-models-67699e6bb51fb7e8b35fd7e7) | [IBM Introduces Granite 3.0](https://newsroom.ibm.com/2024-10-21-ibm-introduces-granite-3-0-high-performing-ai-models-built-for-business) | 2B, 8B | 4K / 128K tokens | Apache 2.0 | [IBM watsonx](https://www.ibm.com/watsonx) |
| Command R / R+ | Cohere | 2024/03-04 | [Command R](https://huggingface.co/CohereForAI/c4ai-command-r-v01), [Command R+](https://huggingface.co/CohereForAI/c4ai-command-r-plus) | [Command R: Cohere's scalable generative model](https://cohere.com/blog/command-r) | 35B / 104B | 128K tokens | CC BY-NC 4.0 | [Cohere Platform](https://cohere.com/) |
| DeepSeek-V3 Family | DeepSeek | 2024/12-2025/03 | [DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3), [DeepSeek-V3-0324](https://huggingface.co/deepseek-ai/DeepSeek-V3-0324) | [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) | 37B active (671B total) | 128K tokens | MIT | [DeepSeek Platform](https://chat.deepseek.com/) |
| Qwen 2.5 Family | Alibaba | 2024/09-2025/01 | [Qwen 2.5 Family](https://huggingface.co/collections/Qwen/qwen25-66e81a666513e518adb90d9e), [Qwen 2.5-Max](https://huggingface.co/Qwen/Qwen2.5-Max) | [Qwen2.5: A Party of Foundation Models](https://qwenlm.github.io/blog/qwen2.5/) | 0.5B - 72B / Undisclosed | 32K-128K tokens | Apache 2.0 / Proprietary | [Qwen Chat](https://chat.qwen.ai/) |
| QwQ-32B | Alibaba | 2024/11 | [QwQ-32B-Preview](https://huggingface.co/Qwen/QwQ-32B-Preview) | [QwQ-32B Technical Report](https://arxiv.org/abs/2411.20213) | 32B | 32K tokens | Apache 2.0 | [Qwen Chat](https://chat.qwen.ai/) |
| Mistral Family | Mistral AI | 2023/09-2025/05 | [Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-v0.1), [Mistral Large 2](https://huggingface.co/mistralai/Mistral-Large-Instruct-2407), [Mistral Medium](https://mistral.ai/) | [Mistral 7B](https://mistral.ai/news/announcing-mistral-7b/) | 7B - 123B / Undisclosed | 4K-128K tokens | Apache 2.0 / Proprietary | [Mistral Platform](https://chat.mistral.ai/) |0](https://huggingface.co/collections/ibm-granite/granite-3-0-language-models-6752fa54b3b2f429c4c3be6c), [Granite 3.1](https://huggingface.co/collections/ibm-granite/granite-3-1-language-models-67699e6bb51fb7e8b35fd7e7) | [IBM Introduces Granite 3.0](https://newsroom.ibm.com/2024-10-21-ibm-introduces-granite-3-0-high-performing-ai-models-built-for-business) | 2B, 8B | 4K / 128K tokens | Apache 2.0 | [IBM watsonx](https://www.ibm.com/watsonx) |
| Command R / R+ | 2024/03-04 | [Command R](https://huggingface.co/CohereForAI/c4ai-command-r-v01), [Command R+](https://huggingface.co/CohereForAI/c4ai-command-r-plus) | [Command R: Cohere's scalable generative model](https://cohere.com/blog/command-r) | 35B / 104B | 128K tokens | CC BY-NC 4.0 | [Cohere Platform](https://cohere.com/) |
| DeepSeek-V3 Family | 2024/12-2025/03 | [DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3), [DeepSeek-V3-0324](https://huggingface.co/deepseek-ai/DeepSeek-V3-0324) | [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) | 37B active (671B total) | 128K tokens | MIT | [DeepSeek Platform](https://chat.deepseek.com/) |
| Qwen 2.5 Family | 2024/09-2025/01 | [Qwen 2.5 Family](https://huggingface.co/collections/Qwen/qwen25-66e81a666513e518adb90d9e), [Qwen 2.5-Max](https://huggingface.co/Qwen/Qwen2.5-Max) | [Qwen2.5: A Party of Foundation Models](https://qwenlm.github.io/blog/qwen2.5/) | 0.5B - 72B / Undisclosed | 32K-128K tokens | Apache 2.0 / Proprietary | [Qwen Chat](https://chat.qwen.ai/) |
| QwQ-32B | 2024/11 | [QwQ-32B-Preview](https://huggingface.co/Qwen/QwQ-32B-Preview) | [QwQ-32B Technical Report](https://arxiv.org/abs/2411.20213) | 32B | 32K tokens | Apache 2.0 | [Qwen Chat](https://chat.qwen.ai/) |
| Mistral Family | 2023/09-2025/05 | [Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-v0.1), [Mistral Large 2](https://huggingface.co/mistralai/Mistral-Large-Instruct-2407), [Mistral Medium](https://mistral.ai/) | [Mistral 7B](https://mistral.ai/news/announcing-mistral-7b/) | 7B - 123B / Undisclosed | 4K-128K tokens | Apache 2.0 / Proprietary | [Mistral Platform](https://chat.mistral.ai/) |
| **Previous Generation Models** |
| GPT-4 / GPT-4.5 | 2023/03-2024/06 | [API Access](https://platform.openai.com/docs/models/gpt-4) | [GPT-4 Technical Report](https://arxiv.org/abs/2303.08774) | Undisclosed | 8K-128K tokens | Proprietary | [ChatGPT](https://chat.openai.com/) |
| LLaMA 2 | 2023/06 | [LLaMA 2 Weights](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) | [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://scontent-ham3-1.xx.fbcdn.net/v/t39.2365-6/10000000_662098952474184_2584067087619170692_n.pdf) | 7B - 70B | 4K tokens | [Custom](https://github.com/facebookresearch/llama/blob/main/LICENSE) | [HuggingChat](https://huggingface.co/blog/llama2#demo) |
| PaLM 2 | 2023/05 | [PaLM 2](https://ai.google/discover/palm2/) | [PaLM 2 Technical Report](https://ai.google/static/documents/palm2techreport.pdf) | Undisclosed | 8K tokens | Proprietary | [Bard](https://bard.google.com/) |
| Bard | 2023/03 | [Bard](https://bard.google.com/) | [Bard: An experiment by Google](https://blog.google/technology/ai/bard-google-ai-search-updates/) | Undisclosed | 8K tokens | Proprietary | [Bard](https://bard.google.com/) |
| Chinchilla | 2022/03 | [Chinchilla](https://www.deepmind.com/publications/training-compute-optimal-large-language-models) | [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) | 70B | 2K tokens | Proprietary | [Research Only] |
| Sparrow | 2022/09 | [Sparrow](http://arxiv.org/abs/2209.14375v1) | [Improving alignment of dialogue agents via targeted human judgements](http://arxiv.org/abs/2209.14375v1) | 70B | 4K tokens | Proprietary | [Research Only] |
| Gopher | 2021/12 | [Gopher](https://www.deepmind.com/blog/language-modelling-at-scale-gopher-ethical-considerations-and-retrieval) | [Scaling Language Models: Methods, Analysis & Insights from Training Gopher](https://arxiv.org/abs/2112.11446) | 280B | 2K tokens | Proprietary | [Research Only] |
| YaLM | 2022/06 | [YaLM 100B](https://github.com/yandex/YaLM-100B) | [YaLM 100B](https://github.com/yandex/YaLM-100B) | 100B | 2K tokens | Apache 2.0 | [GitHub](https://github.com/yandex/YaLM-100B) |
| OPT | 2022/05 | [OPT Family](https://github.com/facebookresearch/metaseq/tree/main/projects/OPT) | [OPT: Open Pre-trained Transformer Language Models](https://arxiv.org/abs/2205.01068) | 0.125B - 175B | 2K tokens | MIT | [HuggingFace](https://huggingface.co/facebook/opt-66b) |
| BLOOM | 2022/11 | [BLOOM](https://huggingface.co/bigscience/bloom) | [BLOOM: A 176B-Parameter Open-Access Multilingual Language Model](https://arxiv.org/abs/2211.05100) | 176B | 2K tokens | [OpenRAIL-M v1](https://huggingface.co/spaces/bigcode/bigcode-model-license-agreement) | [HuggingFace](https://huggingface.co/bigscience/bloom) |
| Jurassic-1 / Jurassic-2 | 2021/08 / 2023/03 | [AI21 Studio](https://www.ai21.com/studio) | [Jurassic-1: Technical Details And Evaluation](https://uploads-ssl.webflow.com/60fd4503684b466578c0d307/61138924626a6981ee09caf6_jurassic_tech_paper.pdf) | 178B | 2K / 8K tokens | Proprietary | [AI21 Studio](https://www.ai21.com/studio) |
| Anthropic LM (v4-s3) | 2022/12 | [Anthropic LM](https://www.anthropic.com/) | [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) | 52B | 4K tokens | Proprietary | [Research Only] |
| GLaM | 2021/12 | [GLaM](https://ai.googleblog.com/2021/12/more-efficient-in-context-learning-with.html) | [GLaM: Efficient Scaling of Language Models with Mixture-of-Experts](https://arxiv.org/abs/2112.06905) | 1.2T (64B active) | 2K tokens | Proprietary | [Research Only] |
| GPT-J / GPT-NeoX | 2021/06 / 2022/04 | [GPT-J-6B](https://github.com/kingoflolz/mesh-transformer-jax/#gpt-j-6b), [GPT-NeoX-20B](https://huggingface.co/EleutherAI/gpt-neox-20b) | [GPT-J-6B: 6B JAX-Based Transformer](https://arankomatsuzaki.wordpress.com/2021/06/04/gpt-j/) | 6B / 20B | 2K tokens | Apache 2.0 | [HuggingFace](https://huggingface.co/EleutherAI/gpt-j-6b) |
| Minerva | 2022/06 | [Minerva](https://arxiv.org/abs/2206.14858) | [Solving Quantitative Reasoning Problems with Language Models](https://arxiv.org/abs/2206.14858) | 540B | 2K tokens | Proprietary | [Research Only] |
| Gallactica | 2022/11 | [Gallactica](https://github.com/paperswithcode/gallactica) | [Gallactica: A Large Language Model for Science](https://arxiv.org/abs/2211.09085) | 120B | 2K tokens | Apache 2.0 | [Removed] |
| Vicuna | 2023/03 | [Vicuna](https://github.com/lm-sys/FastChat) | [Vicuna: An Open-Source Chatbot Impressing GPT-4](https://lmsys.org/blog/2023-03-30-vicuna/) | 7B, 13B, 33B | 2K tokens | Custom | [FastChat](https://github.com/lm-sys/FastChat) |
| Alpaca | 2023/03 | [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) | [Stanford Alpaca: An Instruction-following LLaMA Model](https://github.com/tatsu-lab/stanford_alpaca) | 7B | 2K tokens | Custom | [GitHub](https://github.com/tatsu-lab/stanford_alpaca) |
| **Coding-Specialized Models** |
| Code Llama | 2023/08 | [Code Llama Models](https://github.com/facebookresearch/codellama) | [Code Llama: Open Foundation Models for Code](https://ai.meta.com/research/publications/code-llama-open-foundation-models-for-code/) | 7B - 34B | 4K tokens | [Custom](https://github.com/facebookresearch/llama/blob/main/LICENSE) | [HuggingChat](https://huggingface.co/blog/codellama) |
| StarCoder / StarChat | 2023/05 | [StarCoder](https://huggingface.co/bigcode/starcoder), [StarChat](https://huggingface.co/HuggingFaceH4/starchat-alpha) | [StarCoder: A State-of-the-Art LLM for Code](https://huggingface.co/blog/starcoder) | 1.1B - 16B | 8K tokens | [OpenRAIL-M v1](https://huggingface.co/spaces/bigcode/bigcode-model-license-agreement) | [HuggingFace](https://huggingface.co/bigcode/starcoder) |
| CodeGen2 / CodeGen2.5 | 2023/04-07 | [CodeGen2](https://github.com/salesforce/CodeGen2), [CodeGen2.5](https://huggingface.co/Salesforce/codegen25-7b-multi) | [CodeGen2: Lessons for Training LLMs on Programming and Natural Languages](https://arxiv.org/abs/2305.02309) | 1B - 16B | 2K tokens | Apache 2.0 | [HuggingFace](https://huggingface.co/Salesforce/codegen25-7b-multi) |
| CodeT5+ | 2023/05 | [CodeT5+](https://github.com/salesforce/CodeT5/tree/main/CodeT5+) | [CodeT5+: Open Code Large Language Models for Code Understanding and Generation](https://arxiv.org/abs/2305.07922) | 0.22B - 16B | 512 tokens | BSD-3-Clause | [GitHub](https://github.com/salesforce/CodeT5/tree/main/CodeT5+) |
| Replit Code | 2023/05 | [replit-code-v1-3b](https://huggingface.co/replit/replit-code-v1-3b) | [Training a SOTA Code LLM in 1 week](https://www.latent.space/p/reza-shabani) | 2.7B | Infinity (ALiBi) | CC BY-SA-4.0 | [HuggingFace](https://huggingface.co/replit/replit-code-v1-3b) |
| SantaCoder | 2023/01 | [SantaCoder](https://huggingface.co/bigcode/santacoder) | [SantaCoder: don't reach for the stars!](https://arxiv.org/abs/2301.03988) | 1.1B | 2K tokens | [OpenRAIL-M v1](https://huggingface.co/spaces/bigcode/bigcode-model-license-agreement) | [HuggingFace](https://huggingface.co/bigcode/santacoder) |
| DeciCoder | 2023/08 | [DeciCoder-1B](https://huggingface.co/Deci/DeciCoder-1b) | [Introducing DeciCoder: The New Gold Standard in Efficient and Accurate Code Generation](https://deci.ai/blog/decicoder-efficient-and-accurate-code-generation-llm/) | 1.1B | 2K tokens | Apache 2.0 | [HuggingFace](https://huggingface.co/spaces/Deci/DeciCoder-Demo) |
| **Additional Historical Models** |
| T5 / Flan-T5 | 2019/10 | [T5 & Flan-T5](https://github.com/google-research/t5x/blob/main/docs/models.md#flan-t5-checkpoints) | [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://github.com/google-research/text-to-text-transfer-transformer) | 0.06B - 11B | 512 tokens | Apache 2.0 | [HuggingFace](https://huggingface.co/google/flan-t5-xxl) |
| UL2 / Flan-UL2 | 2022/10 | [UL2 & Flan-UL2](https://github.com/google-research/google-research/tree/master/ul2#checkpoints) | [UL2 20B: An Open Source Unified Language Learner](https://ai.googleblog.com/2022/10/ul2-20b-open-source-unified-language.html) | 20B | 512-2K tokens | Apache 2.0 | [HuggingFace](https://huggingface.co/google/flan-ul2) |
| InstructGPT | 2022/03 | [API Access](https://openai.com/blog/instruction-following/) | [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) | 1.3B - 175B | 2K tokens | Proprietary | [OpenAI API] |
| ChatGPT | 2022/11 | [API Access](https://openai.com/blog/chatgpt/) | [ChatGPT: Optimizing Language Models for Dialogue](https://openai.com/blog/chatgpt/) | ~175B | 4K tokens | Proprietary | [ChatGPT](https://chat.openai.com/) |
| Pythia | 2023/04 | [Pythia 70M - 12B](https://github.com/EleutherAI/pythia) | [Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling](https://arxiv.org/abs/2304.01373) | 0.07B - 12B | 2K tokens | Apache 2.0 | [HuggingFace](https://huggingface.co/EleutherAI/pythia-12b) |
| Dolly | 2023/04 | [dolly-v2-12b](https://huggingface.co/databricks/dolly-v2-12b) | [Free Dolly: Introducing the World's First Truly Open Instruction-Tuned LLM](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm) | 3B, 7B, 12B | 2K tokens | MIT | [HuggingFace](https://huggingface.co/databricks/dolly-v2-12b) |
| RedPajama-INCITE | 2023/05 | [RedPajama-INCITE](https://huggingface.co/togethercomputer) | [Releasing 3B and 7B RedPajama-INCITE family of models](https://www.together.xyz/blog/redpajama-models-v1) | 3B - 7B | 2K tokens | Apache 2.0 | [HuggingFace](https://huggingface.co/togethercomputer/RedPajama-INCITE-Instruct-7B-v0.1) |
| Falcon | 2023/05 | [Falcon-180B](https://huggingface.co/tiiuae/falcon-180B), [Falcon-40B](https://huggingface.co/tiiuae/falcon-40b), [Falcon-7B](https://huggingface.co/tiiuae/falcon-7b) | [The RefinedWeb Dataset for Falcon LLM](https://arxiv.org/abs/2306.01116) | 7B, 40B, 180B | 2K tokens | Apache 2.0 | [HuggingFace](https://huggingface.co/tiiuae/falcon-7b) |
| MPT Family | 2023/05-06 | [MPT-7B](https://huggingface.co/mosaicml/mpt-7b), [MPT-30B](https://huggingface.co/mosaicml/mpt-30b) | [Introducing MPT-7B](https://www.mosaicml.com/blog/mpt-7b) | 7B, 30B | 2K-8K tokens | Apache 2.0 | [MosaicML](https://www.mosaicml.com/) |
| OpenLLaMA | 2023/05 | [OpenLLaMA Models](https://huggingface.co/openlm-research) | [OpenLLaMA: An Open Reproduction of LLaMA](https://github.com/openlm-research/open_llama) | 3B, 7B, 13B | 2K tokens | Apache 2.0 | [HuggingFace](https://huggingface.co/openlm-research/open_llama_7b) |
| h2oGPT | 2023/05 | [h2oGPT](https://github.com/h2oai/h2ogpt) | [Building the World's Best Open-Source Large Language Model](https://h2o.ai/blog/building-the-worlds-best-open-source-large-language-model-h2o-ais-journey/) | 12B - 20B | 256-2K tokens | Apache 2.0 | [h2oGPT](https://gpt.h2o.ai/) |
| FastChat-T5 | 2023/04 | [fastchat-t5-3b-v1.0](https://huggingface.co/lmsys/fastchat-t5-3b-v1.0) | [FastChat-T5: Compact and Commercial-friendly Chatbot](https://twitter.com/lmsysorg/status/1652037026705985537) | 3B | 512 tokens | Apache 2.0 | [HuggingFace](https://huggingface.co/lmsys/fastchat-t5-3b-v1.0) |
| StableLM | 2023/04 | [StableLM-Alpha](https://github.com/Stability-AI/StableLM) | [Stability AI Launches StableLM Suite](https://stability.ai/blog/stability-ai-launches-the-first-of-its-stablelm-suite-of-language-models) | 3B - 65B | 4K tokens | CC BY-SA-4.0 | [HuggingFace](https://huggingface.co/stabilityai/stablelm-base-alpha-7b) |
| Koala | 2023/04 | [Koala](https://bair.berkeley.edu/blog/2023/04/03/koala/) | [Koala: A Dialogue Model for Academic Research](https://bair.berkeley.edu/blog/2023/04/03/koala/) | 13B | 4K tokens | Custom | [BAIR](https://bair.berkeley.edu/blog/2023/04/03/koala/) |
| OpenHermes | 2023/09 | [OpenHermes-7B](https://huggingface.co/teknium/OpenHermes-7B), [OpenHermes-13B](https://huggingface.co/teknium/OpenHermes-13B) | [Nous Research OpenHermes](https://nousresearch.com/) | 7B, 13B | 4K tokens | MIT | [HuggingFace](https://huggingface.co/teknium/OpenHermes-13B) |
| SOLAR | 2023/12 | [Solar-10.7B](https://huggingface.co/upstage/SOLAR-10.7B-v1.0) | [SOLAR 10.7B: Scaling Large Language Models with Simple yet Effective Depth Up-scaling](https://arxiv.org/abs/2312.15166) | 10.7B | 4K tokens | Apache 2.0 | [HuggingFace](https://huggingface.co/upstage/SOLAR-10.7B-v1.0) |
| Phi-2 | 2023/12 | [phi-2](https://huggingface.co/microsoft/phi-2) | [Phi-2: The surprising power of small language models](https://www.microsoft.com/en-us/research/blog/phi-2-the-surprising-power-of-small-language-models/) | 2.7B | 2K tokens | MIT | [HuggingFace](https://huggingface.co/microsoft/phi-2) |
| OpenLM | 2023/09 | [OpenLM 1B](https://huggingface.co/mlfoundations/open_lm_1B), [OpenLM 7B](https://huggingface.co/mlfoundations/open_lm_7B_1.25T) | [Open LM: a minimal but performative language modeling repository](https://github.com/mlfoundations/open_lm) | 1B, 7B | 2K tokens | MIT | [HuggingFace](https://huggingface.co/mlfoundations/open_lm_7B_1.25T) |
| RWKV | 2021/08 | [RWKV Models](https://github.com/BlinkDL/RWKV-LM) | [The RWKV Language Model](https://github.com/BlinkDL/RWKV-LM) | 0.1B - 14B | Infinite (RNN) | Apache 2.0 | [HuggingFace](https://huggingface.co/BlinkDL/rwkv-4-world-7b) |
| DLite | 2023/05 | [dlite-v2-1_5b](https://huggingface.co/aisquared/dlite-v2-1_5b) | [Announcing DLite V2: Lightweight, Open LLMs](https://medium.com/ai-squared/announcing-dlite-v2-lightweight-open-llms-that-can-run-anywhere-a852e5978c6e) | 0.124B - 1.5B | 1K tokens | Apache 2.0 | [HuggingFace](https://huggingface.co/aisquared/dlite-v2-1_5b) |
| Open Assistant | 2023/03 | [OA-Pythia-12B](https://huggingface.co/OpenAssistant/pythia-12b-sft-v8-7k-steps) | [Democratizing Large Language Model Alignment](https://arxiv.org/abs/2304.07327) | 12B | 2K tokens | Apache 2.0 | [HuggingFace](https://huggingface.co/OpenAssistant/pythia-12b-sft-v8-7k-steps) |
| Cerebras-GPT | 2023/03 | [Cerebras-GPT](https://huggingface.co/cerebras) | [Cerebras-GPT: A Family of Open, Compute-efficient, Large Language Models](https://arxiv.org/abs/2304.03208) | 0.111B - 13B | 2K tokens | Apache 2.0 | [HuggingFace](https://huggingface.co/cerebras/Cerebras-GPT-13B) |
| XGen | 2023/06 | [XGen-7B-8K-Base](https://huggingface.co/Salesforce/xgen-7b-8k-base) | [Long Sequence Modeling with XGen](https://blog.salesforceairesearch.com/xgen/) | 7B | 8K tokens | Apache 2.0 | [HuggingFace](https://huggingface.co/Salesforce/xgen-7b-8k-base) |

## Key Developments in 2024

The year 2024 was transformative for the LLM landscape, with multiple breakthrough releases that established new benchmarks and capabilities:

**OpenAI's Major Releases**: **GPT-4o** launched in May 2024 brought true multimodal capabilities with 232ms response times, while **o1** and **o1-mini** in September introduced reasoning models that spend more time "thinking" through problems, achieving 83% on mathematical olympiad problems compared to GPT-4o's 13%.

**Anthropic's Claude 3 Family**: The **Claude 3** series (Haiku, Sonnet, Opus) launched in March 2024 were the first models to challenge GPT-4's dominance on leaderboards, followed by **Claude 3.5 Sonnet** in June and **Claude 3.7 Sonnet** in October, which became particularly popular for coding tasks.

**Google's Gemini Evolution**: **Gemini 1.5 Pro** debuted in February 2024 with up to 2M token context windows, followed by **Gemini 1.5 Flash** in May for faster performance, and **Gemini 2.0 Flash** in December 2024.

**Meta's Llama Progression**: **Llama 3** (8B, 70B) launched in April 2024, followed by the groundbreaking **Llama 3.1** series in July including the massive 405B parameter model - the largest open-source model at the time. **Llama 3.2** brought multimodal capabilities in September, and **Llama 3.3** concluded the year in December.

**Microsoft's Phi Revolution**: Microsoft's **Phi-3** family proved that smaller models could punch above their weight, with **Phi-3 Mini** (3.8B parameters) matching much larger models on benchmarks. The series expanded with **Phi-3 Small** (7B), **Phi-3 Medium** (14B), and **Phi-3.5 Mini** throughout 2024.

**Enterprise-Focused Models**: **IBM Granite 3.0** launched in October 2024 focused on enterprise use cases, while **Cohere's Command R** and **Command R+** models excelled in retrieval-augmented generation tasks.

**Google's Open Models**: **Gemma 2** (9B, 27B parameters) launched in June 2024 became highly popular in the open-source community, consistently ranking high in community evaluations.

## Key Developments in 2025

The year 2025 has been marked by several breakthrough releases in the LLM landscape. **Grok 3**, launched by xAI in February 2025, introduced a 1 million token context window and achieved a record-breaking Elo score of 1402 in the Chatbot Arena, making it the first AI model to surpass this milestone. The model was trained on 12.8 trillion tokens and boasts 10x the computational power of its predecessor.

**Meta's Llama 4 family** represents a major leap forward with the introduction of Mixture-of-Experts (MoE) architecture. Llama 4 Scout features an unprecedented 10 million token context window, while Llama 4 Maverick achieves an ELO score of 1417 on LMSYS Chatbot Arena, outperforming GPT-4o and Gemini 2.0 Flash.

**DeepSeek-R1** emerged as the first major open-source reasoning model, trained purely through reinforcement learning without supervised fine-tuning. The model demonstrates performance comparable to OpenAI's o1 across math, code, and reasoning tasks while being completely open-source under the MIT license.

**Qwen 3**, released by Alibaba in April 2025, features a family of "hybrid" reasoning models ranging from 0.6B to 235B parameters, supporting 119 languages and trained on over 36 trillion tokens. The models seamlessly integrate thinking and non-thinking modes, offering users flexibility to control the thinking budget.

**OpenAI** continued its reasoning model series with **o3** and **o4-mini** in April 2025, while **Anthropic** launched **Claude 4** (Opus 4 and Sonnet 4) in May 2025, setting new standards for coding and advanced reasoning with extended thinking capabilities and tool use.

**Google's Gemini 2.5 Pro** debuted as a thinking model with a 1 million token context window, leading on LMArena leaderboards and excelling in coding, math, and multimodal understanding tasks.

## Notable Trends in 2025

1. **Reasoning Models**: The emergence of models that can "think" through problems step-by-step, with extended reasoning capabilities becoming standard.

2. **Massive Context Windows**: Models now support context windows ranging from 1M to 10M tokens, enabling processing of entire codebases and documents.

3. **Mixture-of-Experts (MoE) Architecture**: More efficient model architectures that activate only a subset of parameters during inference.

4. **Open-Source Reasoning**: DeepSeek-R1's success has democratized access to reasoning capabilities previously available only in proprietary models.

5. **Multimodal Integration**: Native multimodality becoming standard, with models trained on text, images, audio, and video from the ground up.

6. **Tool Use and Agentic Capabilities**: Enhanced ability to use tools, execute code, and perform complex multi-step tasks autonomously.

## Performance Benchmarks (2025)

### Reasoning Benchmarks (AIME 2025)
- Grok 3: 93.3%
- DeepSeek-R1-0528: 87.5%
- Gemini 2.5 Pro: 86.7%
- o3-mini: 86.5%

### Coding Benchmarks (SWE-bench Verified)
- Claude Opus 4: 72.5%
- Claude Sonnet 4: 72.7%
- OpenAI Codex 1: 72.1%
- Llama 4 Maverick: ~70%

### Long Context Performance (1M+ tokens)
- Llama 4 Scout: 10M tokens
- Grok 3: 1M tokens
- Gemini 2.5 Pro: 1M tokens
- Llama 4 Maverick: 1M tokens

## Model Evolution Timeline

### 2022: Foundation Era
- **ChatGPT** revolutionized conversational AI
- **InstructGPT** introduced instruction following
- Large proprietary models dominated (GPT-3, PaLM, Chinchilla)

### 2023: Open Source Awakening
- **LLaMA** sparked the open-source revolution
- **Claude** introduced constitutional AI
- Specialized coding models emerged (Code Llama, StarCoder)
- Model sizes optimized for efficiency (Phi, Mistral)

### 2024: Multimodal & Reasoning Breakthrough
- **GPT-4o** achieved true multimodality
- **o1** introduced step-by-step reasoning
- **Claude 3** challenged GPT-4 dominance
- **Llama 3.1 405B** became largest open model
- **Gemini 1.5** pushed context limits to 2M tokens

### 2025: The Reasoning Revolution
- **Grok 3** achieved highest Arena scores
- **DeepSeek-R1** democratized reasoning capabilities
- **Llama 4** introduced 10M token contexts
- **Claude 4** set new coding standards
- **Qwen 3** pioneered hybrid reasoning modes

## Citation

If you find our survey useful for your research, please cite the following paper:

```bibtex
@article{hadi2024large,
  title={Large language models: a comprehensive survey of its applications, challenges, limitations, and future prospects},
  author={Hadi, Muhammad Usman and Al Tashi, Qasem and Shah, Abbas and Qureshi, Rizwan and Muneer, Amgad and Irfan, Muhammad and Zafar, Anas and Shaikh, Muhammad Bilal and Akhtar, Naveed and Wu, Jia and others},
  journal={Authorea Preprints},
  year={2024},
  publisher={Authorea}
}
```


## Model Organization Summary

### **By Company/Organization:**

**🔴 Proprietary Models:**
- **OpenAI**: GPT-4, GPT-4.5, GPT-4o, o1, o3, o4-mini, ChatGPT, InstructGPT
- **Anthropic**: Claude 3 Family, Claude 3.5, Claude 3.7, Claude 4, Anthropic LM
- **Google/DeepMind**: Gemini 2.5, Gemini 2.0, Gemini 1.5, PaLM 2, Bard, T5, UL2, Chinchilla, Sparrow, Gopher, GLaM, Minerva
- **xAI**: Grok 3, Grok 3 Mini
- **AI21 Labs**: Jurassic-1, Jurassic-2
- **Mistral AI**: Mistral 7B, Mistral Large 2, Mistral Medium

**🟢 Open Source Models:**
- **Meta**: Llama 4, Llama 3.x, Llama 2, OPT, Code Llama, Gallactica
- **Alibaba**: Qwen 3, Qwen 2.5, QwQ-32B
- **DeepSeek**: DeepSeek-R1, DeepSeek-V3
- **Microsoft**: Phi-3 Family, Phi-2
- **IBM**: Granite 3.0, Granite 3.1
- **Google**: Gemma 2
- **Cohere**: Command R, Command R+
- **BigScience**: BLOOM
- **EleutherAI**: GPT-J, GPT-NeoX, Pythia
- **BigCode**: StarCoder, StarChat, SantaCoder
- **Salesforce**: CodeGen2, CodeT5+, XGen
- **TIIUAE**: Falcon
- **Upstage**: SOLAR

**🎓 Academic/Research:**
- **LMSYS**: Vicuna, FastChat-T5
- **Stanford**: Alpaca
- **UC Berkeley**: Koala
- **LAION**: Open Assistant
- **OpenLM Research**: OpenLLaMA
- **MLFoundations**: OpenLM

**🏢 Other Companies:**
- **Yandex**: YaLM
- **Replit**: Replit Code
- **H2O.ai**: h2oGPT
- **Databricks**: Dolly
- **Together**: RedPajama-INCITE
- **MosaicML**: MPT Family
- **Stability AI**: StableLM
- **Nous Research**: OpenHermes
- **Cerebras**: Cerebras-GPT
- **Deci AI**: DeciCoder
- **AI Squared**: DLite
- **BlinkDL**: RWKV

### **By Model Type:**

**🧠 Reasoning Models (2024-2025):**
- OpenAI: o1, o1-mini, o3, o3-mini, o4-mini
- DeepSeek: DeepSeek-R1 Family
- Alibaba: QwQ-32B, Qwen 3 (hybrid reasoning)
- Google: Gemini 2.5 (thinking models)

**💬 Conversational Models:**
- OpenAI: ChatGPT, GPT-4o
- Anthropic: Claude 3/4 Family
- Google: Bard, Gemini
- xAI: Grok 3

**💻 Code-Specialized:**
- Meta: Code Llama
- BigCode: StarCoder, SantaCoder
- Salesforce: CodeGen2, CodeT5+
- Replit: Replit Code
- Deci AI: DeciCoder

**🌐 Multimodal:**
- OpenAI: GPT-4o
- Google: Gemini 2.0/2.5
- Meta: Llama 4, Llama 3.2

**⚡ Efficient/Small:**
- Microsoft: Phi-3 Family, Phi-2
- Google: Gemma 2
- AI Squared: DLite
- Upstage: SOLAR

---

*Last updated: July 2025*  
*Original repository: https://www.techrxiv.org/doi/full/10.36227/techrxiv.23589741.v3*
