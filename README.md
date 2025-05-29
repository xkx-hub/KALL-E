# KALL-E 
<div style="display: flex;">
   <img src="./figures/system_overview.jpg" style="margin-right: 20px;width: 80%; height: auto;"/>
</div>

## News

- **Our inference code has been released**

## Key Features

- **Autoregressive Language Modeling**: Utilizes an autoregressive approach for next-distribution prediction in text-to-speech synthesis.
- **Continuous Speech Distribution**: Directly models and predicts continuous speech distributions conditioned on text, avoiding reliance on diffusion-based components.
- **FlowVAE**: Employs FlowVAE to extract continuous speech distributions from waveforms, rather than using discrete speech tokens.
- **Single AR Language Model**: Uses a single autoregressive language model to predict continuous speech distributions from text, constrained by Kullback-Leibler divergence loss.
- **Simplified Paradigm**: Offers a more straightforward and effective approach for using continuous speech representations in TTS.

## Performance

Results for 4 target speakers on the seedstts-eval test set

| target_speaker                                   | zh          |            | en   |            |
|--------------------------------------------------|-------------|------------|------|------------|
|                                                  | cer         | sim        | wer  | sim        |
| --                                               | 0.95        | --         | 1.68 | --         |
| 70                                               | --          | --         | 2.25 | 0.701      |
| 159                                              | --          | --         | 2.40 | 0.733      |
| aishell3-SSB0341                                 | 0.95        | 0.710      | --   | --         |
| didispeech-00010111                              | 1.02        | 0.750      | --   | --         |
