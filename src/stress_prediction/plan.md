Given the high quality of your training data (2.8 million forms with morphological enrichment via spaCy/Stanza) and the specific linguistic rules identified in the research papers, a **Hybrid Transformer-based Sequence Labeling** approach is significantly more effective than lightgbm for Ukrainian.

lightgbm fails (76%) because it struggles with **long-range character dependencies** (like suffixes interacting with prefixes) and treating words as atomic strings rather than sequences of phonemes/syllables.

### Recommended Approach: Multi-Task Transformer with Rule-Based Feature Injection

The ideal architecture is a **Char-level Transformer** (specifically a **Bi-LSTM-CRF** or a lightweight **Transformer Encoder**) that treats stress prediction as a sequence labeling task (predicting a binary 0/1 for each syllable or character).

#### 1. Data Transformation (The "Syllable-Centric" View)

Instead of predicting a single index for the whole word, transform your data so each character is a timestep.

- **Input:** `[п, о, м, и, л, к, а]`
- **Target:** `[0, 1, 0, 0, 0, 0, 0]` (for _помИлка_) or `[1, 0, 0, 0, 0, 0, 0]` (for _пОмилка_)
- **Context Features:** Concatenate POS and Morphological features (Gender, Case, Number) as embedding vectors to every character in the word.

#### 2. Hybrid Feature Injection (Integrating the Rules)

You can bake the "Expert Rules" into the ML model using **Feature Engineering** or **Constrained Inference**:

- **Rule-based Feature Vectors:** For every character, add a binary flag if it belongs to a "Strong Suffix" (e.g., `-ач`, `-яр`, `-ист`). If a character is part of `-еньк-`, the rule weight is 1.0, so the model receives a strong signal.
- **Interfix Detection:** Specifically flag `-лог` and `-граф` sequences.
- **Weight Integration:** In your loss function, increase the penalty for mispredicting words that fall under "high-weight" rules (Weight > 0.9).

#### 3. Model Architecture

I recommend a **ByT5-style** or **CANINE-style** architecture (Character-level) because it avoids OOV (Out-Of-Vocabulary) issues common in inflected languages like Ukrainian.

- **Encoder:** Character CNN + Bi-LSTM or a small 4-layer Transformer.
- **Conditioning:** Use "Feature-wise Linear Modulation" (FiLM) or simple concatenation to inject the spaCy POS/Morph tags into the hidden states.
- **Output Layer:** A Conditional Random Field (CRF) layer to ensure that the model doesn't predict two primary stresses in a single word (unless it's a compound).

### Machine-Readable Implementation Strategy (Python/PyTorch)

```python
import torch
import torch.nn as nn

class UkrainianStressModel(nn.Module):
    def __init__(self, char_vocab_size, morph_vocab_size):
        super().__init__()
        # Char Embeddings for the word structure
        self.char_emb = nn.Embedding(char_vocab_size, 64)

        # Morphological Embeddings (Case, Gender, POS)
        self.morph_emb = nn.Embedding(morph_vocab_size, 32)

        # Rule-based injection (Weights for suffixes like -еньк, -лог)
        self.rule_projection = nn.Linear(1, 16)

        # Sequence Processor
        self.lstm = nn.LSTM(64 + 32 + 16, 128, bidirectional=True, batch_first=True)
        self.classifier = nn.Linear(256, 1) # Probability of stress on this char

    def forward(self, chars, morph_tags, rule_signals):
        c_feat = self.char_emb(chars)
        m_feat = self.morph_emb(morph_tags).repeat(1, chars.size(1), 1)
        r_feat = self.rule_projection(rule_signals)

        x = torch.cat([c_feat, m_feat, r_feat], dim=-1)
        output, _ = self.lstm(x)
        return torch.sigmoid(self.classifier(output))

```

### Why this beats lightgbm:

1. **Suffix Awareness:** Transformers/LSTMs naturally "see" the `-ар`, `-ач`, `-ак` endings that the statistical research (Shaboldov) highlights as high-probability stress carriers.
2. **Morphological Disambiguation:** By passing spaCy features as embeddings, the model can distinguish between _зáмок_ (Noun, Masc) and _замóк_ (Noun, Masc) if you include the `Definition_ID` or `Sense_Cluster` from your DB.
3. **Handling Homonyms:** Your DB identifies `grammatical_homonym` and `free_variant`. You should train the model using **Label Smoothing** on `free_variants` (assigning 0.5 probability to both vowels) and hard 1.0/0.0 on `single` patterns.

### Next Steps:

1. **Syllabification:** Pre-process your 2.8M words into syllables. Predicting stress per syllable is easier than per character (3-5 units vs 10-15 units).
2. **Rule-based Pre-labeling:** Use the JSON rules I provided earlier to create a "Rule Score" column in your SQLite DB. Use this as an additional input feature.
3. **Loss Weighting:** Use the `pos_confidence` from your export script to weight the importance of each sample during training. Low confidence spaCy tags = lower loss weight.
