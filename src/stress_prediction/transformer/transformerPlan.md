### Phase 1: The "Expert-Guided" Feature Engineering

Instead of letting the model guess the rules, we will "hard-code" the logic from your `rules.json` into the input data.

**The Character-Level Feature Vector:**
For every character (e.g., in the word "бджоляр"), we create a vector containing:

1. **Character ID:** (The letter 'я').
2. **Vowel Flag:** (1 if vowel, 0 if consonant).
3. **Reverse Position:** (Linguists in `recommendations.md` say counting from the end is better. We give each character its index from the end: `[р=0, я=1, л=2...]`).
4. **Morphology Injection (from Stanza/LMDB):** The POS tag (`NOUN`) and features (`Gender=Masc`, `Number=Sing`).
5. **Rule Match Flag:** If a character is part of a suffix identified in `rules.json` (like `-ар`, `-яр`, `-ач`), we toggle a specific "Rule_Active" bit for those characters.

- _Example:_ In "бджоляр", the characters `я` and `р` get a `Rule_B_Mobile=1` flag because they match rule `NOUN-01`.

---

### Phase 2: Architecture — The "Morpho-Transformer"

With your **RTX 4060 Ti 16GB**, you should build a **Transformer Encoder with Feature Concatenation.**

- **Embedding Layer:** We create separate "embedding tables" for Characters, POS tags, and Rule IDs. We "glue" (concatenate) them together.
- **Local Attention (Windowed):** Since Ukrainian stress is often determined by the suffix (`recommendations.md`), we use attention heads that look closely at the end of the word.
- **Global Attention:** One head should look at the relationship between the prefix and the suffix to catch compound word logic (interfixes like `-о-` from your compound rules).
- **The Classifier:** The final layer predicts a probability (0.0 to 1.0) for every vowel.

---

### Phase 3: The "Scientific" Training Strategy

We will use your `rules.json` not just for features, but to **punish the model** when it ignores linguistic reality.

1. **Rule-Weighted Loss:** \* In `rules.json`, you have a `weight` for each rule (e.g., `1.0` for numerals in `-адцять`).

- If the model predicts the wrong stress on "одинадцять", we multiply the "penalty" (Loss) by that weight. This forces the model to prioritize learning the hard rules first.

2. **Variant Handling (Label Smoothing):**

- For `free_variant` words (пОмилка/помИлка from your `README.md`), we don't tell the model one is "wrong." We set the target to `0.5` for both vowels. This prevents the model from getting "confused" and keeps the math stable.

3. **The "Shaboldov" Penalty:**

- Rule `NOUN-01` mentions that `-ар/-яр` suffixes are oxytone (stress on the end) in 81.9% of cases. We can use this percentage as a "prior" probability for the model.

---

### Phase 4: Hardware-Optimized Execution (The "Pragmatic" Part)

Since you are learning ML on a **Ryzen 7 / RTX 4060 Ti** setup:

- **Framework:** Use **PyTorch**. It is better than TensorFlow for this kind of "custom feature glueing."
- **Batching:** Set your batch size to **512 or 1024**. Your 16GB of VRAM can handle this easily, and it will make the training much more stable than the lightgbm trials.
- **Validation:** Use the `split` column from your `stress_training.db`. Ensure you test the model _only_ on the `test` split to see if it actually learned the "Logic of new words" (Morphology) or just memorized the dictionary.

---

### The Updated Inference Logic

Your final service becomes a **Neuro-Symbolic Hybrid**:

1. **Step 1 (Symbolic):** Check LMDB. If found and `variant_type == 'single'`, return answer. **(0ms latency)**.
2. **Step 2 (Neural Fallback):** If word is OOV or a `grammatical_homonym`:

- Run Stanza to get POS/Feats.
- Scan word for `rules.json` patterns.
- Run the Transformer. **(~10-20ms latency on GPU)**.

**Does this "Logic-First" Transformer approach (using the rule weights and suffix flags) feel more like the "Logical Prediction" you were looking for?** I can help you write the Python code to "flag" the suffixes from your `rules.json` if you're ready for that.
