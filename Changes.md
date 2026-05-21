**What was changed and Why:**
1. **The Multi-Stage Triage (The "Triage Funnel")**
***Change:***Instead of sending every tweet to the heavy BART model, the new script uses a "Funnel" approach.
***Why***: In digital forensics, speed and volume are huge issues.
Stage 1 (DistilBERT) is a tiny, lightning-fast model. It removes "Positive" posts immediately.
*Result:* Posts were analyzed faster because the heavy AI was only used for the "suspicious" leftovers. This reflects a professional investigator's need for efficiency.
2. **Slang Normalization**
*Change*: Added a SLANG_MAP and updated clean_text to swap words like "wicked" for "excellent."
*Why*: AI models are often trained on formal text (books/news). They often interpret social media slang literally.
*Example*: In the original results, "Wicked!!" was a threat because the AI thought "Wicked = Evil."
*New Result*: By changing it to "excellent," the Sentiment stage catches it as "Positive," and the post is safely ignored.
3. **Hypothesis Template**
*Change:* Added a more clear hypothesis, the deafult was likely "This post contains {}" and has been changed to "This text expresses a specific intent of {}."
*Why:* The original prompt was too broad. If a post mentioned a "bad bus," the AI thought the post "contained" something "threatening." By adding the word "intent," you are forcing the AI to look for a human actor with a goal, which is the standard for legal evidence in forensics.
4. **Keyword Anchoring** 
*Change:* Added a list of CRITICAL_KEYWORDS (kill, gun, bomb, etc.) and a logic check.
*The Logic*: If the AI says a post is "CRITICAL" but the post doesn't contain a single violent word, the script "vetoes" the AI and downgrades the post to MEDIUM.
*Result*: This stopped the "Smelly Bus" from being flagged as a "Critical Threat." It's still "Medium" (suspicious/annoying), but it won't trigger an emergency alert.