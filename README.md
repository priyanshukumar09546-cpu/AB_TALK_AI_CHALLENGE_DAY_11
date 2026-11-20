# Day 11: Build Your First Document Retrieval System

**Challenge:** 60 Days AI Engineering Coding Challenge  
**Focus Area:** Information Retrieval  
**Tools:** Python, scikit-learn, NumPy  

---

## 1. System Overview
This project implements a TF-IDF document retrieval engine built on top of the modular `Pipeline` class from Day 10. It indexes a **22-document knowledge base** covering Artificial Intelligence and Space Exploration, computes cosine similarity scores using NumPy and scikit-learn, and enforces a **relevance threshold of `0.1`** to reject out-of-domain queries rather than returning hallucinated or low-quality matches.

---

## 2. Retrieval Failure Diagnosis (Query-by-Query Analysis)

During evaluation across 10 test queries (standard, synonym-based, ambiguous, and out-of-domain), the following failures or suboptimal rankings occurred:

1. **Query (`Synonym Test 5`): `"How do intelligent machines learn from annotated examples?"`**
   * **Diagnosis:** Failed to retrieve Document 0 (*"Supervised learning trains machine learning models using labeled input-output pairs"*) as top result because TF-IDF relies on exact lexical overlap and cannot map `"annotated examples"` to `"labeled input-output pairs"`.

2. **Query (`Synonym Test 6`): `"Celestial bodies circling stars beyond our sun"`**
   * **Diagnosis:** Returned `"No relevant document found"` (score `< 0.1`) instead of Document 16 (*"Exoplanets are planets orbiting stars outside our solar system"*) because synonyms `"celestial bodies"`, `"circling"`, and `"beyond"` share zero token overlap with `"exoplanets"`, `"orbiting"`, and `"outside"`.

3. **Query (`Ambiguous 1`): `"How does Apple train models or grow?"`**
   * **Diagnosis:** Retrieved Document 21 (*"Apple orchards require cold winter dormancy periods..."*) ahead of AI model-training documents because `"apple"` has a very high Inverse Document Frequency (IDF) weight compared to common corpus words like `"train"` and `"model"`.

4. **Query (`Ambiguous 2`): `"Light and space observation"`**
   * **Diagnosis:** Ranked Document 15 (*Black holes... light*) nearly equal to Document 13 (*James Webb Space Telescope observes...*) because TF-IDF treats bag-of-words independently without understanding whether the user is asking about astronomical telescopes or gravitational light trapping.

---

## 3. How Vocabulary Mismatch Causes TF-IDF Failure & Why Embeddings Are Needed

### Why TF-IDF Fails on Synonyms
TF-IDF represents documents and queries as **sparse, high-dimensional orthogonal vectors** where every unique word is treated as a completely independent dimension. It relies strictly on **exact lexical matching** (surface-level character/token overlap):
* If a document states *"labeled datasets"* and a user queries *"annotated examples"*, their dot product is **zero** (`cosine_similarity = 0.0`), even though both phrases express the exact same concept.
* Similarly, morphological or domain synonyms (*"exoplanets"* vs. *"celestial bodies"*, *"reusable boosters"* vs. *"reusable rockets"*) suffer severe score penalties whenever exact terms are missing.

### What This Tells Us About the Need for Dense Embeddings
To overcome vocabulary mismatch, modern retrieval systems replace sparse TF-IDF vectors with **dense neural embeddings** (such as Sentence-BERT, OpenAI `text-embedding-3`, or BGE):
* **Semantic Proximity:** Dense embeddings map words and sentences into a continuous vector space where semantically similar concepts (*"annotated"* and *"labeled"*) are positioned close together regardless of exact spelling.
* **Context Awareness:** Unlike bag-of-words TF-IDF, transformer-based embeddings distinguish polysemous words (e.g., *"Apple"* the fruit vs. *"Apple"* the tech company) based on surrounding sentence context.
