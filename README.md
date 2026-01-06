# Geometric Gating: A Hybrid Architecture for Logical Reasoning in RAG and Agents
**Bridging the Gap Between Vector Semantics and Boolean Logic**

## Abstract
Current Large Language Models (LLMs) rely on Euclidean embedding spaces, where relationships are defined by vector similarity. While this is excellent for semantic association, it is mathematically ill-suited for strict logical operations (Negation, Disjunction, Entailment). This leads to well-known failure modes in Retrieval-Augmented Generation (RAG) and Agentic Tool Use, such as the inability to strictly filter out specific topics or the failure to respect "NOT" constraints.

We propose a **Hybrid Neuro-Symbolic Architecture** that runs two parallel embedding streams: **Standard Vectors** for semantic nuance and **Box Embeddings** for logical gating. We demonstrate that projecting queries into geometric hyper-rectangles allows for the enforcement of hard Boolean constraints that standard attention mechanisms cannot violate, regardless of feature magnitude.

---

## 1. The Geometry Gap
In modern AI, there is a fundamental tension between **Semantics** and **Logic**.

| Feature | Euclidean Space (Standard LLMs) | Geometric Space (Box Embeddings) |
| :--- | :--- | :--- |
| **Metric** | Cosine Similarity (Angle) | Volume Intersection (Overlap) |
| **Logic** | Additive (Soft) | Multiplicative (Hard) |
| **Strength** | Fuzzy matching, synonyms, vibe | Hierarchy, Entailment, Negation |
| **Failure Mode** | **Magnitude Trap:** Context overrides Logic | **Convexity:** Cannot naturally do "OR" |

### 1.1 The "Magnitude Trap"
Our experiments demonstrate that standard attention acts as an additive filter. If a document contains a logical violation (e.g., "NOT Apple") but matches the query on 99 other semantic dimensions (Laptop, Screen, Keyboard, Price), the positive magnitude of the semantic matches overwhelms the negative signal of the logical mismatch. The model retrieves the document despite the explicit constraint.

---

## 2. The Solution: Hybrid Geometric Gating

We do not propose replacing Token Embeddings. Instead, we introduce a **Geometric Gating Layer** that operates alongside standard retrieval or generation heads.

### 2.1 Architecture Overview
The system processes a Query $Q$ and a set of Candidates $K$ (documents or tools) through two heads:

1.  **Semantic Head ($H_{sem}$):** Calculates standard Dot-Product Attention.
    $$ S_{sem} = \text{Softmax}(Q_{sem} \cdot K_{sem}^T) $$
2.  **Logic Head ($H_{box}$):** Projects inputs into Hyper-rectangles (Boxes) and calculates Volumetric Intersection.
    $$ S_{logic} = \frac{\text{Vol}(\text{Box}_Q \cap \text{Box}_K)}{\text{Vol}(\text{Box}_K) + \epsilon} $$

### 2.2 The Gating Mechanism
The final attention score is the product of the semantic score and the logical gate.

$$ \text{Attention}(Q, K) = S_{sem}(Q, K) \times \text{Gate}(S_{logic}(Q, K)) $$

*   If $S_{logic} > 0$ (Logical Consistency), the semantic score passes through.
*   If $S_{logic} = 0$ (Logical Violation), the semantic score is annihilated.

This creates a **Hard Guardrail**: No matter how semantically similar a document is, if it is logically disjoint from the query constraints, it cannot be retrieved.

---

## 3. Key Applications

### 3.1 Hard-Negation RAG
**User Query:** *"Find nutritional info for fruits, but NOT citrus."*

*   **Standard Vector RAG:** "Citrus" is semantically close to "Fruit". "Orange" is semantically close to "Nutrition". Vector similarity retrieves "Orange Nutrition Facts."
*   **Hybrid RAG:**
    *   Query Box $Q$ = $\text{Box}(\text{Fruit}) \cap \text{Disjoint}(\text{Citrus})$.
    *   Document $D$ ("Orange") falls inside $\text{Box}(\text{Fruit})$ but also inside $\text{Box}(\text{Citrus})$.
    *   The Disjointness constraint sets Intersection Volume to 0.
    *   **Result:** The document is geometrically masked before the LLM ever sees it.

### 3.2 Compositional Tool Routing
**User Query:** *"Calculate the mortgage rate, or search for current rates if the calculator fails."*

Logic: `(Tool:Calculator) OR (Tool:Search)`

*   **Box Solution:** We use **Multi-Head Box Attention** to handle Disjunction (OR).
    *   Head 1 projects Query to `Calculator` region.
    *   Head 2 projects Query to `Search` region.
    *   The union of these heads captures the user intent, strictly excluding `Tool:Weather` or `Tool:Chat`.

### 3.3 Safety Guardrails
Instead of training a binary classifier on "Toxic" vs "Safe" (which is brittle), we define safety policies as **Forbidden Regions** in the embedding space.
*   Policy: $P = \text{Box}(\text{Medical Advice}) \cap \text{Box}(\text{Unverified})$.
*   If a model output falls into this intersection volume, the Logic Gate closes, preventing the response.

---

## 4. Proof of Concept: Boolean Completeness

We demonstrate that this architecture is **Boolean Complete**, capable of handling AND, OR, and NOT operations simultaneously.

### The Logic Trap Experiment
**Query:** `(A OR B) AND (NOT C)`
**Scenario:** "Fruit OR Veg, but NOT Red."

| Candidate | Logic Profile | Hybrid Score | Status |
| :--- | :--- | :--- | :--- |
| **Banana** | Fruit, Yellow | **High** | ✅ PASS (via Head A) |
| **Spinach** | Veg, Green | **High** | ✅ PASS (via Head B) |
| **Apple** | Fruit, **Red** | **Zero** | ❌ BLOCKED (via Negation Gate) |
| **Car** | Machine, Red | **Zero** | ❌ BLOCKED (Irrelevant) |

*Full reproduction code is available in `hybrid_logic_poc.py`.*

---

## 5. Technical Implementation (Pseudocode)

```python
class HybridRAG(nn.Module):
    def forward(self, query_text, documents):
        # 1. Semantic Stream (Standard Transformers)
        sem_q = self.bert(query_text)
        sem_docs = self.bert(documents)
        semantic_scores = torch.matmul(sem_q, sem_docs.T)
        
        # 2. Logic Stream (Box Embeddings)
        # Project text to Box Parameters (Min, Delta)
        box_q = self.box_proj(query_text) 
        box_docs = self.box_proj(documents)
        
        # Calculate Intersection Volume
        inter_vol = get_intersection_vol(box_q, box_docs)
        
        # 3. Gating
        # If intersection is 0 (Logical mismatch), Gate is 0.
        # If intersection is high (Entailment), Gate is 1.
        logic_gate = torch.clamp(inter_vol, 0, 1)
        
        # 4. Final Retrieval Score
        return semantic_scores * logic_gate
```

---

## 6. Conclusion
Box Embeddings are not a replacement for Token Embeddings; they are a necessary complement. By separating the **fuzzy, creative** aspects of language (Vectors) from the **strict, hierarchical** aspects of reasoning (Boxes), we can build Hybrid Architectures that are both fluent and logically robust.

This approach offers a practical path forward for industrial AI systems that require strict adherence to constraints in RAG, Tool Use, and Safety.
