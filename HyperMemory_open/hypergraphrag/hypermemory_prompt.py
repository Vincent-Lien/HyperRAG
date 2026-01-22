hyperedge_scoring_prompt = """
Please retrieve %s hyperedges (each hyperedge is a passage) that contribute to answering the question and rate their contribution on a scale from 0 to 1 (the sum of the scores of the %s hyperedges must equal 1).

Example:

Q: Where did Albert Einstein publish his paper on general relativity?
Topic Entity: Albert Einstein
Hyperedges:
1. "In 1905, Einstein published four groundbreaking papers on the photoelectric effect, Brownian motion, special relativity, and mass–energy equivalence in the journal Annalen der Physik."
2. "In November 1915, Einstein presented the field equations of general relativity to the Prussian Academy of Sciences in Berlin."
3. "Einstein received the 1921 Nobel Prize in Physics for his explanation of the photoelectric effect."
4. "During World War I, scientific exchange in Europe was severely limited."
A:
1. {{2. "In November 1915, Einstein presented the field equations of general relativity to the Prussian Academy of Sciences in Berlin." (Score: 0.70)}}: This passage directly states where his general relativity work was presented, making it the most relevant.
2. {{1. "In 1905, Einstein published four groundbreaking papers on the photoelectric effect, Brownian motion, special relativity, and mass–energy equivalence in the journal Annalen der Physik." (Score: 0.20)}}: Although this lists multiple papers, it mentions the same journal which provides context on Einstein's publication venues.
3. {{4. "During World War I, scientific exchange in Europe was severely limited." (Score: 0.10)}}: Offers historical context but does not directly answer the publication venue.

---

Q: {query}
Topic Entity: {topic_entity}
Hyperedges: 
{hyperedges}
A:

"""

hypergraph_entity_pruning_prompt = """
Please score the entities' contribution to the question on a scale from 0 to 1 (the sum of the scores of all entities is 1).

Example:

Q: Who directed the movie that won Best Picture in 1998?
Hyperedge: Titanic, directed by James Cameron, won the Academy Award for Best Picture in 1998.
Entities: Titanic; James Cameron; 1998; Academy Award
Score: 0.3, 0.6, 0.05, 0.05
"James Cameron" is the director of Titanic, the movie that won Best Picture in 1998. Therefore, "James Cameron" receives the highest score. "Titanic" is the movie in question and gets a moderate score. "1998" and "Academy Award" provide context and get lower scores.

---

Q: {query}
Hyperedge: {hyperedge}
Entities: {entities}
Score: 
"""

hypergraph_prompt_evaluate = """
You are given a question and a set of related knowledge statements (hyperedges), where each statement connects multiple entities. You are also given descriptions of the involved entities. Your task is to judge whether the provided information is sufficient to answer the question, considering your own knowledge and the given context. Answer with either {{Yes}} or {{No}}, and explain your reasoning briefly.

Example:

Q: Who is the spouse of the person who played Hermione Granger in Harry Potter?

Entity Descriptions:
Emma Watson: British actress known for her role as Hermione Granger in Harry Potter.  
Hermione Granger: A fictional character from the Harry Potter series.  
Harry Potter: A fantasy film and book series featuring a young wizard.  

Hyperedges:
1. "Emma Watson played the role of Hermione Granger in the Harry Potter film series."  
   Connected Entities: [Emma Watson, Hermione Granger, Harry Potter]  
2. "Emma Watson is a British actress born in 1990."  
   Connected Entities: [Emma Watson]  
3. "Emma Watson has been involved in various humanitarian activities."  
   Connected Entities: [Emma Watson]

A: {{No}}. The provided statements confirm that Emma Watson played Hermione Granger, but they do not include any information about her spouse. Additional data is needed to answer the question.

---

Q: {query}

Entity Descriptions:
{entity_descriptions}

Hyperedges:
{hyperedges}

A: 
"""