# spatial-KB
Working implementation of the commonsense spatial reasoner presented in 
[Chiatti et al., 2021](https://arxiv.org/abs/2104.00387), which extends  [SEMAP by Deeken et al. (2018)](https://www.sciencedirect.com/science/article/pii/S0921889017306565).
.

## Dependencies
* PostGRE with PostGIS and SFCGAL extensions  
* ConceptNet 5 installed locally for faster inference
* Wordnet API as exposed through the NLTK library 
* (optional) full Visual Genome 1.4 relationship set and predicate aliases
