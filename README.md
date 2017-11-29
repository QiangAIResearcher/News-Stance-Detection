# News-Stance-Detection

Based on the Emergent data, this project aims to detect stance relationship between a headline and the correspondding article body. The stance includes "agree", "disagree" and "discuss". 

We clean the Emergent dataset from William Ferreira https://github.com/willferreira/mscproject to construct our dataset. Basically, the Emergent dataset studies the relationship of a claim and an article. Several articles were collected by William from the Internet to describe a claim. A headline was summerized from each collected article. These articles may be "for", "against", or "observing" the claim.

Some headline/body pairs in FNC-1 are fabricated by mixing headlines and bodies. Because these pairs don't exit in real Internet world, we want to consrtuct our dataset instead of using the FNC-1 dataet so that we could draw information from internet user interaction. We adopt the same logic with the organizers (Dean Pomerleau et al.) of FNC-1 to construct our dataset. That is, for a given headline/body pair, if both the headline and the body are labelled (by William Ferreira et al.) as "for" (or both "against") with central claim, the headline and body pair were said to "agree" with each other. If headline was determined to "against" the central claim, and the body was determined to "for" the central claim, then the headline and the body pair were said to "disagree" with each other. The same with visa versa (i.e. headline = "for" and body = "disagree") - the pair would be labelled a "disagree" pair. If either the headline or the body (or both) in a pair were labelled as "observing" relative to the central claim, then the pair was labelled as "discuss" in our dataset.

In William Ferreira's project, there are totally 7112 headline/body pairs, among which valid stances("for", "against", "observing") count for 5059. Further, when we convert the text to vectors by Word2Vec and calculate the cosine distance of each pair, 68 pairs have NaN for the cosine distance. Finally, we obtaine 4991 headline/body pairs. Among the 4991 pairs, 2138 pairs are labelled as "agree" with each other according to the abovementioned rules, 35 are labelled as "disagree" and 2818 are labelled as "discuss".

Some statistics
    
    Number of total headline/body pairs in Emergent dataset: 7112
        
        Number of invalid stances: 2053
        
        Number of invalid cosine distance: 68
        
        Number of valid pairs: 4991
            
            Number of "agree" pairs: 2138
            
            Number of "disagree" pairs: 35
            
            Number of "discuss" pairs: 2818

