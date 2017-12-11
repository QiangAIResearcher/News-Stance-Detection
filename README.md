# News-Stance-Detection

Based on the Emergent data, this project aims to detect stance relationship between a headline and the correspondding article body. The stance includes "agree", "disagree" and "discuss". 

We clean the Emergent dataset from William Ferreira https://github.com/willferreira/mscproject to construct our dataset. Basically, the Emergent dataset studies the relationship of a claim and an article. Several articles were collected by William from the Internet to describe a claim. A headline was summerized from each collected article. These articles may be "for", "against", or "observing" the claim.

Some headline/body pairs in FNC-1 https://github.com/FakeNewsChallenge/fnc-1 are fabricated by mixing headlines and bodies. Because these pairs don't exist in the real Internet world, we want to consrtuct our dataset instead of using the FNC-1 dataet so that we can draw information from Internet user interaction. We adopt the same logic with the organizers of FNC-1 (Dean Pomerleau et al.) to construct our dataset. That is, for a given headline/body pair, if both the headline and the body were labelled (by William Ferreira et al.) as "for" (or both "against") with central claim, the headline and body pair are said to "agree" with each other. If headline was determined to "against" the central claim, and the body was determined to "for" the central claim, then the headline and the body pair are said to "disagree" with each other. The same with visa versa (i.e. headline = "for" and body = "disagree") - the pair would be labelled a "disagree" pair. If either the headline or the body (or both) in a pair were labelled as "observing" relative to the central claim, then the pair is labelled as "discuss" in our dataset.

In William Ferreira's project, there are totally 7112 headline/body pairs, among which valid stances("for", "against", "observing") count for 5059. Further, 58 pairs have missing headlines or bodies and 2398 pairs were found to have the same "articleid"(article body) with other pairs. Finally, we obtaine 2603 headline/body pairs. Among the 2603 pairs, 1162 pairs are labelled as "agree" with each other according to the abovementioned rules, 17 are labelled as "disagree" and 1424 are labelled as "discuss".

Some statistics
    
    Number of total headline/body pairs in Emergent dataset: 7112
        
        Number of invalid stances (blank or "ignoring"): 2053
        
        Number of missing headlines or bodies: 58
        
        Number of pairs with the same "articleid"(article body): 2398
        
        Number of valid pairs (our dataset): 2603
            
            Number of "agree" pairs: 1162
            
            Number of "disagree" pairs: 17
            
            Number of "discuss" pairs: 1424

