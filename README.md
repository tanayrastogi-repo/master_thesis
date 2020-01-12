# Load Identification from Aggregated Data using Generative Modeling 
#### By Tanay Rastogi
##### Master Thesis in System Control and Robotics
##### Date:             Feb, 17, 2019
##### KTH Supervisor:   Dr. Pawel Herman (paherman@kth.se)
##### KTH Examiner:     Dr. Hedvig Kjellström (hedvig@csc.kth.se)
##### School of Electrical Engineering and Computer Science (EECS)
##### The project undertaken in Zyax AB (https://www.zyax.se/)

# Abstract
In  the  view  of  an  exponential  increase  in  demand  for  energy,  thereis a need to come up with a sustainable energy consumption systemin residential buildings.  Several pieces of research show that this canbe achieved by providing real-time energy consumption feedback ofeach  appliance  to  its  residents.   This  can  be  achieved  through  Non-Intrusive Load Monitoring System (NILM) that disaggregates the elec-tricity consumption of individual appliances from the total energy con-sumption of a household.   The state-of-art NILM have several chal-lenges that preventing its large-scale implementation due to its lim-ited applicability and scalability on different households.  Most of theNILM  research  only  trains  the  inference  model  for  a  specific  housewith a limited set of appliances and does not create models that cangeneralize appliances that are not present in the dataset. In this Masterthesis, a novel approach is proposed to tackle the above-mentioned is-sue in the NILM. The thesis propose to use a Gaussian Mixture Model(GMM) procedure to create a generalizable electrical signature modelfor each appliance type by training over labelled data from differentappliances of the same type and create various combinations of appli-ances by merging the generated models. Maximum likelihood estima-tion method is used to label the unlabeled aggregated data and disag-gregate it into individual appliances.  As a proof of concept, the pro-posed algorithm is evaluated on two datasets, Toy dataset and ACS-F2 dataset, and is compared with a modified version of state-of-the-artRNN  network  on  ACS-F2  dataset.   For  evaluation,  Precision,  Recalland  F-score  metrics  are  used  on  all  the  implementations.   From  theevaluation, it can be stated that the GMM procedure can create a gen-eralizable appliance signature model, can disaggregate the aggregateddata  and  label  previously  unseen  appliances.   The  thesis  work  alsoshows that given a small set of training data, the proposed algorithmperforms better than RNN implementation.   On the other hand,  theproposed algorithm highly depends on the quality of the data. The al-gorithm also fails to create an accurate model for appliances due to thepoor initialization of parameters for the GMM. In addition,  the pro-posed algorithm suffers from the same inaccuracies as the state of art.

# Link to the Thesis
http://www.diva-portal.org/smash/record.jsf?pid=diva2:1304677

