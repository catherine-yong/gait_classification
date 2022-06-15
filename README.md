# Gait classification
# Projet au sein du stage de 4A à Capgemini

Nous cherchons à définir la tranche d'âge d'une personne ainsi que son sexe à partir de sa démarche.

Code basé sur https://github.com/AbnerHqC/GaitSet 

Article associé : https://www.researchgate.net/publication/328997464_GaitSet_Regarding_Gait_as_a_Set_for_Cross-View_Gait_Recognition

## pretreatment.py : 
Pour exécuter : 
python pretreatment.py --input_path='CASIA-B' --output_path='output_pretreatment_B'
(attention il faudra supprimer le dossier output_pretreatment_B)

Ce qui a été modifié par rapport au code sur le git de AbnerHqC : 
utilisation de imageio.imwrite au lieu de scisc.imsave 

## config.py : 

## train.py : 

## test.py : 



## Dataset : 
CASIA-B : http://www.cbsr.ia.ac.cn/english/Gait%20Databases.asp

## A faire
- Comment estimer la taille de quelqu'un avec une caméra classique ? 
- Est-ce qu'une caméra classique suffit ? 
- Comment transformer une photo/vidéo en "silhouette", en format comme dans CASIA-B ?
- Interface graphique à faire 
